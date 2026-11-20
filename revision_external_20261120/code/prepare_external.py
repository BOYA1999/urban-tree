from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import array_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, transform
from shapely.geometry import Point, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "revision_20261120" / "code"))
from local_reference import COUNT, STEP, SUB, bounds_grid
from neighbourhood import DATA, EQ

CACHE = ROOT / "data" / "cache"
PARIS_SERVICE = "https://carto2.apur.org/apur/rest/services/OPENDATA/HAUTEUR_VEGETATION_2021/ImageServer"
CITY_IDS = {"london2020_22": 5816, "paris2021": 2878, "melbourne2021": 1747, "saopaulo2017": 7277}


def burn(geometries, tr, shape):
    geometries = [geom for geom in geometries if geom is not None and not geom.is_empty]
    if geometries:
        return rasterize(geometries, out_shape=shape, transform=tr, dtype="uint8")
    return np.zeros(shape, dtype=np.uint8)


def pixel_counts(field):
    h, w = field.shape
    return field.reshape(h // SUB, SUB, w // SUB, SUB).sum(axis=(1, 3), dtype=np.uint16)


def windows(height, width):
    for r in range(0, height, 100):
        for c in range(0, width, 100):
            h, w = min(height - r, 100), min(width - c, 100)
            yield (slice(r, r + h), slice(c, c + w)), Window(c * SUB, r * SUB, w * SUB, h * SUB)


def london_data():
    directory = CACHE / "london_canopy"
    with ZipFile(CACHE / "london_canopy.zip") as z:
        for n in z.namelist():
            dest = (directory / n).resolve()
            if dest.is_relative_to(directory.resolve()) and not n.startswith("__MACOSX/"):
                z.extract(n, directory)
    boundary_file = CACHE / "london_bounds.zip"
    layer = "statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"
    boundary = pyogrio.read_dataframe(f"zip://{boundary_file.as_posix()}!{layer}", columns=[])
    tiles = [(p, pyogrio.read_info(p)) for p in sorted(directory.rglob("TreeCanopyCover*.shp"))]
    assert len(tiles) == 467
    tiles = [(p, v["total_bounds"]) for p, v in tiles if v["features"]]

    def canopy_for(bounds):
        native = transform_bounds(EQ, "EPSG:27700", *bounds, densify_pts=21)
        west, south, east, north = native
        for path, (w, s, e, n) in tiles:
            if w < east and e > west and s < north and n > south:
                yield from pyogrio.read_dataframe(path, columns=[], bbox=native).to_crs(EQ).geometry

    return boundary.to_crs(EQ).geometry.union_all(), canopy_for


def melbourne_data():
    boundary = gpd.read_file(CACHE / "melbourne_bounds.geojson", columns=[]).to_crs(EQ)
    canopy = gpd.read_file(CACHE / "melbourne_canopy.geojson", columns=[]).to_crs(EQ)
    assert len(canopy) == 57980 and canopy.geom_type.isin(("Polygon", "MultiPolygon")).all()
    index = canopy.sindex

    def canopy_for(bounds):
        return canopy.geometry.iloc[index.query(box(*bounds))]

    return boundary.geometry.union_all(), canopy_for


def vector_counts(cells, boundary, canopy_for):
    domain, tr = bounds_grid(boundary.bounds, cells)
    r0, c0, r1, c1 = domain
    surveyed, tree = (np.zeros((r1 - r0, c1 - c0), dtype=np.uint16) for _ in range(2))
    for dest, sample in windows(*tree.shape):
        shape = int(sample.height), int(sample.width)
        target = transform(sample, tr)
        bounds = array_bounds(*shape, target)
        if not boundary.intersects(box(*bounds)):
            continue
        valid = burn([boundary], target, shape)
        sampled_tree = burn(canopy_for(bounds), target, shape)
        surveyed[dest] = pixel_counts(valid)
        tree[dest] = pixel_counts(sampled_tree & valid)
    return domain, surveyed, tree


def paris_source(boundary):
    destination = CACHE / "paris_canopy_2m.tif"
    if destination.exists():
        return destination
    meta = requests.get(PARIS_SERVICE, params={"f": "json"}, timeout=40)
    meta.raise_for_status()
    extent = meta.json()["fullExtent"]
    extent_box = box(*(extent[k] for k in ("xmin", "ymin", "xmax", "ymax")))
    boundary = boundary.intersection(extent_box)
    step = 6000
    w, s, e, n = boundary.bounds
    jobs = [box(x, y, x + step, y + step) for x in range(int(w // step) * step, int(e) + 1, step)
            for y in range(int(s // step) * step, int(n) + 1, step)]
    jobs = [tuple(map(int, tile.bounds)) for tile in jobs if boundary.intersects(tile)]

    def download(bounds):
        tile = CACHE / f"paris_{bounds[0]}_{bounds[1]}.tif"
        if tile.exists():
            return tile
        size = step // STEP
        args = dict(f="image", bbox=",".join(map(str, bounds)), bboxSR=2154, imageSR=2154,
                    size=f"{size},{size}", format="tiff", pixelType="F32", interpolation="RSP_NearestNeighbor",
                    renderingRule='{"rasterFunction":"None"}', adjustAspectRatio="false", compression="LZW")
        response = requests.get(f"{PARIS_SERVICE}/exportImage", params=args, timeout=120)
        response.raise_for_status()
        with rasterio.MemoryFile(response.content) as file, file.open() as src:
            assert tuple(src.bounds) == bounds and src.crs.to_epsg() == 2154 and src.res == (STEP, STEP)
            raw = src.read(1, masked=True).filled(np.nan)
            height = np.where(raw >= 1000, raw - 1000, np.where(raw >= 100, raw - 100, raw))
            defined = np.isfinite(raw) & (raw >= 0)
            inside = burn([extent_box], src.transform, raw.shape).astype(bool)
            classified = np.where(defined & inside, 1 + (height > 3), 0).astype("uint8")
            profile = src.profile
        profile.update(dtype="uint8", nodata=0, compress="deflate")
        pending = tile.with_suffix(".part.tif")
        with rasterio.open(pending, "w", **profile) as out:
            out.write(classified, 1)
        pending.replace(tile)
        print("Paris tile", bounds[:2], int((classified == 2).sum()), flush=True)
        return tile

    with ThreadPoolExecutor(max_workers=4) as pool:
        tiles = list(pool.map(download, jobs))
    pending = destination.with_suffix(".part.tif")
    merge(tiles, dst_path=pending, dst_kwds={"compress": "deflate", "tiled": True})
    pending.replace(destination)
    return destination


def saopaulo_source(boundary):
    index_file = (CACHE / "saopaulo_opentopo_index.zip").as_posix()
    tiles = pyogrio.read_dataframe(f"zip://{index_file}!BR17_SaoPaulo_TileIndex.shp", columns=["file_name"])
    tiles["code"] = tiles.file_name.str.extract(r"_(\d{4}-\d{3})\.laz")
    groups = sorted(tiles.loc[tiles.intersects(boundary), "code"].str[:4].dropna().unique())
    classified = []
    for group in groups:
        spec = ET.parse(CACHE / f"saopaulo_{group}.vrt")
        x, dx, _, y, _, dy = map(float, spec.findtext("GeoTransform").split(","))
        footprints = []
        for source in spec.findall(".//ComplexSource"):
            code = Path(source.findtext("SourceFilename")).stem.removeprefix("VHM-")
            ox, oy, width, height = (float(source.find("DstRect").attrib[k]) for k in ("xOff", "yOff", "xSize", "ySize"))
            tile = tiles.loc[tiles.code.eq(code), "geometry"].union_all()
            footprints.append(tile.intersection(box(x + ox * dx, y + (oy + height) * dy,
                                                    x + (ox + width) * dx, y + oy * dy)))
        coverage = unary_union(footprints).intersection(boundary)
        assert not coverage.is_empty, group
        out = CACHE / f"saopaulo_canopy_{group}.tif"
        if not out.exists():
            with ZipFile(CACHE / f"saopaulo_{group}.zip") as archive:
                path = CACHE / f"VHM-0-merged-{group}.tif"
                if not path.exists():
                    archive.extract(path.name, CACHE)
            w, s, e, n = coverage.bounds
            w, s, e, n = np.array([np.floor(w / STEP), np.floor(s / STEP),
                                    np.ceil(e / STEP), np.ceil(n / STEP)], dtype=int) * STEP
            tr = from_origin(w, n, STEP, STEP)
            width, height = (e - w) // STEP, (n - s) // STEP
            with rasterio.open(path) as src:
                assert src.crs.to_epsg() == 31983 and src.res == (.5, .5) and src.nodata == 0
                with WarpedVRT(src, crs=src.crs, transform=tr, width=width, height=height,
                               resampling=Resampling.nearest, nodata=0, tolerance=0.001) as vrt:
                    raw = vrt.read(1)
            valid = burn([coverage], tr, raw.shape).astype(bool) & np.isfinite(raw)
            classes = np.where(valid, 1 + (raw > 3), 0).astype("uint8")
            pending = out.with_suffix(".part.tif")
            with rasterio.open(pending, "w", driver="GTiff", height=height, width=width, count=1, crs="EPSG:31983",
                               transform=tr, nodata=0, dtype="uint8", compress="deflate") as sink:
                sink.write(classes, 1)
            pending.replace(out)
            print("São Paulo mosaic", group, int(valid.sum()), int((classes == 2).sum()), flush=True)
        classified.append(out)
    destination = CACHE / "saopaulo_canopy_2m.tif"
    pending = destination.with_suffix(".part.tif")
    merge(classified, method="max", dst_path=pending,
          dst_kwds={"compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512})
    pending.replace(destination)
    return destination


def raster_counts(cells, path, boundary):
    with rasterio.open(path) as src:
        extent = box(*transform_bounds(src.crs, EQ, *src.bounds, densify_pts=41))
        boundary = boundary.intersection(extent)
        domain, tr = bounds_grid(boundary.bounds, cells)
        r0, c0, r1, c1 = domain
        shape = r1 - r0, c1 - c0
        surveyed, tree = (np.zeros(shape, dtype=np.uint16) for _ in range(2))
        with WarpedVRT(src, crs=EQ, transform=tr, width=shape[1] * SUB, height=shape[0] * SUB,
                       resampling=Resampling.nearest, nodata=0, tolerance=0.001, warp_mem_limit=96) as vrt:
            for dest, window in windows(*shape):
                target = transform(window, tr)
                h, w = int(window.height), int(window.width)
                if not boundary.intersects(box(*array_bounds(h, w, target))):
                    continue
                sample = vrt.read(1, window=window)
                valid = burn([boundary], target, (h, w)).astype(bool)
                valid &= sample > 0
                surveyed[dest] = pixel_counts(valid)
                tree[dest] = pixel_counts(valid & (sample == 2))
    return domain, surveyed, tree


def prepare(label):
    destination = ROOT / "data" / f"{label}.npz"
    if destination.exists():
        return
    with np.load(DATA / "worldcover" / f"{CITY_IDS[label]}.npz") as z:
        cells = {k: z[k] for k in ("row", "col")}
    if label == "london2020_22":
        domain, land, tree = vector_counts(cells, *london_data())
    elif label == "melbourne2021":
        domain, land, tree = vector_counts(cells, *melbourne_data())
    elif label == "paris2021":
        metros = gpd.read_file(CACHE / "paris_bounds.geojson", columns=[])
        source = paris_source(metros.to_crs("EPSG:2154").geometry.union_all())
        domain, land, tree = raster_counts(cells, source, metros.to_crs(EQ).geometry.union_all())
    else:
        city = pd.read_csv(DATA / "centre_sample.csv").set_index("id").loc[CITY_IDS[label]]
        project = Transformer.from_crs("EPSG:4326", "EPSG:31983", always_xy=True)
        point = Point(*project.transform(city.longitude, city.latitude))
        municipality = gpd.read_file(CACHE / "saopaulo_bounds.geojson").to_crs("EPSG:31983")
        native = point.buffer(6000, quad_segs=64).intersection(municipality.geometry.union_all())
        target = gpd.GeoSeries([native], crs="EPSG:31983").to_crs(EQ).iloc[0]
        domain, land, tree = raster_counts(cells, saopaulo_source(native), target)
    r0, c0, r1, c1 = domain
    r, c = cells["row"], cells["col"]
    inside = (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
    for key, field in (("land", land), ("tree", tree)):
        aligned = np.zeros(len(r), dtype=np.uint16)
        aligned[inside] = field[r[inside] - r0, c[inside] - c0]
        cells[key] = aligned
    assert np.any(cells["land"] == COUNT) and np.all(cells["tree"] <= cells["land"])
    pending = destination.with_suffix(".part.npz")
    np.savez_compressed(pending, **cells)
    pending.replace(destination)
    print(label, "complete survey cells", int((cells["land"] == COUNT).sum()), flush=True)


if __name__ == "__main__":
    prepare(sys.argv[1])
