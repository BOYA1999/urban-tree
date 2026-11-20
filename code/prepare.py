import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
CRS = "ESRI:54009"
X0, Y0 = -18_041_000, 9_000_000
JRC = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
UCDB = f"{JRC}/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_GLOBE_R2024A/V1-2/GHS_UCDB_GLOBE_R2024A_V1_2.zip"
SMOD = f"{JRC}/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V2-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip"
POP = "https://jrc-ghsl.s3.eu-central-1.amazonaws.com/ghs-pop/r2023a/54009/100m/2020/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
M49 = "https://unstats.un.org/unsd/methodology/m49/overview/"
ESA = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map"
IO = "https://api.impactobservatory.com/stac-aws/collections/io-10m-annual-lulc/items"
REGIONS = ["Africa", "Asia", "Europe", "Latin America and the Caribbean", "Northern America", "Oceania"]


def unpack_raster_or_vector(url, extension):
    import shutil
    import zipfile

    CACHE.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE / url.rsplit("/", 1)[-1]
    if not archive_path.exists():
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with archive_path.with_suffix(".part").open("wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
        archive_path.with_suffix(".part").replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        name, = (n for n in archive.namelist() if n.endswith(extension))
        path = CACHE / Path(name).name
        if not path.exists() or path.stat().st_size != archive.getinfo(name).file_size:
            pending = path.with_suffix(".part")
            with archive.open(name) as src, pending.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            pending.replace(path)
    return path


def sample_centres():
    sample_path = DATA / "centres.gpkg"
    if sample_path.exists():
        return gpd.read_file(sample_path)
    path = unpack_raster_or_vector(UCDB, ".gpkg")
    names = gpd.read_file(path, layer="GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A",
                          columns=["ID_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_GAD_2025", "GC_CNT_UNN_2025"])
    pop = gpd.read_file(path, layer="GHSL_UCDB_THEME_GHSL_GLOBE_R2024A",
                        columns=["ID_UC_G0", "GH_POP_TOT_2020"], ignore_geometry=True)
    names.columns = ["id", "city", "country", "country_un", "geometry"]
    pop.columns = ["id", "pop_ucdb_2020"]
    frame = names.merge(pop, on="id", validate="one_to_one")
    frame = frame.loc[frame.pop_ucdb_2020 >= 300000].copy()
    response = requests.get(M49, timeout=25)
    response.raise_for_status()
    m49 = next(t for t in pd.read_html(io.StringIO(response.text)) if "Country or Area" in t)
    region = m49.set_index("Country or Area")["Region Name"].to_dict()
    region.update(m49.loc[m49["Region Name"].eq("Americas")].set_index("Country or Area")["Sub-region Name"])
    aliases = {
        "CÃ´te d'Ivoire": "Côte d’Ivoire",
        "CuraÃ§ao": "Curaçao",
        "RÃ©union": "Réunion",
        "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
        "Netherlands": "Netherlands (Kingdom of the)",
        "Turkey": "Türkiye",
        "Dem. People's Republic of Korea": "Democratic People's Republic of Korea",
    }
    region.update({"Kosovo (under UNSC res. 1244)": "Europe", "China, Taiwan Province of China": "Asia"})
    frame["region"] = frame.country_un.replace(aliases).map(region)
    if not frame.region.isin(REGIONS).all():
        raise ValueError(frame.loc[~frame.region.isin(REGIONS), ["country_un", "region"]].drop_duplicates())
    frame = frame.sort_values(["region", "pop_ucdb_2020", "id"], ascending=[True, False, True])
    frame = frame.loc[frame.groupby(["region", "country"]).cumcount() < 2]
    frame = frame.groupby("region", sort=False).head(8).copy()
    frame["rank_in_region"] = frame.groupby("region").cumcount() + 1
    centroids = frame.geometry.representative_point().to_crs("EPSG:4326")
    frame["longitude"], frame["latitude"] = centroids.x, centroids.y
    frame.to_file(sample_path, index=False)
    frame.drop(columns=["geometry"]).to_csv(DATA / "centre_sample.csv", index=False)
    return frame


def worldcover_tiles(bounds):
    west, south, east, north = bounds
    for lat, lon in product(range(int(np.floor(south / 3)) * 3, int(np.ceil(north / 3)) * 3, 3),
                            range(int(np.floor(west / 3)) * 3, int(np.ceil(east / 3)) * 3, 3)):
        tag = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        url = f"{ESA}/ESA_WorldCover_10m_2020_v100_{tag}_Map.tif"
        reply = requests.head(url, timeout=25)
        if reply.status_code == 404:
            continue
        reply.raise_for_status()
        yield url


def impact_tiles(bounds):
    response = requests.get(IO, timeout=25, params={
        "bbox": ",".join(str(x) for x in bounds), "limit": 100,
        "datetime": "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
    })
    response.raise_for_status()
    return sorted({f["assets"]["supercell"]["href"] for f in response.json()["features"]})


def class_counts(urls, transform, shape, tree_code, water_code, invalid_codes):
    mosaic = np.zeros(shape, dtype=np.uint8)
    for url in urls:
        with rasterio.open(url) as ds:
            with WarpedVRT(ds, crs=CRS, transform=transform, width=shape[1], height=shape[0],
                           resampling=Resampling.nearest, src_nodata=0, nodata=0,
                           tolerance=0.001, warp_mem_limit=72) as vrt:
                values = vrt.read(1)
        for code in invalid_codes:
            values[values == code] = 0
        np.copyto(mosaic, values, where=(mosaic == 0) & (values != 0))
    axes = (1, 3)
    h, w = shape
    grouped = mosaic.reshape(h // 10, 10, w // 10, 10)
    valid = (grouped != 0).sum(axis=axes, dtype=np.uint8)
    land = valid - (grouped == water_code).sum(axis=axes, dtype=np.uint8)
    tree = (grouped == tree_code).sum(axis=axes, dtype=np.uint8)
    return valid, land, tree


def prepare(row, cover, smod):
    finished = DATA / cover / f"{row.id}.npz"
    if finished.exists():
        return row.city, "cached"
    geom = row.geometry
    x1, y1, x2, y2 = geom.bounds
    left, top = np.floor(x1 / 100) * 100, np.ceil(y2 / 100) * 100
    width = int((np.ceil(x2 / 100) * 100 - left) / 100)
    height = int((top - np.floor(y1 / 100) * 100) / 100)
    transform = from_origin(left, top, 100, 100)
    shape = (height, width)
    bbox = array_bounds(*shape, transform)
    footprint = geometry_mask([geom], out_shape=shape, transform=transform, invert=True)
    config = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="35",
                  GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1", GDAL_CACHEMAX=192 * 1024 * 1024)
    with rasterio.Env(**config):
        with rasterio.open(POP) as ds:
            window = from_bounds(*bbox, ds.transform)
            assert ds.window_transform(window).almost_equals(transform) and ds.res == (100, 100)
            population = ds.read(1, window=window, masked=True).filled(-1)
        with rasterio.open(smod) as ds:
            with WarpedVRT(ds, crs=CRS, transform=transform, width=width, height=height) as vrt:
                core = vrt.read(1) == 30
        geo_bbox = transform_bounds(CRS, "EPSG:4326", *bbox, densify_pts=41)
        if cover == "worldcover":
            urls = list(worldcover_tiles(geo_bbox))
            codes = (10, 80, ())
        else:
            urls = impact_tiles(geo_bbox)
            codes = (2, 1, (10, 255))
        if not urls:
            raise ValueError(f"No {cover} image for {row.city}")
        valid, land, tree = class_counts(urls, from_origin(left, top, 10, 10), (height * 10, width * 10), *codes)
    y, x = np.where(footprint)
    if np.any(tree > land) or not np.any(core[footprint]):
        raise ValueError(f"Inconsistent rasters for {row.city}")
    if np.mean(valid[y, x] / 100) < 0.95:
        raise ValueError(f"Less than 95% valid mapped pixels for {row.city}")
    pending = finished.with_suffix(".part.npz")
    np.savez_compressed(pending, row=y + int((Y0 - top) / 100), col=x + int((left - X0) / 100),
                        core=core[y, x], population=population[y, x], valid=valid[y, x],
                        land=land[y, x], tree=tree[y, x], sources=json.dumps(urls))
    pending.replace(finished)
    return row.city, f"{len(x):,} cells, {len(urls)} tiles"


def main(cover):
    assert cover in {"worldcover", "impact"}
    (DATA / cover).mkdir(parents=True, exist_ok=True)
    sample = sample_centres()
    smod = unpack_raster_or_vector(SMOD, ".tif")
    print(f"{cover}: {len(sample)} centres across {sample.region.nunique()} regions", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        tasks = {pool.submit(prepare, row, cover, smod): row.city for row in sample.itertuples()}
        for task in as_completed(tasks):
            try:
                city, details = task.result()
                print(city, details, flush=True)
            except Exception as error:
                errors.append((tasks[task], repr(error)))
                print("FAILED", errors[-1], flush=True)
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main(sys.argv[1])
