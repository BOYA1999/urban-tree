import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from neighbourhood import DATA, EQ, PRODUCTS, RESULTS, ROOT, grow, kernel, measure, overall

Y0, X0 = 9_000_000, -18_041_000
OUT = ROOT / "data"
CACHE = OUT / "cache"
STEP = 2
SUB = 100 // STEP
COUNT = SUB ** 2
NYC21 = "https://zenodo.org/api/records/14053441/files/landcover_nyc_2021_6in.tif/content"
SOURCES = {
    "nyc2017": {"id": 8099, "source": CACHE / "nyc-land-cover-2017.tif"},
    "nyc2021": {"id": 8099, "source": NYC21},
    "toronto2018": {"id": 2869, "source": CACHE / "LandCover2018.gdb"},
}


def bounds_grid(bbox, cells):
    w, s, e, n = bbox
    r0 = max(cells["row"].min(), int(np.floor((Y0 - n) / 100)))
    r1 = min(cells["row"].max() + 1, int(np.ceil((Y0 - s) / 100)))
    c0 = max(cells["col"].min(), int(np.floor((w - X0) / 100)))
    c1 = min(cells["col"].max() + 1, int(np.ceil((e - X0) / 100)))
    return (r0, c0, r1, c1), from_origin(X0 + c0 * 100, Y0 - r0 * 100, STEP, STEP)


def counts(classes):
    H, W = classes.shape
    grouped = classes.reshape(H // SUB, SUB, W // SUB, SUB)
    valid = (grouped != 0).sum(axis=(1, 3), dtype=np.uint16)
    land = valid - (grouped == 4).sum(axis=(1, 3), dtype=np.uint16)
    tree = (grouped == 1).sum(axis=(1, 3), dtype=np.uint16)
    return valid, land, tree


def raster_counts(path, cells):
    config = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="40", GDAL_HTTP_MAX_RETRY="2",
                  GDAL_CACHEMAX=240 * 1024 * 1024, CPL_VSIL_CURL_CACHE_SIZE=32 * 1024 * 1024)
    with rasterio.Env(**config):
        with rasterio.open(path, overview_level=2) as source:
            print("Reading reference", str(path).rsplit("/", 1)[-1], source.crs, source.res, flush=True)
            bbox = transform_bounds(source.crs, EQ, *source.bounds, densify_pts=41)
            bounds, tr = bounds_grid(bbox, cells)
            r0, c0, r1, c1 = bounds
            height, width = (r1 - r0), (c1 - c0)
            outputs = [np.zeros((height, width), dtype=np.uint16) for _ in range(3)]
            with WarpedVRT(source, crs=EQ, transform=tr, width=width * SUB, height=height * SUB,
                           resampling=Resampling.nearest, src_nodata=0, nodata=0, tolerance=0.001,
                           warp_mem_limit=96) as vrt:
                for start in range(0, height, 60):
                    span = min(60, height - start)
                    cut = vrt.read(1, window=Window(0, start * SUB, width * SUB, span * SUB))
                    if np.any(cut > 8):
                        raise ValueError("Unexpected NYC categorical reference value")
                    for result, crop in zip(outputs, counts(cut)):
                        result[start:start + span] = crop
            return bounds, outputs


def vector_counts(path, cells):
    info = pyogrio.read_info(path, layer="LandCover2018")
    frame = gpd.read_file(path, layer="LandCover2018", columns=["gridcode"], where="gridcode > 0").to_crs(EQ)
    assert frame.gridcode.isin(range(1, 9)).all()
    bbox = transform_bounds(info["crs"], EQ, *info["total_bounds"], densify_pts=41)
    bounds, tr = bounds_grid(bbox, cells)
    r0, c0, r1, c1 = bounds
    print("Rasterizing Toronto coverage", len(frame), flush=True)
    pixel = rasterize(zip(frame.geometry, frame.gridcode), transform=tr,
                      out_shape=((r1 - r0) * SUB, (c1 - c0) * SUB), dtype="uint8")
    return bounds, counts(pixel)


def prepare(label):
    destination = OUT / f"{label}.npz"
    if destination.exists():
        return
    source = SOURCES[label]
    with np.load(DATA / "worldcover" / f'{source["id"]}.npz') as z:
        raw = {k: z[k] for k in ("row", "col")}
    method = vector_counts if label == "toronto2018" else raster_counts
    (r0, c0, r1, c1), outputs = method(source["source"], raw)
    rr, cc = raw["row"], raw["col"]
    valid_window = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    aligned = {}
    for k, values in zip(("valid", "land", "tree"), outputs):
        full = np.zeros(len(rr), dtype=np.uint16)
        full[valid_window] = values[rr[valid_window] - r0, cc[valid_window] - c0]
        aligned[k] = full
    assert np.any(aligned["land"] == COUNT) and np.all(aligned["tree"] <= aligned["land"])
    pending = destination.with_suffix(".part.npz")
    np.savez_compressed(pending, **raw, **aligned)
    pending.replace(destination)
    print(label, "cells wholly terrestrial", (aligned["land"] == COUNT).sum(), flush=True)


def compare(city, refs):
    sources = {}
    for product in PRODUCTS:
        with np.load(DATA / product / f"{city.id}.npz") as z:
            sources[product] = {k: z[k] for k in ("row", "col", "core", "population", "land", "tree")}
    for label in refs:
        with np.load(OUT / f"{label}.npz") as z:
            sources[label] = {k: z[k] for k in ("row", "col", "land", "tree")}
    wc = sources["worldcover"]
    for source in sources.values():
        for k in ("row", "col"):
            np.testing.assert_array_equal(wc[k], source[k])
    row, col = wc["row"], wc["col"]
    rr, cc = row - row.min(), col - col.min()
    shape = (int(rr.max()) + 1, int(cc.max()) + 1)
    neighbourhood = kernel(city.longitude, city.latitude, 300)
    common = wc["core"] & (wc["population"] >= 0)
    for name, source in sources.items():
        common &= source["land"] == (100 if name in PRODUCTS else COUNT)
    matrix = np.zeros(shape, dtype=float)
    matrix[rr, cc] = common
    eligible = grow(matrix, neighbourhood)[rr, cc] == neighbourhood.sum()
    p = wc["population"][eligible].astype(float)
    a = np.ones(len(p))
    full_pop = wc["population"][wc["core"] & (wc["population"] >= 0) & (wc["land"] > 0)].sum(dtype=float)
    keys = dict(id=city.id, city=city.city, n_cells=len(p), land_km2=len(p) / 100,
                population_million=p.sum() / 1e6, fraction_core_population=p.sum() / full_pop)
    summary = []
    for radius in (0, 300):
        fields = {}
        for name, source in sources.items():
            denominator = 100 if name in PRODUCTS else COUNT
            if radius == 0:
                field = source["tree"][eligible] / denominator
            else:
                matrix[rr, cc] = source["tree"]
                field = grow(matrix, neighbourhood)[rr[eligible], cc[eligible]] / (denominator * neighbourhood.sum())
            fields[name] = field
        for reference in refs:
            for name in (reference, *PRODUCTS):
                cover = fields[name]
                values = row[eligible], col[eligible], a, p, cover
                base = overall(*values)
                result = measure(values, 1000, base)
                signed = cover - fields[reference]
                summary.append(keys | dict(reference=reference, product=name, radius_m=radius,
                     land_mean_pct=100 * base[0], resident_mean_fine_pct=100 * base[1],
                     signed_area_error_pp=100 * signed.mean(), area_mae_pp=100 * abs(signed).mean(),
                     resident_signed_error_pp=100 * np.average(signed, weights=p),
                     resident_mae_pp=100 * np.average(abs(signed), weights=p),
                     fine_gap_pp=base[2], coarse_gap_pp=result["resident_minus_area_pp"],
                     shift_pp=result["coarsening_change_pp"], gross_pp=result["gross_reassignment_pp"],
                     gross_at30=result["resident_gross_switch_at_30_pct"]))
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT.mkdir(exist_ok=True)
        prepare(sys.argv[1])
    else:
        RESULTS.mkdir(exist_ok=True)
        cities = pd.read_csv(DATA / "centre_sample.csv").set_index("id", drop=False)
        output = []
        for city in sorted({s["id"] for s in SOURCES.values()}):
            refs = [name for name, s in SOURCES.items() if s["id"] == city and (OUT / f"{name}.npz").exists()]
            output.extend(compare(cities.loc[city], refs))
        table = pd.DataFrame(output)
        table.to_csv(RESULTS / "local_reference_comparison.csv", index=False, float_format="%.10g")
        print(table[["city", "reference", "product", "radius_m", "land_km2", "land_mean_pct",
                     "resident_mean_fine_pct", "shift_pp", "area_mae_pp", "gross_at30"]].round(3).to_string(index=False))
