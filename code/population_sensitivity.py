from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import reproject, transform_bounds
from shapely.geometry import box, mapping

from analysis import arrays, measure, overall
from prepare import CRS, DATA, X0, Y0

API = "https://api.worldpop.org/v3/raster"
DIR = DATA / "worldpop_cases"


def check(city):
    path = DIR / f"{city.id}.tif"
    with np.load(DATA / "worldcover" / f"{city.id}.npz") as source:
        raw = {key: source[key] for key in ("population", "row", "col", "valid", "land", "tree", "core")}
    y = raw["row"] - raw["row"].min()
    x = raw["col"] - raw["col"].min()
    transform = from_origin(X0 + 100 * int(raw["col"].min()), Y0 - 100 * int(raw["row"].min()), 100, 100)
    destination = np.full((int(y.max()) + 1, int(x.max()) + 1), -1.0)
    params = transform_bounds(CRS, "EPSG:4326", *array_bounds(*destination.shape, transform), densify_pts=41)
    if not path.exists():
        west, south, east, north = params
        bbox = box(west - .02, south - .02, east + .02, north + .02)
        response = requests.post(API, json={"geojson": mapping(bbox), "year": 2020, "resolution": "100m"},
                                 timeout=(20, 100))
        response.raise_for_status()
        if response.headers.get("Content-Type") != "image/tiff":
            raise ValueError("WorldPop returned an unexpected format")
        pending = path.with_suffix(".part")
        pending.write_bytes(response.content)
        pending.replace(path)
    with rasterio.open(path) as source:
        reproject(rasterio.band(source, 1), destination, dst_crs=CRS, dst_transform=transform,
                  dst_nodata=-1, resampling=Resampling.sum)
    wp = destination[y, x]
    reference, reference_audit = arrays(raw, "core2020")
    common = (wp >= 0) & (raw["population"] >= 0)
    results = []
    for label, population in (("GHS-POP", raw["population"]), ("WorldPop", wp)):
        same_support = raw | {"population": np.where(common, population, -1)}
        values, audit = arrays(same_support, "core2020")
        base = overall(*values)
        metrics = measure(values, 1000, base)
        results.append({
            "id": city.id, "city": city.city, "region": city.region, "population_product": label,
            "population_million": audit["population"] / 1e6,
            "common_land_pct": 100 * audit["land_km2"] / reference_audit["land_km2"],
            "gap_100m_pp": base[2], "gap_1km_pp": metrics["resident_minus_area_pp"],
            "shift_pp": metrics["coarsening_change_pp"],
            "gross_pp": metrics["gross_reassignment_pp"],
        })
    return results


if __name__ == "__main__":
    DIR.mkdir(exist_ok=True)
    sample = gpd.read_file(DATA / "centres.gpkg").query("rank_in_region == 1")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [item for city in pool.map(check, sample.itertuples()) for item in city]
    report = pd.DataFrame(results)
    report.to_csv(Path(__file__).resolve().parents[1] / "results" / "population_sensitivity.csv",
                  index=False, float_format="%.8f")
    print(report.round(3).to_string(index=False))
