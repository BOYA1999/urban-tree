import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def verify_city(city, dataset, table):
    with np.load(DATA / dataset / f"{city}.npz") as z:
        use = z["core"] & (z["population"] >= 0) & (z["land"] > 0)
        pixels = pd.DataFrame({k: z[k][use].astype(float) for k in ("row", "col", "land", "population", "tree")})
    n = pixels.population.sum()
    area = pixels.tree.sum() / pixels.land.sum()
    fine = pixels.tree / pixels.land
    coarse = pixels.groupby([pixels.row // 10, pixels.col // 10])[["land", "tree"]].transform("sum")
    assigned = coarse.tree / coarse.land
    p = pixels.population
    fine_mean = np.average(fine, weights=p)
    coarse_mean = np.average(assigned, weights=p)
    expected = np.array([area, fine_mean, coarse_mean, np.average(abs(assigned - fine), weights=p),
                         p[(fine >= .30) != (assigned >= .30)].sum() / n]) * 100
    recorded = table.loc[(dataset, city)]
    actual = [recorded.loc[100, "area_tree_pct"], recorded.loc[100, "resident_tree_pct"],
              recorded.loc[1000, "resident_tree_pct"], recorded.loc[1000, "gross_reassignment_pp"],
              recorded.loc[1000, "resident_gross_switch_at_30_pct"]]
    np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-7)
    return {
        "id": city, "product": dataset,
        "max_difference_pp": float(np.max(np.abs(expected - actual))),
    }


def verify_source(spot, sources):
    name, city, lon, lat = spot
    x, y = transform("EPSG:4326", "ESRI:54009", [lon], [lat])
    col = int((x[0] + 18041000) // 100)
    row = int((9000000 - y[0]) // 100)
    xs = -18041000 + col * 100 + 5 + np.arange(10) * 10
    ys = 9000000 - row * 100 - 5 - np.arange(10) * 10
    xs, ys = np.meshgrid(xs, ys)
    records = []
    for dataset, tree_code in (("worldcover", 10), ("impact", 2)):
        with np.load(DATA / dataset / f"{city}.npz") as f:
            index = (f["row"] == row) & (f["col"] == col)
            if not index.any():
                continue
            prepared = int(f["tree"][index][0])
        for url in sources[dataset][str(city)]:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="20"):
                with rasterio.open(url) as source:
                    a, b = transform("ESRI:54009", source.crs, xs.ravel().tolist(), ys.ravel().tolist())
                    r, c = rowcol(source.transform, a, b)
                    if min(r) < 0 or min(c) < 0 or max(r) >= source.height or max(c) >= source.width:
                        continue
                    data = source.read(1, window=Window(c.min(), r.min(), np.ptp(c) + 1, np.ptp(r) + 1))
                    sampled = data[r - r.min(), c - c.min()]
                    if np.any(sampled == 0):
                        continue
                    direct = int(np.count_nonzero(sampled == tree_code))
                    assert prepared == direct, (name, dataset, prepared, direct)
                    records.append({"place": name, "id": city, "product": dataset, "tree_pixels": direct})
                    break
    return records


if __name__ == "__main__":
    metrics = pd.read_csv(RESULTS / "city_metrics.csv")
    primary = metrics.loc[metrics.boundary.eq("core2020")].set_index(["product", "id", "scale_m"]).sort_index()
    ids = primary.index.droplevel("scale_m").unique()
    checked = [verify_city(city, p, primary) for p, city in ids]
    points = [("Central Park", 8099, -73.9770, 40.7763), ("Toronto Queen Street", 2869, -79.398, 43.650),
              ("Shanghai Century Park", 11345, 121.5485, 31.213), ("London Hyde Park", 5816, -0.161, 51.509),
              ("Parque Ibirapuera", 7277, -46.656, -23.585), ("Tiergarten Berlin", 5483, 13.3574, 52.512)]
    sources = json.loads((RESULTS / "source_tiles.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        sampled = [r for rs in pool.map(lambda place: verify_source(place, sources), points) for r in rs]
    assert len(sampled) >= 8
    pd.DataFrame(checked).to_csv(RESULTS / "verification_city.csv", index=False)
    pd.DataFrame(sampled).to_csv(RESULTS / "verification_source.csv", index=False)
    print(f"{len(checked)} independent aggregation comparisons. {len(sampled)} direct source checks.")
