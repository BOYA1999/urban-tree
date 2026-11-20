from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
DATA = BASE / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(BASE / "code"))
from analysis import block_totals, measure, overall, shifted

EQ = "ESRI:54009"
TO_PROJ = Transformer.from_crs("EPSG:4326", EQ, always_xy=True)
FROM_PROJ = Transformer.from_crs(EQ, "EPSG:4326", always_xy=True)
GEOD = Geod(ellps="WGS84")
PRODUCTS = ("worldcover", "impact")


def kernel(lon, lat, radius):
    x, y = TO_PROJ.transform(lon, lat)
    dy, dx = np.mgrid[-10:11, -10:11] * 100
    dest_lon, dest_lat = FROM_PROJ.transform(x + dx, y - dy)
    _, _, distance = GEOD.inv(np.full(dx.shape, lon), np.full(dx.shape, lat), dest_lon, dest_lat)
    w = (distance <= radius).astype(np.float64)
    assert w[10, 10] == 1 and w[0, :].sum() + w[-1, :].sum() + w[:, 0].sum() + w[:, -1].sum() == 0
    return w


def grow(field, weights):
    return np.rint(fftconvolve(field, weights[::-1, ::-1], mode="same"))


def local_fields(row, col, shape, fields, weights):
    matrix = np.zeros(shape, dtype=np.float64)
    for key, field in fields.items():
        matrix[row, col] = field
        yield key, grow(matrix, weights)[row, col]


def threshold_bracket(values, threshold=.30):
    y, x, a, p, t = values
    block, area, population, _ = block_totals(*values, 10)
    cover = t / a
    low = np.full(len(area), np.inf)
    high = np.full(len(area), -np.inf)
    np.minimum.at(low, block, cover)
    np.maximum.at(high, block, cover)
    certain = population[low >= threshold].sum() / p.sum()
    ambiguous = population[(low < threshold) & (high >= threshold)].sum() / p.sum()
    actual = p[cover >= threshold].sum() / p.sum()
    assert certain - 1e-12 <= actual <= certain + ambiguous + 1e-12
    return 100 * certain, 100 * (certain + ambiguous), 100 * ambiguous


def analyse(city, files):
    rasters = {}
    for product, path in files.items():
        with np.load(path) as z:
            rasters[product] = {k: z[k] for k in ("row", "col", "land", "tree", "valid", "core", "population")}
    main = rasters[next(iter(rasters))]
    for other in rasters.values():
        for key in ("row", "col", "core", "population"):
            np.testing.assert_array_equal(main[key], other[key])
    row, col = main["row"], main["col"]
    r, c = row - row.min(), col - col.min()
    shape = (r.max() + 1, c.max() + 1)
    core = main["core"]
    population = main["population"]
    large_kernel = kernel(city.longitude, city.latitude, 500)
    footprints = {"core": core}
    for product, v in rasters.items():
        footprints[product] = v["valid"]
    window_checks = dict(local_fields(r, c, shape, footprints, large_kernel))
    eligible = (population >= 0) & (window_checks["core"] == large_kernel.sum())
    for product, v in rasters.items():
        eligible &= (v["land"] == 100) & (window_checks[product] >= .95 * 100 * large_kernel.sum())
    core_land = core & (population >= 0) & (main["land"] > 0)
    retained = population[eligible].sum(dtype=float) / population[core_land].sum(dtype=float)
    keys = {"id": city.id, "city": city.city, "region": city.region, "n_cells": int(eligible.sum()),
            "population_million": population[eligible].sum(dtype=float) / 1e6,
            "fraction_core_population_retained": retained}
    a = np.full(int(eligible.sum()), 100.0)
    p = population[eligible].astype(float)
    results = []
    for radius in (0, 300, 500):
        weights = kernel(city.longitude, city.latitude, radius) if radius else None
        for product, v in rasters.items():
            if radius:
                totals = dict(local_fields(r, c, shape, {"tree": v["tree"], "land": v["land"]}, weights))
                t = totals["tree"][eligible] / totals["land"][eligible] * a
            else:
                t = v["tree"][eligible].astype(float)
            values = row[eligible], col[eligible], a, p, t
            base = overall(*values)
            fine, coarse = (measure(values, s, base) for s in (100, 1000))
            lo, hi, mixed = threshold_bracket(values)
            assert coarse["resident_gross_switch_at_30_pct"] <= mixed + 1e-8
            np.testing.assert_allclose(fine["resident_tree_pct"] + coarse["coarsening_change_pp"],
                                       coarse["resident_tree_pct"], atol=1e-8)
            correction = {}
            if radius == 300:
                changes = np.array([z["resident_tree_pct"] - fine["resident_tree_pct"]
                                    for z in shifted(values, base[0])])
                np.testing.assert_allclose(changes[0], coarse["coarsening_change_pp"], atol=1e-8)
                correction = dict(positive_origins=int((changes > 1e-8).sum()),
                                  median_shift_all_origins_pp=np.median(changes),
                                  shift_span_all_origins_pp=np.ptp(changes))
            results.append(keys | correction | dict(product=product, radius_m=radius,
                local_area_mean_pct=fine["area_tree_pct"], fine_resident_pct=fine["resident_tree_pct"],
                coarse_resident_pct=coarse["resident_tree_pct"], fine_gap_pp=base[2],
                coarse_gap_pp=coarse["resident_minus_area_pp"], shift_pp=coarse["coarsening_change_pp"],
                gross_pp=coarse["gross_reassignment_pp"], fine_at30=fine["resident_share_at_30_pct"],
                coarse_at30=coarse["resident_share_at_30_pct"], gross_at30=coarse["resident_gross_switch_at_30_pct"],
                lo_at30=lo, hi_at30=hi, mixed_at30=mixed))
    return results


def summaries(frame):
    records = []
    for (prod, radius), v in frame.groupby(["product", "radius_m"], sort=False):
        net = abs(v.coarse_at30 - v.fine_at30)
        records.append({
            "product": prod, "radius_m": int(radius), "cities": int(len(v)),
            "positive_shift": int((v.shift_pp > 1e-8).sum()),
            "strict_reversal": int(((v.fine_gap_pp * v.coarse_gap_pp) < -1e-8).sum()),
            "median_shift_pp": v.shift_pp.median(),
            "median_gross_pp": v.gross_pp.median(),
            "median_gross_at30": v.gross_at30.median(),
            "median_abs_net_at30": net.median(),
            "median_mixed_at30": v.mixed_at30.median(),
            "median_fraction_population_retained": v.fraction_core_population_retained.median(),
            "min_fraction_population_retained": v.fraction_core_population_retained.min(),
            "max_fraction_population_retained": v.fraction_core_population_retained.max(),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    cities = pd.read_csv(DATA / "centre_sample.csv")
    data = pd.DataFrame(row for city in cities.itertuples(index=False) for row in analyse(
        city, {p: DATA / p / f"{city.id}.npz" for p in PRODUCTS}))
    assert len(data) == 240 and data.groupby(["product", "radius_m"]).size().eq(40).all()
    data.to_csv(RESULTS / "fixed_neighbourhood_cities.csv", index=False, float_format="%.10g")
    s = summaries(data)
    s.to_csv(RESULTS / "fixed_neighbourhood_summary.csv", index=False, float_format="%.10g")
    print(s.round(4).to_string(index=False))
