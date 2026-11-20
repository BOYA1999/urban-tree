import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCALES = (100, 500, 1000, 2000)
THRESHOLDS = (10, 20, 30)
EPS = 1e-9


def arrays(raw, boundary):
    in_boundary = raw["core"] if boundary == "core2020" else np.ones(len(raw["core"]), bool)
    p, a = raw["population"], raw["land"]
    retained = in_boundary & (p >= 0) & (a > 0)
    audit = {
        "footprint_cells": int(in_boundary.sum()),
        "n_cells": int(retained.sum()),
        "population": p[retained].sum(dtype=float),
        "excluded_population": p[in_boundary & (p > 0) & (a == 0)].sum(dtype=float),
        "land_km2": a[retained].sum(dtype=float) / 10000,
        "land_km2_no_population_data": a[in_boundary & (p < 0)].sum(dtype=float) / 10000,
        "valid_pixel_percent": raw["valid"][in_boundary].mean(),
    }
    variables = [raw[name][retained] for name in ("row", "col", "land", "population", "tree")]
    return (*variables[:2], *(v.astype(float) for v in variables[2:])), audit


def block_totals(row, col, area, pop, tree, size, offset_row=0, offset_col=0):
    y = (row - offset_row) // size
    x = (col - offset_col) // size
    block = (y - y.min()) * (int(np.ptp(x)) + 1) + x - x.min()
    a, p, t = (np.bincount(block, weights=v) for v in (area, pop, tree))
    return block, a, p, t


def overall(row, col, area, pop, tree):
    n = pop.sum()
    total = area.sum()
    A = tree.sum() / total
    C = tree / area
    d = pop / area
    mean_d = n / total
    R = np.dot(pop, C) / n
    gap = 100 * (R - A)
    np.testing.assert_allclose(gap, 100 * np.dot(area / total, (d - mean_d) * (C - A)) / mean_d,
                               atol=EPS, rtol=0)
    return A, R, gap, C, d, mean_d


def measure(variables, scale, base):
    row, col, area, pop, tree = variables
    A, R, fine_gap, C, d, mean_d = base
    block, a, p, t = block_totals(*variables, scale // 100)
    keep = a > 0
    c = np.divide(t, a, out=np.zeros_like(t), where=keep)
    density = np.divide(p, a, out=np.zeros_like(p), where=keep)
    resident = np.dot(p, c) / pop.sum()
    between = 100 * np.dot(a / a.sum(), (density - mean_d) * (c - A)) / mean_d
    within = 100 * np.dot(area / a.sum(), (d - density[block]) * (C - c[block])) / mean_d
    np.testing.assert_allclose([a.sum(), p.sum(), t.sum()], [area.sum(), pop.sum(), tree.sum()],
                               rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose([fine_gap, resident * 100 - A * 100],
                               [within + between, between], atol=EPS, rtol=0)
    result = {
        "scale_m": scale, "area_tree_pct": A * 100, "resident_tree_pct": 100 * resident,
        "resident_minus_area_pp": between, "within_pp": within, "coarsening_change_pp": -within,
        "blocks": int(keep.sum()),
    }
    assigned = c[block]
    delta = assigned - C
    gain = 100 * np.dot(pop, np.maximum(delta, 0)) / pop.sum()
    loss = 100 * np.dot(pop, np.maximum(-delta, 0)) / pop.sum()
    np.testing.assert_allclose(gain - loss, -within, atol=EPS, rtol=0)
    result.update(gross_reassignment_pp=gain + loss, upward_shift_pp=gain, downward_shift_pp=loss,
                  cancelled_shift_pp=2 * min(gain, loss))
    for threshold in THRESHOLDS:
        fine_pass, coarse_pass = C >= threshold / 100, assigned >= threshold / 100
        result[f"resident_share_at_{threshold}_pct"] = 100 * pop[coarse_pass].sum() / pop.sum()
        result[f"resident_gross_switch_at_{threshold}_pct"] = 100 * pop[fine_pass != coarse_pass].sum() / pop.sum()
        result[f"resident_up_switch_at_{threshold}_pct"] = 100 * pop[~fine_pass & coarse_pass].sum() / pop.sum()
        result[f"resident_down_switch_at_{threshold}_pct"] = 100 * pop[fine_pass & ~coarse_pass].sum() / pop.sum()
    if scale == 1000:
        lo = np.full(len(a), np.inf)
        hi = np.full(len(a), -np.inf)
        np.minimum.at(lo, block, C)
        np.maximum.at(hi, block, C)
        result["fine_min_bound_pct"] = 100 * np.dot(p[keep], lo[keep]) / p.sum()
        result["fine_max_bound_pct"] = 100 * np.dot(p[keep], hi[keep]) / p.sum()
        assert result["fine_min_bound_pct"] - EPS <= R * 100 <= result["fine_max_bound_pct"] + EPS
    return result


def shifted(variables, A):
    for r, c in product(range(10), repeat=2):
        _, a, p, t = block_totals(*variables, 10, r, c)
        retain = a > 0
        R = np.dot(p[retain], t[retain] / a[retain]) / p.sum()
        yield {"row_shift_m": r * 100, "col_shift_m": c * 100, "gap_pp": (R - A) * 100,
               "resident_tree_pct": R * 100}


def common_land(sample):
    rows = []
    for city in sample.itertuples(index=False):
        with np.load(DATA / "worldcover" / f"{city.id}.npz") as w, np.load(DATA / "impact" / f"{city.id}.npz") as o:
            for key in ("row", "col", "population", "core"):
                np.testing.assert_array_equal(w[key], o[key])
            common = (w["land"] == 100) & (o["land"] == 100) & w["core"]
            for name, source in (("worldcover", w), ("impact", o)):
                _, reference = arrays(source, "core2020")
                cropped = {key: source[key] for key in ("row", "col", "population", "tree", "land", "valid")}
                values, retained = arrays(cropped | {"core": common}, "core2020")
                base = overall(*values)
                result = measure(values, 1000, base)
                rows.append({"id": city.id, "city": city.city, "product": name,
                             "land_retained_pct": 100 * retained["land_km2"] / reference["land_km2"],
                             "population_retained_pct": 100 * retained["population"] / reference["population"],
                             "area_tree_pct": result["area_tree_pct"], "gap_100m_pp": base[2],
                             "shift_pp": result["coarsening_change_pp"], "gross_pp": result["gross_reassignment_pp"]})
    pd.DataFrame(rows).to_csv(RESULTS / "common_land_sensitivity.csv", index=False, float_format="%.8f")


def run():
    RESULTS.mkdir(exist_ok=True)
    sample = pd.read_csv(DATA / "centre_sample.csv")
    metrics, offsets, sources = [], [], {}
    for dataset, boundary in product(("worldcover", "impact"), ("core2020", "footprint2025")):
        for city in sample.itertuples(index=False):
            path = DATA / dataset / f"{city.id}.npz"
            if not path.exists():
                continue
            with np.load(path) as archive:
                values, audit = arrays(archive, boundary)
                sources.setdefault(dataset, {})[str(city.id)] = json.loads(archive["sources"].item())
            key = {"id": city.id, "city": city.city, "country": city.country.replace("MÃ©xico", "Mexico"),
                   "region": city.region, "product": dataset, "boundary": boundary}
            base = overall(*values)
            for scale in SCALES:
                metrics.append(key | audit | measure(values, scale, base))
            if boundary == "core2020":
                offsets.extend(key | s for s in shifted(values, base[0]))
    pd.DataFrame(metrics).to_csv(RESULTS / "city_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(offsets).to_csv(RESULTS / "grid_origin_sensitivity.csv", index=False, float_format="%.12g")
    (RESULTS / "source_tiles.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    common_land(sample)
    print(f"{len(metrics)} city-product-boundary-scale results and {len(offsets)} grid-origin results")


if __name__ == "__main__":
    run()
