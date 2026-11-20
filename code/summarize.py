import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ORDER = ["Africa", "Asia", "Europe", "Latin America and the Caribbean", "Northern America", "Oceania"]
GAP = "resident_minus_area_pp"
RESIDENT = "resident_tree_pct"
EPS = 1e-7


def q(values):
    return dict(zip(("min", "q1", "median", "q3", "max"), np.quantile(values, (0, .25, .5, .75, 1)).tolist()))


def pairwise(x, y):
    r, c = np.triu_indices(len(x), 1)
    a = x.to_numpy()[r] - x.to_numpy()[c]
    b = y.to_numpy()[r] - y.to_numpy()[c]
    different = (abs(a) > EPS) & (abs(b) > EPS)
    reverse = (a * b < 0) & different
    meaningful = (abs(a) >= 1) & (abs(b) >= 1)
    return {"spearman": spearmanr(x, y).statistic, "reversed": int(reverse.sum()),
            "comparable": int(different.sum()), "reversed_margin_1pp": int((reverse & meaningful).sum()),
            "comparable_margin_1pp": int(meaningful.sum())}


def describe(group):
    fine, coarse = (group.loc[group.scale_m.eq(i)].set_index("id").sort_index() for i in (100, 1000))
    different_sign = fine[GAP] * coarse[GAP] < -EPS
    changes = coarse[RESIDENT] - fine[RESIDENT]
    return {
        "cities": len(fine), "countries": fine.country.nunique(),
        "population": fine.population.sum(), "land_km2": fine.land_km2.sum(),
        "retained_cells": int(fine.n_cells.sum()), "excluded_population": fine.excluded_population.sum(),
        "min_valid_pixel_percent": fine.valid_pixel_percent.min(),
        "max_city_pop_excluded_pct": (100 * fine.excluded_population / (fine.population + fine.excluded_population)).max(),
        "land_km2_no_population_data": fine.land_km2_no_population_data.sum(),
        "negative_fine": int((fine[GAP] < -EPS).sum()), "negative_coarse": int((coarse[GAP] < -EPS).sum()),
        "sign_reversals": list(fine.loc[different_sign, "city"]),
        "sign_reversals_1pp": list(fine.loc[different_sign & (abs(fine[GAP]) >= 1) & (abs(coarse[GAP]) >= 1), "city"]),
        "toward_zero": int((coarse[GAP].abs() < fine[GAP].abs() - EPS).sum()),
        "above_zero_change": int((changes > EPS).sum()),
        "negative_change": list(fine.loc[changes < -EPS, "city"]),
        "area": q(fine.area_tree_pct),
        "gap_fine": q(fine[GAP]), "gap_coarse": q(coarse[GAP]), "r_fine": q(fine[RESIDENT]),
        "r_coarse": q(coarse[RESIDENT]), "shift": q(changes),
        "abs_shift": q(changes.abs()), "gross": q(coarse.gross_reassignment_pp),
        "cancelled": q(coarse.cancelled_shift_pp),
        "cancellation_fraction": q(coarse.cancelled_shift_pp / coarse.gross_reassignment_pp),
        "bounds_width": q(coarse.fine_max_bound_pct - coarse.fine_min_bound_pct),
        "bounds_straddle_area": int(((coarse.fine_min_bound_pct < coarse.area_tree_pct) &
                                    (coarse.fine_max_bound_pct > coarse.area_tree_pct)).sum()),
        "rank_area_fine": pairwise(fine.area_tree_pct, fine[RESIDENT]),
        "rank_fine_coarse": pairwise(fine[RESIDENT], coarse[RESIDENT]),
        "rank_gap_fine_coarse": pairwise(fine[GAP], coarse[GAP]),
        "scales": {int(s): {"negative": int((g[GAP] < -EPS).sum()),
                            "gap": q(g[GAP]), "gross": q(g.gross_reassignment_pp)}
                   for s, g in group.groupby("scale_m")},
    }


def main():
    metrics = pd.read_csv(RESULTS / "city_metrics.csv")
    shifts = pd.read_csv(RESULTS / "grid_origin_sensitivity.csv")
    sample = pd.read_csv(ROOT / "data" / "centre_sample.csv")
    assert len(sample) == 40 and metrics.groupby(["product", "boundary", "scale_m"]).size().eq(len(sample)).all()
    assert metrics.groupby(["product", "boundary", "scale_m"]).ngroups == 16
    output = {f"{p}_{b}": describe(z) for (p, b), z in metrics.groupby(["product", "boundary"])}
    primary = metrics.loc[(metrics["product"] == "worldcover") & (metrics["boundary"] == "core2020")]
    fine, coarse = (primary.loc[primary.scale_m.eq(s)].set_index("id").sort_index() for s in (100, 1000))
    region = []
    thresholds = []
    for name in ORDER:
        f, c = (source.loc[source["region"] == name] for source in (fine, coarse))
        region.append({"region": name, "cities": len(f), "countries": f.country.nunique(),
                       "population_million": f.population.sum() / 1e6, "area_tree_pct": f.area_tree_pct.median(),
                       "gap_100m_pp": f[GAP].median(), "gap_1000m_pp": c[GAP].median(),
                       "change_pp": (c[RESIDENT] - f[RESIDENT]).median(),
                       "gross_pp": c.gross_reassignment_pp.median()})
    for tau in (10, 20, 30):
        A, Rf, Rc = fine.area_tree_pct >= tau, fine[RESIDENT] >= tau, coarse[RESIDENT] >= tau
        zf = fine[f"resident_share_at_{tau}_pct"]
        zc = coarse[f"resident_share_at_{tau}_pct"]
        up = coarse[f"resident_up_switch_at_{tau}_pct"]
        down = coarse[f"resident_down_switch_at_{tau}_pct"]
        gross = coarse[f"resident_gross_switch_at_{tau}_pct"]
        np.testing.assert_allclose((zc - zf).to_numpy(), (up - down).to_numpy(), atol=EPS)
        thresholds.append({"threshold_pct": tau, "area_pass": int(A.sum()), "resident_fine_pass": int(Rf.sum()),
                           "resident_coarse_pass": int(Rc.sum()), "area_fine_disagree": int((A != Rf).sum()),
                           "scale_upgrade": int((~Rf & Rc).sum()), "scale_downgrade": int((Rf & ~Rc).sum()),
                           "median_residents_above_fine_pct": zf.median(),
                           "median_residents_above_coarse_pct": zc.median(), "median_gross_switch_pct": gross.median(),
                           "median_absolute_net_switch_pct": (zc-zf).abs().median(),
                           "gross_switch_q1": gross.quantile(.25), "gross_switch_q3": gross.quantile(.75)})
    shifts = shifts.loc[shifts["product"] == "worldcover"]
    g = shifts.groupby("id").gap_pp.agg(["min", "max", "size"])
    assert g["size"].eq(100).all() and len(g) == len(sample)
    g["span_pp"] = g["max"] - g["min"]
    g["positive_origins"] = shifts.groupby("id").gap_pp.apply(lambda x: (x > EPS).sum())
    g["city"] = fine.city
    output["origin"] = {
        "span": q(g.span_pp), "either_sign": list(g.loc[(g["min"] < -EPS) & (g["max"] > EPS), "city"]),
        "always_positive": list(g.loc[g.positive_origins.eq(100), "city"]),
        "any_positive": list(g.loc[g.positive_origins.gt(0), "city"]),
    }
    for a, b, label in [("worldcover_core2020", "impact_core2020", "cross_product"),
                        ("worldcover_core2020", "worldcover_footprint2025", "cross_boundary")]:
        p1, b1 = a.split("_")
        p2, b2 = b.split("_")
        output[label] = {}
        for scale in (100, 1000):
            left = metrics.loc[(metrics["product"] == p1) & (metrics["boundary"] == b1) &
                               (metrics["scale_m"] == scale)].set_index("id").sort_index()
            right = metrics.loc[(metrics["product"] == p2) & (metrics["boundary"] == b2) &
                                (metrics["scale_m"] == scale)].set_index("id").sort_index()
            output[label][str(scale)] = {
                "gap_spearman": spearmanr(left[GAP], right[GAP]).statistic,
                "same_direction_count": int((np.sign(left[GAP]) == np.sign(right[GAP])).sum()),
                "abs_gap_difference": q((left[GAP] - right[GAP]).abs()),
                "mean_abs_area_difference": np.mean(abs(left.area_tree_pct - right.area_tree_pct)),
                "same_direction_change_count": int((np.sign(left.coarsening_change_pp) ==
                                                     np.sign(right.coarsening_change_pp)).sum()),
            }
    pd.DataFrame(region).to_csv(RESULTS / "regions.csv", index=False, float_format="%.6f")
    pd.DataFrame(thresholds).to_csv(RESULTS / "thresholds.csv", index=False, float_format="%.6f")
    g.to_csv(RESULTS / "city_grid_origin_summary.csv", float_format="%.6f")
    (RESULTS / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
