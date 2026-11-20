from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "revision_external_20261120"
sys.path.insert(0, str(PRIOR / "code"))
from compare_external import COUNT, DATA, EARLIER, PRODUCTS, REFERENCE_CITIES, arrays, grow, kernel

EXTRAS = {1747: "melbourne2022_catchments", 7277: "saopaulo2017_municipal"}
EPS = 1e-8


def domains(city, existing_only):
    refs = REFERENCE_CITIES[city.id]
    previous = EARLIER if city.id in (8099, 2869) else PRIOR
    yield "original", {r: arrays(previous / "data" / f"{r}.npz") for r in refs}
    if city.id in EXTRAS:
        yield "complete_core_global", {}
        extra = EXTRAS[city.id]
        path = ROOT / "data" / f"{extra}.npz"
        if path.exists() and not existing_only:
            more = {extra: arrays(path)}
            yield "expanded", more
            if city.id == 1747:
                yield "municipal_two_references", more | {refs[0]: arrays(previous / "data" / f"{refs[0]}.npz")}


def evaluate(city, existing_only):
    source_global = {k: arrays(DATA / k / f"{city.id}.npz") for k in PRODUCTS}
    wc = source_global["worldcover"]
    row, col, population = (wc[k] for k in ("row", "col", "population"))
    for k in ("row", "col", "core", "population"):
        np.testing.assert_array_equal(wc[k], source_global["impact"][k])
    rr, cc = row - row.min(), col - col.min()
    work = np.zeros((int(rr.max()) + 1, int(cc.max()) + 1))
    neighbourhood = kernel(city.longitude, city.latitude, 300)
    n = int(neighbourhood.sum())
    core = wc["core"] & (population >= 0)
    full_population = population[core & (wc["land"] > 0)].sum(dtype=float)
    base = core & (wc["land"] == 100) & (source_global["impact"]["land"] == 100)
    work[rr, cc] = base
    core_eligible = grow(work, neighbourhood)[rr, cc] == n
    # Upper quartile of population per occupied 100 m core cell, defined by land area.
    density_cutoff = np.quantile(population[core_eligible & (population > 0)], .75)
    canopy = {}
    for scope, local in domains(city, existing_only):
        source = local | source_global
        denom = {k: 100 if k in PRODUCTS else int(v.get("tree_denominator", COUNT)) for k, v in source.items()}
        presence = base.copy()
        for name, data in source.items():
            for k in ("row", "col"):
                np.testing.assert_array_equal(wc[k], data[k])
            if name not in PRODUCTS:
                presence &= data["land"] == COUNT
            if name not in canopy:
                work[rr, cc] = data["tree"]
                canopy[name] = grow(work, neighbourhood)[rr, cc].astype(np.int64)
        work[rr, cc] = presence
        keep = grow(work, neighbourhood)[rr, cc] == n
        r, c, p = row[keep], col[keep], population[keep].astype(float)
        assert len(p) > 0 and p.sum() > 0
        totals = {k: canopy[k][keep] for k in source}
        denom = {k: n * v for k, v in denom.items()}
        fields = {k: v / denom[k] for k, v in totals.items()}
        weights = 100 * p / p.sum()
        fine_est = {k: np.dot(weights, v) for k, v in fields.items()}
        cutoffs = range(20, 41)
        fine_pass = {k: {t: 100 * value >= t * denom[k] for t in cutoffs} for k, value in totals.items()}
        fine_att = {k: {t: weights[m].sum() for t, m in v.items()} for k, v in fine_pass.items()}
        domain_id = dict(id=city.id, city=city.city, scope=scope, n_cells=len(p), land_km2=.01 * len(p),
                         population=p.sum(), core_population_share_pct=100 * p.sum() / full_population,
                         density_per_km2=p.sum() / (.01 * len(p)),
                         dense_core_resident_share_pct=weights[p >= density_cutoff].sum())
        references = tuple(local) if local else ("complete_core_global",)
        for off_r, off_c in product(range(10), repeat=2):
            y, x = (r - off_r) // 10, (c - off_c) // 10
            g = (y - y.min()) * (int(np.ptp(x)) + 1) + x - x.min()
            blocks = np.bincount(g)
            people = np.bincount(g, weights=weights)
            size = blocks[g]
            origin = dict(row_shift_m=off_r * 100, col_shift_m=off_c * 100)
            results_30 = {}
            for name, numerator in totals.items():
                amounts = np.bincount(g, weights=numerator).astype(np.int64)
                amounts_live = amounts[g]
                assigned = amounts_live / (size * denom[name])
                delta = np.dot(weights, assigned - fields[name])
                gross_mean = np.dot(weights, abs(assigned - fields[name]))
                thresholds = cutoffs if off_r == off_c == 0 else (25, 30, 35)
                for t in thresholds:
                    coarse_pass = 100 * amounts_live >= t * size * denom[name]
                    fine = fine_pass[name][t]
                    upward = weights[~fine & coarse_pass].sum()
                    downward = weights[fine & ~coarse_pass].sum()
                    up = upward - downward
                    coarse_att = weights[coarse_pass].sum()
                    np.testing.assert_allclose(up, coarse_att - fine_att[name][t], atol=EPS)
                    if t == 30:
                        results_30[name] = (fine, coarse_pass, up, upward + downward, delta)
                    if t == 30 and off_r == off_c == 0:
                        minimum = np.full(len(blocks), np.inf)
                        maximum = np.full(len(blocks), -np.inf)
                        np.minimum.at(minimum, g, fields[name])
                        np.maximum.at(maximum, g, fields[name])
                        mixed = people[(minimum < .30) & (maximum >= .30)].sum()
                        assert upward + downward <= mixed + EPS
                        yield "domain", domain_id | dict(product=name, n_blocks=int((blocks > 0).sum()),
                            neighbourhood_mean_pct=100 * fields[name].mean(), fine_resident_mean_pct=fine_est[name],
                            shift_pp=delta, gross_mean_pp=gross_mean, mixed_at30_pct=mixed, fine_at30_pct=fine_att[name][t],
                            coarse_at30_pct=coarse_att, gross_at30_pct=upward+downward, net_at30_pp=up)
                    yield "threshold", dict(id=city.id, scope=scope, product=name, threshold_pct=t) | origin | dict(
                        mean_shift_pp=delta, fine_attainment_pct=fine_att[name][t], coarse_attainment_pct=coarse_att,
                        upward_pct=upward, downward_pct=downward, gross_pct=upward+downward, net_pp=up)
            for ref in references:
                if ref not in results_30:
                    continue
                f, q, net, total, shift = results_30[ref]
                for gmap in PRODUCTS:
                    ff, qq, comp_net, comp_total, comp_shift = results_30[gmap]
                    yield "agreement", dict(id=city.id, scope=scope, reference=ref, comparator=gmap) | origin | dict(
                        fine_disagreement_pct=weights[f != ff].sum(), coarse_disagreement_pct=weights[q != qq].sum(),
                        reference_net_pp=net, comparator_net_pp=comp_net, reference_gross_pct=total, comparator_gross_pct=comp_total,
                        both_mean_positive=shift > EPS and comp_shift > EPS,
                        opposite_net_sign=(net < -EPS and comp_net > EPS) or (net > EPS and comp_net < -EPS),
                        either_net_zero=abs(net) <= EPS or abs(comp_net) <= EPS)


def main(existing_only):
    cities = pd.read_csv(DATA / "centre_sample.csv").set_index("id", drop=False)
    groups = {k: [] for k in ("domain", "threshold", "agreement")}
    for city in REFERENCE_CITIES:
        for key, data in evaluate(cities.loc[city], existing_only):
            groups[key].append(data)
    d, t, a = (pd.DataFrame(groups[k]) for k in groups)
    prev = pd.read_csv(PRIOR / "results" / "local_map_comparison.csv").query("radius_m == 300")
    anchor = d.query("scope == 'original'")
    for rec in prev.itertuples():
        new = anchor[(anchor.id == rec.id) & (anchor["product"] == rec.product)].iloc[0]
        np.testing.assert_allclose([new.shift_pp, new.gross_at30_pct, new.net_at30_pp, new.land_km2],
                                   [rec.shift_pp, rec.gross_at30, rec.net_at30, rec.land_km2], atol=2e-8)
    names = ("reporting_domains", "threshold_sensitivity", "paired_attainment")
    suf = "_original" if existing_only else ""
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for name, frame in zip(names, (d, t, a)):
        frame.to_csv(results / f"{name}{suf}.csv", index=False, float_format="%.10g")
    print(d[["city", "scope", "product", "land_km2", "core_population_share_pct", "shift_pp",
             "gross_at30_pct", "net_at30_pp"]].round(4).to_string(index=False), flush=True)
    print(a.groupby(["reference", "scope", "comparator"])[["both_mean_positive", "opposite_net_sign"]].sum(), flush=True)


if __name__ == "__main__":
    main("--original-only" in sys.argv)
