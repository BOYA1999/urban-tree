from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = ROOT / "results"
sys.path.insert(0, str(BASE / "revision_review_20261120" / "code"))
from review_analysis import COUNT, DATA, PRODUCTS, REFERENCE_CITIES, arrays, domains, grow, kernel
from neighbourhood import EQ, FROM_PROJ, GEOD, TO_PROJ

ORIGINAL = BASE / "revision_20261120" / "results"
REVIEW = BASE / "revision_review_20261120" / "results"
PERCENTILES = (.70, .80, .90)
X0, Y0 = -18_041_000, 9_000_000


def grid_index(row, col):
    return ((row - row.min()) * (int(np.ptp(col)) + 1) + col - col.min()).astype(np.int64)


def aggregate(z, g, cell_size=100):
    n = np.bincount(g)
    t = np.bincount(g, weights=z * cell_size)
    return t[g] / (cell_size * n[g]), n


def threshold(z, coarse, w, t):
    fine, grouped = z >= t, coarse >= t
    h, j = w[fine].sum(), w[grouped].sum()
    gross = w[fine != grouped].sum()
    assert abs(j - h) <= gross + 1e-9 and gross <= min(h + j, 200 - h - j) + 1e-8
    return dict(cutoff_pct=100 * t, fine_pct=h, coarse_pct=j, net_pp=j - h, gross_pct=gross)


def resident_quantiles(z, w):
    sort = np.argsort(z, kind="stable")
    sort = sort[w[sort] > 0]
    step = np.cumsum(w[sort])
    return {q: z[sort[np.searchsorted(step, q * step[-1], side="left")]] for q in PERCENTILES}


def projection_audit(city, kernels):
    x, y = TO_PROJ.transform(city.longitude, city.latitude)
    left = X0 + np.floor((x - X0) / 1000) * 1000
    top = Y0 - np.floor((Y0 - y) / 1000) * 1000
    off = np.linspace(0, 1000, 17)

    def length(xx, yy):
        lon, lat = FROM_PROJ.transform(xx, yy)
        return float(GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2].sum())

    row = dict(id=city.id, city=city.city, latitude=city.latitude, longitude=city.longitude,
               north_edge_m=length(left + off, np.full(len(off), top)),
               south_edge_m=length(left + off, np.full(len(off), top-1000)),
               west_edge_m=length(np.full(len(off), left), top - off),
               east_edge_m=length(np.full(len(off), left+1000), top - off))
    for radius, k in kernels.items():
        row[f"cells_{radius}m"] = int(k.sum())
        row[f"equivalent_radius_{radius}m"] = np.sqrt(k.sum() * 10000 / np.pi)
    return row


def global_checks(city, original):
    sources = {name: arrays(DATA / name / f"{city.id}.npz") for name in PRODUCTS}
    wc, io = (sources[name] for name in PRODUCTS)
    row, col, p, core = (wc[k] for k in ("row", "col", "population", "core"))
    for key in ("row", "col", "population", "core"):
        np.testing.assert_array_equal(wc[key], io[key])
    rr, cc = row - row.min(), col - col.min()
    shape = (int(rr.max()) + 1, int(cc.max()) + 1)
    work = np.zeros(shape)
    fully_land = (wc["land"] == 100) & (io["land"] == 100) & (p >= 0)
    kernels = {r: kernel(city.longitude, city.latitude, r) for r in (250, 300, 350, 500)}

    def convolve(v, k):
        work[rr, cc] = v
        return grow(work, k)[rr, cc]

    yield "geometry", projection_audit(city, kernels)
    use_core, use_footprint = {}, {}
    for r in (300, 500):
        k = kernels[r]
        labels = fully_land.copy()
        for src in sources.values():
            labels &= convolve(src["valid"], k) >= 95 * k.sum()
        use_core[r] = labels & (convolve(core, k) == k.sum())
        use_footprint[r] = labels & (convolve(np.ones(len(row)), k) == k.sum())
    main = use_core[500]
    groups = grid_index(row[main] // 10, col[main] // 10)
    counts = np.bincount(groups)
    partial = counts[groups] < 100
    full = main.copy()
    full[main] = ~partial
    yield "partial", dict(id=city.id, city=city.city, n_blocks=int((counts > 0).sum()),
                         n_partial=int(((counts > 0) & (counts < 100)).sum()),
                         n_full=int((counts == 100).sum()),
                         partial_population_pct=100 * p[main][partial].sum(dtype=float) / p[main].sum(dtype=float))

    site = CRS.from_proj4(f"+proj=aeqd +lat_0={city.latitude} +lon_0={city.longitude} +datum=WGS84 +units=m")
    to_site = Transformer.from_crs(EQ, site, always_xy=True)
    xx, yy = to_site.transform(X0 + (col[main] + .5)*100, Y0 - (row[main] + .5)*100)
    local_bins = {
        (a, b): grid_index(np.floor((yy - a) / 1000), np.floor((xx - b) / 1000))
        for a, b in product((0, 500), repeat=2)
    }
    field_by_map = {}
    for name, src in sources.items():
        fields = field_by_map[name] = {}
        for r in (250, 300, 350):
            k = kernels[r]
            t, a = convolve(src["tree"], k), convolve(src["land"], k)
            fields[r] = np.divide(t, a, out=np.zeros_like(t), where=a > 0)
    common_positive = np.logical_and.reduce([field[300][main] > 0 for field in field_by_map.values()])
    for name, src in sources.items():
        fields = field_by_map[name]
        z = fields[300]
        own = core & (p >= 0) & (src["land"] > 0)
        key = dict(id=city.id, city=city.city, region=city.region, product=name)
        for label, mask in (("retained", main), ("excluded", own & ~main)):
            w = p[mask].astype(float)
            area = src["land"][mask].sum(dtype=float)
            local = mask & use_core[300]
            local_p = p[local].astype(float)
            yield "eligibility", key | dict(
                group=label, cells=int(mask.sum()), population=w.sum(), area_km2=area/10000,
                residents_per_km2=10000*w.sum()/area if area else np.nan,
                same_cell_area_tree_pct=100*src["tree"][mask].sum(dtype=float)/area if area else np.nan,
                m300_evaluable_population=local_p.sum(),
                m300_resident_tree_pct=100*np.average(z[local], weights=local_p) if local_p.sum() else np.nan)
        for label, mask, radius in (
            ("core_common500", main, 300), ("core_300_only", use_core[300], 300),
            ("2025_common500", use_footprint[500], 300),
            ("2025_300_only", use_footprint[300], 300), ("full_blocks", full, 300),
            ("radius250", main, 250), ("radius350", main, 350),
        ):
            v, w = fields[radius][mask], p[mask].astype(float)
            if not w.sum():
                continue
            w *= 100 / w.sum()
            g = grid_index(row[mask] // 10, col[mask] // 10)
            assigned, _ = aggregate(v, g)
            delta = assigned - v
            stats = threshold(v, assigned, w, .30)
            yield "sensitivity", key | dict(scenario=label, n_cells=len(v), population=p[mask].sum(dtype=float),
                mean_pp=np.sum(w*delta), mean_abs_pp=np.sum(w*abs(delta))) | stats
            if label != "core_common500":
                continue
            old = original.loc[(city.id, name)]
            np.testing.assert_allclose([len(v), np.sum(w*delta), stats["fine_pct"], stats["coarse_pct"],
                stats["gross_pct"]], [old.n_cells, old.shift_pp, old.fine_at30, old.coarse_at30, old.gross_at30],
                atol=1e-7, rtol=1e-8)
            for t in range(10, 41):
                yield "thresholds", key | dict(rule="fixed", target=t, tie_share_pct=w[v == t/100].sum()) | threshold(v, assigned, w, t/100)
            for q, cutoff in resident_quantiles(v, w).items():
                yield "thresholds", key | dict(rule="resident_quantile", target=100*q,
                                               tie_share_pct=w[v == cutoff].sum()) | threshold(v, assigned, w, cutoff)
            pv, pj, pw = v[common_positive], assigned[common_positive], w[common_positive].copy()
            share = pw.sum()
            assert share > 0
            pw *= 100 / share
            for q, cutoff in resident_quantiles(pv, pw).items():
                yield "thresholds", key | dict(rule="positive_common_quantile", target=100*q,
                    conditional_population_pct=share, tie_share_pct=pw[pv == cutoff].sum()) | threshold(pv, pj, pw, cutoff)
            for (a, b), g2 in local_bins.items():
                local_z, _ = aggregate(v, g2)
                local_delta = local_z - v
                yield "local_grid", key | dict(origin_y_m=a, origin_x_m=b, n_cells=len(v),
                    mean_pp=np.sum(w*local_delta), mean_abs_pp=np.sum(w*abs(local_delta))) | threshold(v, local_z, w, .30)


def local_checks(city, original):
    globals_ = {m: arrays(DATA / m / f"{city.id}.npz") for m in PRODUCTS}
    wc = globals_["worldcover"]
    row, col, p, core = (wc[k] for k in ("row", "col", "population", "core"))
    rr, cc = row - row.min(), col - col.min()
    work = np.zeros((int(rr.max()) + 1, int(cc.max()) + 1))
    k = kernel(city.longitude, city.latitude, 300)
    n = k.sum()
    base = core & (p >= 0)

    def moving(v):
        work[rr, cc] = v
        return grow(work, k)[rr, cc]

    for scope, local in domains(city, existing_only=False):
        if scope not in ("original", "expanded"):
            continue
        sources = local | globals_
        d = {m: int(s.get("tree_denominator", COUNT)) if m in local else 100 for m, s in sources.items()}
        land = {m: s["land"] == (COUNT if m in local else 100) for m, s in sources.items()}
        keep = moving(base & np.logical_and.reduce(list(land.values()))) == n
        w = p[keep].astype(float)
        w *= 100 / w.sum()
        denom = {m: n*d[m] for m in sources}
        trees = {m: moving(s["tree"]).astype(np.int64) for m,s in sources.items()}
        zs = {m: trees[m][keep] / denom[m] for m in sources}
        g = grid_index(row[keep] // 10, col[keep] // 10)
        sizes = np.bincount(g)[g]
        coarse = {m: np.bincount(g, weights=trees[m][keep])[g] / (sizes*denom[m]) for m in sources}
        key = dict(id=city.id, city=city.city, scope=scope)
        for m in sources:
            check = original.loc[(city.id, scope, m)]
            assert len(w) == check.n_cells
            np.testing.assert_allclose(np.sum(w*(coarse[m]-zs[m])), check.shift_pp, atol=1e-8)
            if city.id in (5816, 2878):
                ref = next(iter(local.values()))
                for mask, elig in (("common", keep),
                    ("own_in_local_footprint", moving(base & (ref["land"] > 0) & land[m]) == n)):
                    v = trees[m][elig] / denom[m]
                    yield "map_levels", key | dict(product=m, mask=mask, cells=len(v),
                        population=p[elig].sum(dtype=float), area_mean_pct=100*v.mean(),
                        resident_mean_pct=100*np.average(v, weights=p[elig]))
        for ref in local:
            for name in PRODUCTS:
                distance = abs(zs[ref]-zs[name])
                yield "agreement", key | dict(local_map=ref, product=name,
                    area_mae_pp=100*distance.mean(), resident_mae_pp=np.sum(w*distance),
                    fine_disagreement_pct=w[(zs[ref] >= .30) != (zs[name] >= .30)].sum(),
                    coarse_disagreement_pct=w[(coarse[ref] >= .30) != (coarse[name] >= .30)].sum())
        if city.id not in (5816, 2878):
            continue
        selected = np.logical_and.reduce([z > 0 for z in zs.values()])
        local_cutoffs = {}
        for rule, take in (("resident_quantile", np.ones(len(w), bool)),
                           ("positive_common_quantile", selected)):
            share = w[take].sum()
            assert share > 0
            w_eval = w[take] * (100/share)
            local_cutoffs[rule] = take, share, w_eval, {m: resident_quantiles(z[take], w_eval) for m,z in zs.items()}
        for a,b in product(range(10), repeat=2):
            g2 = grid_index((row[keep]-a)//10, (col[keep]-b)//10)
            n2 = np.bincount(g2)[g2]
            for m,z in zs.items():
                assigned = np.bincount(g2, weights=trees[m][keep])[g2] / (n2*denom[m])
                for rule, (take, share, w_eval, cuts) in local_cutoffs.items():
                    for q,t in cuts[m].items():
                        yield "local_quantiles", key | dict(product=m, rule=rule, quantile=100*q,
                            row_shift_m=100*a, col_shift_m=100*b, conditional_population_pct=share,
                            tie_share_pct=w_eval[z[take] == t].sum()) | threshold(z[take], assigned[take], w_eval, t)


def bootstrap(data):
    results = []
    data = data.query("scenario == 'core_common500'").sort_values(["product", "id"])
    left = data.query("product == 'worldcover'").reset_index(drop=True)
    right = data.query("product == 'impact'").reset_index(drop=True)
    np.testing.assert_array_equal(left.id, right.id)
    rng = np.random.default_rng(20261120)
    choices = np.column_stack([rng.choice(ix, (10000, len(ix))) for ix in left.groupby("region").indices.values()])
    for name, frame in (("worldcover", left), ("impact", right)):
        for metric in ("mean_pp", "mean_abs_pp", "fine_pct", "coarse_pct", "gross_pct"):
            v = frame[metric].to_numpy()
            for quantile in (.25, .5, .75):
                q = np.quantile(v[choices], quantile, axis=1)
                results.append(dict(product=name, metric=metric, sample_quantile=quantile,
                                    estimate=np.quantile(v,quantile), lo=np.quantile(q,.025), hi=np.quantile(q,.975)))
    return pd.DataFrame(results)


if __name__ == "__main__":
    global_only = sys.argv[1:] == ["--global-only"]
    test = threshold(np.array([.29, .30]), np.array([.295, .295]), np.array([40., 60.]), .30)
    assert (test["fine_pct"], test["coarse_pct"], test["gross_pct"]) == (60., 0., 60.)
    cities = pd.read_csv(DATA / "centre_sample.csv")
    before = pd.read_csv(ORIGINAL / "fixed_neighbourhood_cities.csv").query("radius_m == 300").set_index(["id", "product"])
    if not global_only:
        local_before = pd.read_csv(REVIEW / "reporting_domains.csv").set_index(["id", "scope", "product"])
    outputs = {}
    for city in cities.itertuples(index=False):
        checks = list(global_checks(city, before))
        if city.id in REFERENCE_CITIES and not global_only:
            checks += list(local_checks(city, local_before))
        for name, row in checks:
            outputs.setdefault(name, []).append(row)
        print(f"{city.id} {city.city}: baseline replayed", flush=True)
    tables = {name: pd.DataFrame(rows) for name, rows in outputs.items()}
    tables["bootstrap"] = bootstrap(tables["sensitivity"])
    OUT.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, float_format="%.10g")
    print({k: len(v) for k,v in tables.items()}, flush=True)
