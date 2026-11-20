from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EARLIER = ROOT.parent / "revision_20261120"
sys.path.insert(0, str(EARLIER / "code"))
from local_reference import COUNT
from neighbourhood import DATA, PRODUCTS, grow, kernel, measure, overall, shifted

REFERENCE_CITIES = {
    8099: ("nyc2017", "nyc2021"), 2869: ("toronto2018",),
    5816: ("london2020_22",), 2878: ("paris2021",), 1747: ("melbourne2021",), 7277: ("saopaulo2017",),
}


def arrays(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def comparison(city, references):
    sources = {k: arrays(DATA / k / f"{city.id}.npz") for k in PRODUCTS}
    for ref in references:
        parent = EARLIER if ref.startswith(("nyc", "toronto")) else ROOT
        sources[ref] = arrays(parent / "data" / f"{ref}.npz")
    wc = sources["worldcover"]
    row, col = wc["row"], wc["col"]
    rr, cc = row - row.min(), col - col.min()
    matrix = np.zeros((rr.max() + 1, cc.max() + 1))
    pop = wc["population"]
    common = wc["core"] & (pop >= 0)
    denom = {k: 100 if k in PRODUCTS else COUNT for k in sources}
    for name, fields in sources.items():
        for key in ("row", "col"):
            np.testing.assert_array_equal(fields[key], wc[key])
        common &= fields["land"] == denom[name]
    for key in ("core", "population"):
        np.testing.assert_array_equal(sources["impact"][key], wc[key])
    neighbourhood = kernel(city.longitude, city.latitude, 300)
    matrix[rr, cc] = common
    matched = grow(matrix, neighbourhood)[rr, cc] == neighbourhood.sum()
    p = pop[matched].astype(float)
    full_pop = pop[wc["core"] & (pop >= 0) & (wc["land"] > 0)].sum(dtype=float)
    keys = dict(id=city.id, city=city.city, n_cells=len(p), land_km2=len(p) / 100,
                population_million=p.sum() / 1e6, fraction_core_population=p.sum() / full_pop,
                neighbourhood_cells=int(neighbourhood.sum()), inhabited_cells=int((p > 0).sum()))
    for radius in (0, 300):
        mapped = {}
        for name, source in sources.items():
            matrix[rr, cc] = source["tree"]
            mapped[name] = (grow(matrix, neighbourhood)[rr[matched], cc[matched]] /
                            (denom[name] * neighbourhood.sum()) if radius else source["tree"][matched] / denom[name])
        for ref in references:
            for product in (ref, *PRODUCTS):
                cover = mapped[product]
                values = row[matched], col[matched], np.ones(len(p)), p, cover
                base = overall(*values)
                fine, coarse = (measure(values, size, base) for size in (100, 1000))
                signed = cover - mapped[ref]
                result = keys | dict(reference=ref, product=product, radius_m=radius,
                    land_mean_pct=100 * base[0], resident_mean_fine_pct=100 * base[1],
                    resident_mean_coarse_pct=coarse["resident_tree_pct"], blocks_1km=coarse["blocks"],
                    fine_gap_pp=base[2], coarse_gap_pp=coarse["resident_minus_area_pp"],
                    shift_pp=coarse["coarsening_change_pp"], gross_pp=coarse["gross_reassignment_pp"],
                    signed_area_error_pp=100 * signed.mean(), area_mae_pp=100 * abs(signed).mean(),
                    resident_signed_error_pp=100 * np.average(signed, weights=p),
                    resident_mae_pp=100 * np.average(abs(signed), weights=p))
                if radius == 300:
                    change = np.array([v["resident_tree_pct"] - fine["resident_tree_pct"] for v in shifted(values, base[0])])
                    np.testing.assert_allclose(change[0], result["shift_pp"], atol=1e-8)
                    result.update(origins_positive=int((change > 1e-8).sum()), origins_negative=int((change < -1e-8).sum()),
                                  origin_min_pp=change.min(), origin_max_pp=change.max())
                for t in (10, 20, 30):
                    share, gross, up, down = (f"resident_{k}_at_{t}_pct" for k in
                                              ("share", "gross_switch", "up_switch", "down_switch"))
                    net = coarse[share] - fine[share]
                    np.testing.assert_allclose([net, coarse[gross]], [coarse[up] - coarse[down],
                                               coarse[up] + coarse[down]], atol=1e-8)
                    result.update({f"fine_at{t}": fine[share], f"coarse_at{t}": coarse[share],
                                   f"net_at{t}": net, f"gross_at{t}": coarse[gross],
                                   f"up_at{t}": coarse[up], f"down_at{t}": coarse[down]})
                yield result


def main(selected):
    cities = pd.read_csv(DATA / "centre_sample.csv").set_index("id", drop=False)
    ids = [i for i, refs in REFERENCE_CITIES.items() if not selected or any(ref in selected for ref in refs)]
    table = pd.DataFrame(result for i in ids for result in comparison(cities.loc[i], REFERENCE_CITIES[i]))
    keys = ["reference", "product", "radius_m"]
    old = pd.read_csv(EARLIER / "results" / "local_reference_comparison.csv").set_index(keys)
    matched = table.set_index(keys)
    common_rows = matched.index.intersection(old.index)
    for key in old.select_dtypes("number").columns:
        np.testing.assert_allclose(matched.loc[common_rows, key], old.loc[common_rows, key],
                                   rtol=1e-8, atol=1e-8, err_msg=key)
    if not selected:
        out = ROOT / "results"
        out.mkdir(exist_ok=True)
        table.to_csv(out / "local_map_comparison.csv", index=False, float_format="%.10g")
    focus = table[table.radius_m.eq(300)]
    print(focus[["reference", "product", "land_km2", "population_million", "fraction_core_population",
                 "land_mean_pct", "shift_pp", "gross_at30", "net_at30", "origins_positive"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
