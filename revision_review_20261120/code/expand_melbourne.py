from hashlib import file_digest
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "revision_external_20261120"
sys.path.insert(0, str(PRIOR / "code"))
import fetch_reference_sources as acquire
from prepare_external import COUNT, EQ, SUB, bounds_grid, pixel_counts, windows

SOURCES = {"melbourne_tree_series.tif": (48892726, "d7673dc799e0a58f1cedbfc161bbae77"),
           "melbourne_sites.tif": (48892720, "56ff880bd97d9dbacc2ec8cfc37dc1ba")}


def main():
    output = ROOT / "data" / "melbourne2022_catchments.npz"
    if output.exists():
        return
    acquire.CACHE = ROOT / "data" / "cache"
    acquire.CACHE.mkdir(parents=True, exist_ok=True)
    for name, (file_id, md5) in SOURCES.items():
        acquire.fetch((name, f"https://ndownloader.figshare.com/files/{file_id}"))
        with (acquire.CACHE / name).open("rb") as stream:
            assert file_digest(stream, "md5").hexdigest() == md5, name
    with np.load(ROOT.parent / "data" / "worldcover" / "1747.npz") as file:
        row, col, core = file["row"], file["col"], file["core"]
    with rasterio.open(acquire.CACHE / "melbourne_tree_series.tif") as src, rasterio.open(
            acquire.CACHE / "melbourne_sites.tif") as sites:
        assert src.count == 4 and np.allclose(src.res, (5, 5)) and np.isnan(src.nodata)
        assert src.crs == sites.crs and src.shape == sites.shape
        np.testing.assert_allclose(tuple(src.transform), tuple(sites.transform), atol=.02, rtol=0)
        band = src.descriptions.index("trees_2022") + 1
        bbox = transform_bounds(src.crs, EQ, *src.bounds, densify_pts=41)
        (r0, c0, r1, c1), tr = bounds_grid(bbox, {"row": row[core], "col": col[core]})
        shape = r1 - r0, c1 - c0
        land, tree = (np.zeros(shape, dtype=np.uint16) for _ in range(2))
        in_bbox = (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
        use = in_bbox & core
        needed = np.zeros(shape, dtype=bool)
        needed[row[use] - r0, col[use] - c0] = True
        params = dict(crs=EQ, transform=tr, height=shape[0]*SUB, width=shape[1]*SUB,
                      resampling=Resampling.nearest, tolerance=.001, warp_mem_limit=96)
        with WarpedVRT(src, nodata=-1, **params) as vrt, WarpedVRT(sites, nodata=0, **params) as footprint:
            for dest, win in windows(*shape):
                if not needed[dest].any():
                    continue
                z = vrt.read(band, window=win)
                valid = footprint.read(1, window=win) > 0
                tree_value = valid & np.isfinite(z) & (z >= 0)
                assert ((z[tree_value] <= 25) & (z[tree_value] == np.rint(z[tree_value]))).all()
                land[dest] = pixel_counts(valid)
                tree[dest] = pixel_counts(np.where(tree_value, z, 0).astype("uint8"))
    aligned = {}
    for key, value in (("land", land), ("tree", tree)):
        aligned[key] = np.zeros(len(row), dtype=np.uint16)
        aligned[key][use] = value[row[use] - r0, col[use] - c0]
    assert np.all(aligned["tree"] <= 25 * aligned["land"].astype(np.uint32))
    assert (aligned["land"] == COUNT).any()
    pending = output.with_suffix(".part.npz")
    np.savez_compressed(pending, row=row, col=col, tree_denominator=25 * COUNT, **aligned)
    pending.replace(output)
    print("Surveyed 100 m cells", np.count_nonzero(aligned["land"] == COUNT), flush=True)


if __name__ == "__main__":
    main()
