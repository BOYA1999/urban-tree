from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import numpy as np
import pyogrio
from pyproj import Transformer
from shapely import MultiPoint

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "revision_external_20261120"
sys.path.insert(0, str(PRIOR / "code"))
import fetch_reference_sources as acquire
import prepare_external as prepare

BASE = ROOT.parent / "data"
CACHE = ROOT / "data" / "cache" / "saopaulo_municipality"
INDEX = "saopaulo_opentopo_index.zip"
BOUNDARY = "saopaulo_bounds.geojson"


def fetch(task):
    name, _ = task
    local = PRIOR / "data" / "cache" / name
    target = CACHE / name
    if not target.exists() and local.exists():
        os.link(local, target)
    try:
        acquire.fetch(task)
    except BadZipFile:
        pending = target.with_suffix(".zip.part")
        with pending.open("rb") as f:
            prefix = f.read(4)
        if prefix not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            raise
        group = target.stem.split("_")[-1]
        packed = target.with_suffix(".packed.zip")
        with ZipFile(packed, "w") as z:
            z.write(pending, f"VHM-0-merged-{group}.tif")
        packed.replace(target)
        print("Packed native GeoTIFF", group, flush=True)


def domain(cells):
    index = pyogrio.read_dataframe(f"zip://{(CACHE / INDEX).as_posix()}!BR17_SaoPaulo_TileIndex.shp",
                                   columns=["file_name"])
    official = gpd.read_file(CACHE / BOUNDARY).to_crs(31983).geometry.union_all()
    core = cells["core"]
    project = Transformer.from_crs(prepare.EQ, 31983, always_xy=True)
    x, y = project.transform(-18_041_000 + 100 * cells["col"][core] + 50,
                             9_000_000 - 100 * cells["row"][core] - 50)
    mask = official.intersection(MultiPoint(np.column_stack((x, y))).convex_hull.buffer(300))
    groups = sorted(index.loc[index.intersects(mask), "file_name"].str.extract(r"_(\d{4})-")[0].dropna().unique())
    return mask, groups


def main():
    output = ROOT / "data" / "saopaulo2017_municipal.npz"
    if output.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    acquire.CACHE = prepare.CACHE = CACHE
    for name in (INDEX, BOUNDARY):
        fetch((name, acquire.SOURCES[name]))
    with np.load(BASE / "worldcover" / "7277.npz") as source:
        grid = {name: source[name] for name in ("row", "col", "core")}
    native, groups = domain(grid)
    print("São Paulo VHM groups", len(groups), " ".join(groups), flush=True)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "saopaulo_municipal_groups.txt").write_text("\n".join(groups) + "\n", encoding="utf-8")
    jobs = []
    for group in groups:
        stem = f"{group}-VHM/VHM-0-merged-{group}"
        for filename, source_type in (("vrt", "vrt"), ("zip", "tif")):
            url = f"{acquire.KAGGLE}{quote(stem + '.' + source_type, safe='')}?datasetVersionNumber=4"
            jobs.append((f"saopaulo_{group}.{filename}", url))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, jobs))
    print("Completed source acquisition, bytes", sum((CACHE / name).stat().st_size for name, _ in jobs), flush=True)
    if sys.argv[1:] == ["--fetch-only"]:
        return
    target = gpd.GeoSeries([native], crs=31983).to_crs(prepare.EQ).iloc[0]
    bounds, land, tree = prepare.raster_counts(grid, prepare.saopaulo_source(native), target)
    r0, c0, r1, c1 = bounds
    row, col = (grid[name] for name in ("row", "col"))
    inside = (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
    counts = {}
    for name, values in (("land", land), ("tree", tree)):
        aligned = np.zeros(len(row), dtype=np.uint16)
        aligned[inside] = values[row[inside] - r0, col[inside] - c0]
        counts[name] = aligned
    assert np.any(counts["land"] == 2500) and np.all(counts["tree"] <= counts["land"])
    pending = output.with_suffix(".part.npz")
    np.savez_compressed(pending, row=row, col=col, **counts)
    pending.replace(output)
    print("Surveyed 100 m cells", np.count_nonzero(counts["land"] == 2500), flush=True)


if __name__ == "__main__":
    main()
