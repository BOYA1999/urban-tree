from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
MEL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/andasampa/height-model/"
TILES = ("3312", "3313", "3314", "3315", "3316", "3323", "3325", "3331", "3332", "3341")

SOURCES = {
    "london_canopy.zip": "https://data.london.gov.uk/download/2w4m3/e2e52636-7fc7-4e1f-8ccc-27954cfa2e24/Tree_Canopy_2024.zip",
    "london_bounds.zip": "https://data.london.gov.uk/download/20od9/9ba8c833-6370-4b11-abdc-314aa020d5e0/statistical-gis-boundaries-london.zip",
    "melbourne_canopy.geojson": f"{MEL}/tree-canopies-2021-urban-forest/exports/geojson",
    "melbourne_bounds.geojson": f"{MEL}/municipal-boundary/exports/geojson",
    "paris_bounds.geojson": "https://geo.api.gouv.fr/epcis/200054781?geometry=contour&format=geojson",
    "saopaulo_bounds.geojson": "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/3550308?formato=application%2Fvnd.geo%2Bjson&qualidade=maxima",
    "saopaulo_opentopo_index.zip": "https://opentopography.s3.sdsc.edu/pc-bulk/BR17_SaoPaulo/BR17_SaoPaulo_TileIndex.zip",
}
for group in TILES:
    stem = f"{group}-VHM/VHM-0-merged-{group}"
    for suffix, remote in (("vrt", "vrt"), ("zip", "tif")):
        SOURCES[f"saopaulo_{group}.{suffix}"] = f"{KAGGLE}{quote(stem + '.' + remote, safe='')}?datasetVersionNumber=4"


def fetch(task):
    name, url = task
    destination = CACHE / name
    if destination.exists():
        return
    pending = destination.with_suffix(f".{destination.suffix.lstrip('.')}.part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(25, 80)) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f"Unexpected status {response.status_code} for {name}")
                with pending.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            if name.endswith(".zip"):
                with ZipFile(pending) as archive:
                    assert archive.infolist()
            elif name.endswith(".vrt"):
                raw = pending.read_bytes()
                if raw.startswith(b"PK"):
                    with ZipFile(BytesIO(raw)) as archive:
                        raw = archive.read(archive.namelist()[0])
                assert raw.lstrip().startswith(b"<VRTDataset")
                pending.write_bytes(raw)
            pending.replace(destination)
            print(name, destination.stat().st_size, flush=True)
            return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, SOURCES.items()))
