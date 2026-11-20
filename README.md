# Urban tree cover reporting across 40 urban centres

Python code for *How reporting units change urban tree cover indicators across 40 urban centres*. The main comparison keeps the residential population, mapped tree cover, study domain and approximately 300 m neighbourhood fixed while changing how values are assigned at 100 m and nominal 1 km² reporting supports.

## Environment

Use Python 3.12 and install the pinned dependencies from the repository root.

```sh
python -m pip install -r requirements.txt
```

Run the remaining commands from the repository root. The repository's MIT licence covers the code. Each source dataset retains its provider's terms.

## Global analysis

The two preparation commands reconstruct the published 40-centre selection and align the public 2020 inputs on the 100 m GHS Mollweide grid. They require network access and several gigabytes of temporary storage. For a replay from existing aligned inputs, place `centre_sample.csv`, `worldcover/<id>.npz` and `impact/<id>.npz` under `data/` and start with `code/analysis.py`.

```sh
python -X utf8 code/prepare.py worldcover
python -X utf8 code/prepare.py impact
python -X utf8 code/analysis.py
python -X utf8 code/summarize.py
python -X utf8 revision_20261120/code/neighbourhood.py
python -X utf8 revision_response_20261120/code/followup.py --global-only
```

- `results/` contains the same-cell comparisons at all report widths, grid-origin sensitivity and summary tables.
- `revision_20261120/results/` contains the fixed-neighbourhood results for the same-cell, 300 m and 500 m definitions. The 300 m check evaluates all 100 nominal 1 km grid origins.
- `revision_response_20261120/results/` contains fixed and relative threshold diagnostics, geometry and eligibility checks, local metric grid results and paired bootstrap summaries.

The exploratory population check uses the six region-leading footprints in `data/centres.gpkg` and streams or reuses the six WorldPop clips. The independent source check reads the original tree-cover tiles.

```sh
python -X utf8 code/population_sensitivity.py
python -X utf8 code/verify.py
```

Global sources are the European Commission Joint Research Centre's [GHS Urban Centre Database R2024A v1.2](https://doi.org/10.2905/1A338BE6-7EAF-480C-9664-3A8ADE88CBCD), [GHS-SMOD R2023A](https://doi.org/10.2905/A0DF7A6F-49DE-46EA-9BDE-563437A6E2BA) and [GHS-POP R2023A](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE), together with [UN M49](https://unstats.un.org/unsd/methodology/m49/overview/), [ESA WorldCover 2020 v100](https://doi.org/10.5281/zenodo.5571936), [Impact Observatory, Microsoft and Esri 2020 land cover](https://registry.opendata.aws/io-lulc/) and [WorldPop Global2 R2025A v1](https://doi.org/10.5258/SOTON/WP00839).

## Local canopy comparisons

Start with the global arrays and fixed-neighbourhood baseline above. Save the NYC 2017 COG as `revision_20261120/data/cache/nyc-land-cover-2017.tif` and extract the Toronto geodatabase to `revision_20261120/data/cache/LandCover2018.gdb`. The NYC 2021 command reads the Zenodo raster by windows. [NYC 2017 catalogue and COG](https://source.coop/betanyc/nyc-land-cover-2017) · [Toronto geodatabase download](https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/61642048-56bb-4050-b7c3-f569fcf94527/resource/69419e11-2dfa-4bcc-bed0-43a9dd2d0973/download/landcover2018_gdb.zip) · [NYC 2021 data](https://doi.org/10.5281/zenodo.14053441)

```sh
python -X utf8 revision_20261120/code/local_reference.py nyc2017
python -X utf8 revision_20261120/code/local_reference.py nyc2021
python -X utf8 revision_20261120/code/local_reference.py toronto2018
python -X utf8 revision_20261120/code/local_reference.py

python -X utf8 revision_external_20261120/code/fetch_reference_sources.py
python -X utf8 revision_external_20261120/code/prepare_external.py london2020_22
python -X utf8 revision_external_20261120/code/prepare_external.py paris2021
python -X utf8 revision_external_20261120/code/prepare_external.py melbourne2021
python -X utf8 revision_external_20261120/code/prepare_external.py saopaulo2017
python -X utf8 revision_external_20261120/code/compare_external.py

python -X utf8 revision_review_20261120/code/expand_melbourne.py
python -X utf8 revision_review_20261120/code/expand_saopaulo.py
python -X utf8 revision_review_20261120/code/review_analysis.py
python -X utf8 revision_response_20261120/code/followup.py
```

The local sequence produces matched-domain comparisons for six centres, the Melbourne and São Paulo footprint extensions, London and Paris threshold diagnostics, and tables of agreement between maps. The expanded source files and the municipal rasters require substantial disk space. The provider inputs, survey vintages, coverage and licences are documented in the [local source manifest](revision_external_20261120/source_manifest.md).

The aligned arrays use row and column offsets from `(-18,041,000, 9,000,000)` in ESRI:54009. Global arrays store land, tree and valid 10 m pixel counts, estimated residential population and the 2020 core flag. Local arrays use 2 m sampling of polygons or rasters and the same 100 m grid. Signed changes in canopy or attainment are in percentage points. Gross switching is a share of residents.
