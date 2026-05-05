# fiber-ml-project
Projekt naukowo-wdrożeniowy: modelowanie temperatury (T) i wilgotności względnej (RH)
na podstawie rozproszonych danych światłowodowych z czujnika **Luna OBR-4600**.

## Cel projektu

Opracowanie pipeline'u ML obejmującego 6 zadań:
1. Regresja statyczna T i RH
2. Regresja dynamiczna T i RH (z uwzględnieniem czasu)
3. Klasyfikacja reżimu pracy czujnika
4. Anomaly detection
5. Analiza histerezy T/RH
6. Analiza struktury przestrzenno-czasowej sygnału

Wymóg Trusted AI: pełna reprodukowalność, kontrola data leakage, interpretowalność modeli.

## Quickstart

```bash
git clone https://github.com/lyskawson/fiber-ml-project.git
cd fiber-ml-project

# Instalacja środowiska (Python 3.11)
uv sync --extra dev

# Pobranie danych z Hugging Face Hub (~1.7 GB, 5-10 min)
# Wymaga tokena HF z dostępem do private datasetu — patrz "Dataset access" niżej
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'
uv run python scripts/download_from_hf.py

# Testy na danych sample
uv run pytest tests/ -v
```

## Dataset access

Raw measurements (~1.5 GB) i processed Zarr (~162 MB) są przechowywane na
[Hugging Face Hub](https://huggingface.co/datasets/lyskawson/fiber-ml-luna-obr-4600)
jako prywatny dataset.

### Setup dla nowego członka zespołu

1. Załóż konto: https://huggingface.co/join
2. Wyślij swój HF username liderowi (lyskawson) — dostaniesz `read` access
3. Wygeneruj token z rolą `Read`: https://huggingface.co/settings/tokens
4. Pobierz dataset:

```bash
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'
uv run python scripts/download_from_hf.py             # all
uv run python scripts/download_from_hf.py --what raw       # tylko raw
uv run python scripts/download_from_hf.py --what processed # tylko Zarr
```

### Re-upload (tylko team lead, wymaga tokena `Write`)

```bash
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'   # Write scope
uv run python scripts/upload_to_hf.py
```

## Struktura repo
.
├── src/fiber_ml/          # Główny pakiet Python
│   ├── ingest/            # Parser .txt, manifest, konwersja do Zarr
│   ├── preprocessing/     # (TBD) normalizacja, segmentacja
│   ├── features/          # (TBD) feature engineering
│   ├── models/            # (TBD) modele ML
│   ├── eval/              # (TBD) metryki, wykresy
│   └── utils/             # Ścieżki, helpery
├── scripts/
│   ├── 01_build_manifest.py    # raw .txt -> manifest.csv
│   ├── 02_ingest_to_zarr.py    # raw .txt -> Zarr dataset
│   ├── upload_to_hf.py         # local -> HF Hub (team lead)
│   └── download_from_hf.py     # HF Hub -> local (każdy)
├── tests/                 # Testy pytest (działają na data/sample/)
├── data/sample/           # 2 pliki pomiarowe — w gicie do CI/testów
├── data/raw/              # 700 plików (~1.5 GB) — gitignored, na HF Hub
├── data/manifest.csv      # Mapa plików -> warunki — w gicie
├── data_processed/        # Zarr dataset — gitignored, na HF Hub
├── docs/                  # Opis formatu, ADR, dokumentacja projektu
├── notebooks/             # Eksploracja EDA
└── reports/               # Metryki, wykresy (generowane)

## Workflow regeneracji datasetu

Zarr generowany jest deterministycznie z raw przez:

```bash
uv run python scripts/01_build_manifest.py --data-dir data/raw --output data/manifest.csv
uv run python scripts/02_ingest_to_zarr.py --manifest data/manifest.csv --output data_processed/dataset.zarr
```

Po regeneracji team lead robi re-upload (`scripts/upload_to_hf.py`).

## Workflow dla zespołu

### Nazewnictwo branchy
task/<num>-<short-desc>    # np. task/1-static-regression-T

### Conventional commits
feat: add static T regression model
fix: handle NaN in spectral shift channel
docs: update data_format.md with supervisor clarification
chore: bump zarr to 2.18
test: add regression test for sparse spectral shift

### Pull Request

- Branch z `main`, PR do `main`
- CI musi przejść (lint + typecheck + testy)
- Minimum 1 reviewer

## Zespół

Patrz [CONTRIBUTORS.md](CONTRIBUTORS.md).