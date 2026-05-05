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

# Pobranie danych z Hugging Face Hub (~1.7 GB, ~2 min na dobrym łączu)
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'
uv run python scripts/download_from_hf.py

# Walidacja środowiska
uv run pytest tests/ -v
```

## Dataset access

Raw measurements (~1.5 GB) i processed Zarr (~162 MB) są przechowywane na
[Hugging Face Hub](https://huggingface.co/datasets/lyskawson/fiber-ml-luna-obr-4600)
jako prywatny dataset.

### Setup dla nowego członka zespołu

1. Załóż konto: https://huggingface.co/join
2. Wyślij swój HF username liderowi (lyskawson) — dostaniesz **Write** access (każdy w zespole jest równoprawnym kontrybutorem)
3. Wygeneruj token z rolą **Write**: https://huggingface.co/settings/tokens
4. Pobierz dataset:

```bash
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'
uv run python scripts/download_from_hf.py                  # wszystko
uv run python scripts/download_from_hf.py --what raw       # tylko raw
uv run python scripts/download_from_hf.py --what processed # tylko Zarr
```

### Re-upload zmienionych danych

Każdy w zespole z tokenem Write może upload'ować zmiany:

```bash
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'   # Write scope
uv run python scripts/upload_to_hf.py
```

> **Uwaga**: Tokenów HF nigdy nie commituj do gita ani nie wklejaj w czatach/issues/PR.
> Trzymaj w env var (`export HF_TOKEN=...`) lub w lokalnym `.env` (gitignored).

## Struktura repo
```text
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
│   ├── upload_to_hf.py         # local -> HF Hub
│   └── download_from_hf.py     # HF Hub -> local
├── tests/                 # Testy pytest (działają na data/sample/)
├── data/sample/           # 2 pliki pomiarowe — w gicie do CI/testów
├── data/raw/              # 700 plików (~1.5 GB) — gitignored, na HF Hub
├── data/manifest.csv      # Mapa plików -> warunki — w gicie
├── data_processed/        # Zarr dataset — gitignored, na HF Hub
├── docs/                  # Opis formatu, ADR, dokumentacja projektu
├── notebooks/             # Eksploracja EDA
└── reports/               # Metryki, wykresy (generowane)
...
```

## Workflow regeneracji datasetu

Zarr generowany jest deterministycznie z raw przez:

```bash
uv run python scripts/01_build_manifest.py \
    --data-dir data/raw \
    --output data/manifest.csv

uv run python scripts/02_ingest_to_zarr.py \
    --manifest data/manifest.csv \
    --output data_processed/dataset.zarr
```

Po regeneracji wykonaj re-upload na HF: `uv run python scripts/upload_to_hf.py`.

## Workflow dla zespołu

### Nazewnictwo branchy
task/<num>-<short-desc>    # np. task/1-static-regression-T
fix/<short-desc>           # np. fix/spectral-shift-nan-handling
docs/<short-desc>          # np. docs/add-data-format-spec

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

## Onboarding — first steps

Po setupie środowiska każdy z zespołu:

1. Czyta `CONTEXT.md` w całości (architektura, open questions, anti-patterns)
2. Bierze sobie jedno z 8 ML tasks z sekcji "ML Tasks" w `CONTEXT.md`
3. Tworzy branch `task/X-...` i notebook eksploracyjny w `notebooks/`
4. **Zanim zacznie feature engineering**: open question #1 (Spectral Shift) musi być wyjaśniony z prowadzącym

## Zespół

Patrz [CONTRIBUTORS.md](CONTRIBUTORS.md).