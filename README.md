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

Wymóg Trusted AI: pełna reprodukowalność, wersjonowanie danych (DVC), kontrola data leakage, interpretowalność modeli.

## Quickstart

```bash
git clone <repo-url>
cd fiber-ml-project

# Instalacja środowiska (Python 3.11)
uv sync --extra dev

# Pobranie danych (wymaga skonfigurowanego DVC remote — patrz sekcja DVC Setup)
dvc pull

# Testy na danych sample (wbudowane w repo)
uv run pytest tests/ -v

# Ingest na danych sample (test pipeline bez pełnych danych)
uv run python scripts/01_build_manifest.py --data-dir data/sample --output data/manifest_sample.csv
uv run python scripts/02_ingest_to_zarr.py --manifest data/manifest_sample.csv --output /tmp/sample.zarr --sample
```

## Struktura repo

```
.
├── src/fiber_ml/          # Główny pakiet Python
│   ├── ingest/            # Parser .txt, budowanie manifestu, konwersja do Zarr
│   ├── preprocessing/     # (TBD) normalizacja, segmentacja
│   ├── features/          # (TBD) feature engineering
│   ├── models/            # (TBD) modele ML
│   ├── eval/              # (TBD) metryki, wykresy
│   └── utils/             # Ścieżki, helpery
├── scripts/               # CLI: 01_build_manifest, 02_ingest_to_zarr
├── tests/                 # Testy pytest (działają na data/sample/)
├── data/sample/           # 2 pliki pomiarowe (T35_RH20) — w gicie
├── data/raw/              # 700 plików (~1.6 GB) — DVC tracked
├── data_processed/        # Zarr dataset — DVC tracked
├── docs/                  # Opis formatu, ADR, dokumentacja projektu
├── notebooks/             # Eksploracja EDA
├── reports/               # Metryki, wykresy (generowane)
├── dvc.yaml               # Pipeline DAG
└── params.yaml            # Hiperparametry i konfiguracja
```

## DVC Setup

### 1. Utwórz folder na Google Drive

Utwórz pusty folder na swoim Google Drive. Skopiuj **Folder ID** z URL:
`https://drive.google.com/drive/folders/<FOLDER_ID>`

### 2. Zaktualizuj konfigurację DVC

```bash
# Podmień FOLDER_ID_PLACEHOLDER na właściwe ID
dvc remote modify gdrive url gdrive://<FOLDER_ID>
```

### 3. Skonfiguruj Service Account (zalecane dla zespołu)

Dokumentacja: [DVC Google Drive Service Account](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)

```bash
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path path/to/credentials.json
```

> **Uwaga**: Plik `credentials.json` dodaj do `.gitignore` — **nie commituj sekretów**.

### 4. Pierwsze pobranie danych

```bash
dvc pull
```

## Workflow dla zespołu

### Nazewnictwo branchy

```
task/<num>-<short-desc>    # np. task/1-static-regression-T
```

### Conventional commits

```
feat: add static T regression model
fix: handle NaN in spectral shift channel
docs: update data_format.md with supervisor clarification
chore: bump zarr to 2.18
test: add regression test for sparse spectral shift
```

### Pull Request

- Branch z `main`, PR do `main`
- CI musi przejść (lint + typecheck + testy)
- Minimum 1 reviewer

## Zespół

Patrz [CONTRIBUTORS.md](CONTRIBUTORS.md).
