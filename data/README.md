# Dane projektowe

## Struktura

```
data/
├── manifest.csv          # Indeks wszystkich plików (generowany przez skrypt)
├── sample/               # 2 pliki próbkowe — commitowane w gicie
│   └── T35_RH20/
│       ├── Pomiar1.txt
│       └── Pomiar3.txt
└── raw/                  # 700 plików pomiarowych (~1.6 GB) — DVC tracked, NIE w gicie
    └── T{X}_RH{Y}/
        └── Pomiar{N}.txt
```

## Instrukcja kopiowania danych surowych

Po sklonowaniu repo i wykonaniu `uv sync`, Alek kopiuje dane ręcznie:

```bash
# Skopiuj katalog z pomiarami do data/raw/
cp -r /Users/aleklyskawa/Desktop/project_szczupak/project/ data/raw/

# Sprawdź strukturę
ls data/raw/ | head -10
# Oczekiwane: T35_RH20/ T35_RH30/ ... T75_RH80/

# Zbuduj manifest
uv run python scripts/01_build_manifest.py --data-dir data/raw --output data/manifest.csv

# Wgraj do DVC remote (po skonfigurowaniu Google Drive — patrz README.md)
dvc add data/raw
dvc push
```

## Konwencja katalogów

`T{X}_RH{Y}/` — temperatura X °C, wilgotność względna Y %RH

- X ∈ {35, 45, 55, 65, 75}
- Y ∈ {20, 30, 40, 50, 60, 70, 80}
- 35 kombinacji × 20 replik = 700 plików

## Znane anomalie

- `T65_RH50/` i `T75_RH40/` — pliki z sufiksem ` (1)` (np. `Pomiar10 (1).txt`) → duplikaty po ponownym pobraniu z Drive. Flagowane jako `has_duplicate_marker=True` w manifeście.
- `T75_RH40/zakres_odpowiedzi_czujnika.tif` — nie jest plikiem pomiarowym. Skopiuj do `docs/references/sensor_response_range.tif`.
- Spectral Shift jest sparse (~808 z 41648 wierszy) — patrz `docs/data_format.md`.
