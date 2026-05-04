# Luna OBR-4600 — Format pliku .txt

## Struktura pliku

Każdy plik pomiarowy ma dokładnie **41664 linii** (CRLF `\r\n`):

```
Linie  1–14   │ Nagłówek (metadata key-value)
Linia     15  │ Pusta linia
Linia     16  │ Nagłówek kolumn
Linie 17–41664│ 41648 wierszy danych
```

### Nagłówek (linie 1–14)

Format: `Key:  value` (dwa lub więcej spacji po dwukropku).
Wyjątek: linia `NOTE: Data stored in this file is not decimated.` — format z jedną spacją,
traktowana jako informacyjna (logowana na DEBUG, nie parsowana jako klucz-wartość).

Przykład (`Pomiar1.txt`):
```
Acquired on 3/23/2026 at 20:45:11
Calibrated on 2/21/2023 at 03:14:48
Device Descriptor:  [none]
Scan Range (nm):  1545.445  --  1588.034
Measurement Type:  Reflection
Group Index:  1.500
Gain:  24 dB
Domain:  Time
Spatial Resolution (mm):  0.100209
Frequency domain window was not applied to measurement data.
Gage length:  1.000766 cm
Sensor spacing:  0.099884 cm
Number of Data Points (in this file):  41648
NOTE: Data stored in this file is not decimated.
```

### Kolumny danych (linia 16 + linie 17–41664)

| Kolumna (oryginalna) | Nazwa wewnętrzna | Typ | Uwagi |
|---|---|---|---|
| `Length (m)` | `length_1_m` | float64 | Pozycja sensorowa, wypełniona w ~41648 wierszach |
| `Length (m)` | `length_2_m` | float64 | Druga oś długości — **sparse**, patrz niżej |
| `Amplitude (dB/mm)` | `amplitude_db_mm` | float64 | Wypełniona we wszystkich wierszach |
| `Spectral Shift (GHz)` | `spectral_shift_ghz` | float64 | **Sparse**, patrz niżej |
| `Spectral Shift Quality` | `spectral_shift_quality` | float64 | **Sparse** (0.0 gdy brak danych) |

Separator: tab. Liczby w notacji naukowej `E+000`.
Puste pola → `NaN` w pandas.

## Anomalia: Sparse Spectral Shift ⚠️

### Obserwacja

Kolumny `length_2_m`, `spectral_shift_ghz` i `spectral_shift_quality` zawierają wartości
**tylko dla pierwszych ~808 wierszy** (zakres `length_1_m ≈ 2.592–2.607 m`).
W pozostałych ~40840 wierszach te kolumny są puste lub zero.

Empirycznie (Pomiar1.txt):
- `spectral_shift_ghz`: ~808 non-null wartości (z 41648)
- Zakres z wartościami: `2.592 m – 2.607 m`

### Sprzeczność z opisem pomiarów

Plik `opis_pomiarow_ML.txt` definiuje dwa zakresy pomiarowe:
1. `2.65999 – 2.80083 m`
2. `3.22034 – 3.36018 m`

W tych zakresach Spectral Shift jest **pusty** w plikach .txt.

### Status

**Open question** — wymaga wyjaśnienia z prowadzącym projektu.
Możliwe interpretacje:
- Spectral Shift jest wyeksportowany oddzielnie / w innym formacie
- Zakres z wartościami to artefakt kalibracji czujnika
- Pliki zawierają więcej danych niż opisano w `opis_pomiarow_ML.txt`

**Decyzja tymczasowa**: wczytujemy wszystkie 5 kolumn as-is (nie filtrujemy, nie interpolujemy).
Feature engineering zostaje zawieszony do wyjaśnienia.

## Szczególne przypadki w nazewnictwie plików

| Wzorzec | Przykład | Działanie |
|---|---|---|
| `Pomiar{N}.txt` | `Pomiar1.txt` | Normalny ingest |
| `Pomiar{N} (1).txt` | `Pomiar10 (1).txt` | `has_duplicate_marker=True` w manifeście |
| `*.tif`, inne | `zakres_odpowiedzi_czujnika.tif` | `notes="non-measurement file, skipped"` |

Duplikaty (`has_duplicate_marker=True`) są wykluczane z ingestu do Zarr do czasu ręcznej weryfikacji.
