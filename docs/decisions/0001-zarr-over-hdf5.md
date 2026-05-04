# ADR-0001: Zarr jako primary data format zamiast HDF5

## Status

Przyjęty

## Kontekst

Potrzebujemy wydajnego formatu dla ~1.6 GB surowych danych (700 plików × 41648 punktów × 5 kanałów)
oraz przetworzonych zbiorów. Główne opcje to HDF5 i Zarr.

## Decyzja

Używamy **Zarr v2** jako primary format dla `data_processed/`.

## Uzasadnienie

| Kryterium | Zarr | HDF5 |
|---|---|---|
| Integracja z xarray | Natywna | Przez `netcdf4` lub `h5netcdf` |
| Dask / równoległy odczyt | Cloud-native, brak blokady | Wymaga `HDF5_USE_ROS3_VFD` lub `h5py` z `swmr` |
| DVC + Google Drive | Katalog → wiele małych plików (OK dla chunków) | Jeden plik → łatwiejszy transfer |
| Kompresja blosc/zstd | Wbudowane `numcodecs` | Wymaga konfiguracji |
| Dojrzałość | Nowszy, mniejsza baza użytkowników | HDF5 bardziej ugruntowany |
| Python API | `zarr` + `xarray.to_zarr()` | `h5py` / `netcdf4` |

Decydujące argumenty za Zarr:
1. Natywna kompatybilność z `xarray.Dataset.to_zarr()` — brak konwersji
2. Chunk-level parallelism bez blokad — krytyczne przy trenowaniu modeli na GPU
3. Cloud-native design — przyszłe przeniesienie na GCS/S3 bez zmian kodu

## Konsekwencje

- Zarr v2 (`zarr>=2.16,<3.0`) — **nie** Zarr v3 (breaking changes w API)
- Chunking: `(1, 41648, 5)` — jeden eksperyment per chunk, pełna oś pozycji
- Kompresja: `Blosc(cname='zstd', clevel=5, shuffle=BITSHUFFLE)` — standard dla float32
- Odczyt: `xr.open_zarr("data_processed/dataset.zarr")` (lazy, chunked)
