# Setup — Kroki manualne (Alek)

Repo jest gotowe lokalnie. Poniżej co musisz zrobić ręcznie.

## 1. Opublikuj repo na GitHub

```bash
gh repo create fiber-ml-project --public --source=. --push
```

Lub ręcznie:
```bash
git remote add origin https://github.com/<twoj-username>/fiber-ml-project.git
git push -u origin main
```

## 2. Skonfiguruj DVC remote (Google Drive)

1. Utwórz pusty folder na Google Drive (np. `fiber-ml-dvc-remote`)
2. Otwórz folder w przeglądarce — skopiuj **Folder ID** z URL:
   `https://drive.google.com/drive/folders/<FOLDER_ID>`
3. Podmień placeholder w konfiguracji DVC:
   ```bash
   dvc remote modify gdrive url gdrive://<FOLDER_ID>
   git add .dvc/config
   git commit -m "chore: set dvc gdrive remote folder id"
   git push
   ```

## 3. Skonfiguruj Service Account (zalecane dla zespołu)

Bez service account każdy członek zespołu będzie musiał osobno autoryzować OAuth.
Dokumentacja: https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive

```bash
# Po stworzeniu service account i pobraniu credentials.json:
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path path/to/credentials.json
# NIE commituj credentials.json — jest w .gitignore
```

## 4. Wgraj pliki referencyjne

```bash
cp /path/to/OBR-4600-UG6_SW3.10.1.pdf docs/references/
cp /path/to/sensors-24-07913.pdf docs/references/
# sensor_response_range.tif pochodzi z T75_RH40/ w danych surowych
cp /path/to/project_szczupak/project/T75_RH40/zakres_odpowiedzi_czujnika.tif \
   docs/references/sensor_response_range.tif
```

## 5. Skopiuj dane surowe i wgraj do DVC

```bash
# Skopiuj dane
cp -r /Users/aleklyskawa/Desktop/project_szczupak/project/ data/raw/

# Sprawdź strukturę (powinno być 35 katalogów T{X}_RH{Y})
ls data/raw/ | wc -l

# Zbuduj pełny manifest
uv run python scripts/01_build_manifest.py --data-dir data/raw --output data/manifest.csv

# Sprawdź manifest — weryfikuj ręcznie duplikaty
grep "True" data/manifest.csv  # has_duplicate_marker=True

# Zarejestruj dane w DVC i wypchnij
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "data: add raw data dvc tracking"
dvc push
```

## 6. Wyślij link zespołowi

Instrukcja onboardingu dla nowych członków zespołu:
```bash
git clone https://github.com/<twoj-username>/fiber-ml-project.git
cd fiber-ml-project
uv sync --extra dev
# Skonfiguruj DVC credentials (patrz README.md sekcja DVC Setup)
dvc pull
uv run pytest tests/ -v
```

## 7. Opcjonalnie: zainstaluj pre-commit hooks lokalnie

```bash
uv run pre-commit install
```

## Weryfikacja anomalii z danymi

Przed startą ML wyjaśnij z prowadzącym:
- **Spectral Shift sparsity** — patrz `docs/data_format.md` i `CONTEXT.md` (open questions)
- **Duplikaty plików** z ` (1)` — sprawdź które pliki są kanoniczne (T65_RH50, T75_RH40)
