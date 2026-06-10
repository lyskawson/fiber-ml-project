# Statyczna regresja T i RH — wyniki i wnioski

> Materiał pod prezentację końcową projektu *fiber-ml-project*.
> Zakres: **Zadanie 1 (statyczna regresja T)** i **Zadanie 2 (statyczna regresja RH)**.
> Źródła: skrypty `scripts/04–08`, metryki w `reports/metrics/`, moduły `src/fiber_ml/`.
> Wszystkie liczby pochodzą bezpośrednio z plików CSV w `reports/metrics/`.

---

## 1. TL;DR (slajd otwierający)

- **Najlepszy model ogółem: SVR (RBF, strojony)** — najlepszy lub bliski najlepszego w obu reżimach oceny.
  - Interpolacja: **T MAE = 0.045 °C**, **RH MAE = 0.614 %RH**.
  - Ekstrapolacja (LOCO): **T MAE = 0.099 °C**, **RH MAE = 1.13 %RH** — najlepszy ze wszystkich modeli na obu targetach.
- **T jest łatwe, RH jest trudne.** Temperaturę przewiduje niemal każdy model z R² > 0.999; cała różnica między modelami rozgrywa się na RH.
- **Modele drzewiaste (RF, GB) zapadają się przy ekstrapolacji** — RF RH MAE rośnie z ~0.5 (interpolacja) do **8.9 %RH** (LOCO). To podręcznikowy przykład braku ekstrapolacji u drzew.
- **Modele głębokie (CNN1D, GRU) na surowym profilu wypadają gorzej niż klasyka** — za mało danych (~560 próbek treningowych), brak strojenia. Wynik negatywny, raportowany uczciwie.
- **Najmocniejszy sygnał na RH to cecha cross-channel `diff_mean` = mean(CH2) − mean(CH1)** (residuum kanału poliimidowego względem referencyjnego).

---

## 2. Problem i dane

**Cel:** z danych rozproszonego czujnika światłowodowego (Luna OBR-4600) odtworzyć temperaturę **T** i wilgotność względną **RH** w reżimie statycznym (pomiary po godzinnej stabilizacji).

**Dane:**
- 700 plików pomiarowych, ingest do `dataset.zarr`, kształt **(700, 41648, 5)**.
- **35 stanów** = 5 temperatur (35, 45, 55, 65, 75 °C) × 7 wilgotności (20–80 %RH co 10) × **20 replik** każdy.
- Dwa kanały sensorowe wyodrębniane po współrzędnej `length_2_m` (rozwiązanie Open Question #1 z `CONTEXT.md`, zaimplementowane w `preprocessing/channels.py`):
  - **CH1_REF_T** — kanał referencyjny, dominująca czułość temperaturowa (`length_2_m ∈ [2.660, 2.801]`, ~141 punktów).
  - **CH2_PI_TRH** — kanał poliimidowy, czułość T + RH (`length_2_m ∈ [3.220, 3.360]`, ~140 punktów).
- Para `(y1, y2)` z `opis_pomiarow_ML.txt` budowana pozycyjnie; kanały przycięte do **140 punktów** dla parowania 1:1.

---

## 3. Metodologia

### 3.1 Dwie reprezentacje wejścia

| Reprezentacja | Wymiar | Modele |
|---|---|---|
| **Cechy agregowane** (Level 1) | 22 cechy / pomiar | ridge, lasso, random_forest, gradient_boosting, SVR, MLP |
| **Surowy profil** (Level 2) | 2 kanały × 140 punktów `spectral_shift_ghz` | CNN1D, GRU |

22 cechy agregowane (`features/aggregated.py`): per kanał — `mean, std, min, max, median, p25, p75, range, grad_mean_abs, q_mean, q_frac_low` (×2 kanały = 22), plus cross-channel `diff_mean, diff_median, ratio_std`.

### 3.2 Dwa protokoły oceny (kluczowe dla interpretacji)

1. **Replicate split (14/3/3)** — `models/splits.py:replicate_split`.
   Wszystkie 35 stanów obecne w train/val/test, ale **inne repliki**. Mierzy **interpolację**: jak model radzi sobie na nowych powtórzeniach znanych stanów.

2. **LOCO-CV — Leave-One-Condition-Out** (35 foldów) — `models/splits.py:loco_cv`.
   Z treningu usuwany jest **cały jeden stan (T, RH)**; test = wszystkie repliki tego stanu. Mierzy **generalizację do nieznanej kombinacji (T, RH)**.

   > ⚠️ **Ważne doprecyzowanie (na obronę przed pytaniem komisji):** LOCO usuwa *kombinację* (T, RH), a nie poziom T ani poziom RH. Przy 5 wartościach T i 7 RH, gdy odejmiemy np. (35, 20), model dalej widzi T=35 w sześciu innych stanach i RH=20 w czterech. **Żaden brzegowy poziom nigdy nie jest naprawdę nieznany.** LOCO testuje więc interpolację między znanymi poziomami nieznanej *pary*, a nie ekstrapolację do nieznanej temperatury/wilgotności. Twierdzenie o generalizacji formułujemy precyzyjnie: „generalizacja do nieznanej **kombinacji** (T, RH)".

### 3.3 Brak wycieku danych (no leakage)
- Podział **po stanie / replice**, nigdy po wierszach (`splits.py`, sekcja 7 specyfikacji).
- `Split.assert_disjoint()` waliduje rozłączność foldów.
- Strojenie hiperparametrów (SVR, MLP) wyłącznie na zbiorze treningowym przez `GridSearchCV` (3-fold CV) — test pozostaje niewidziany.

### 3.4 Metryki (`eval/metrics.py`)
- **MAE / RMSE / R² / max_abs**, raportowane **osobno dla T i RH** (nigdy uśrednione w jedną liczbę).
- Dodatkowo per-condition (rozkład błędu po siatce 35 stanów).
- **Wiodąca metryka: MAE w jednostkach fizycznych (°C, %RH).** R² na T jest niemiarodajne — zawyżone szerokim zakresem T (35–75) i praktycznie nie różnicuje modeli.

---

## 4. Wyniki

### 4.1 Interpolacja — Replicate split (test 14/3/3)

| Model | T MAE [°C] | T RMSE | T R² | RH MAE [%RH] | RH RMSE | RH R² |
|---|---:|---:|---:|---:|---:|---:|
| ridge | 0.119 | 0.153 | 0.99988 | 1.318 | 1.760 | 0.99225 |
| lasso | 0.116 | 0.149 | 0.99989 | 1.303 | 1.736 | 0.99246 |
| random_forest | 0.091 | 0.202 | 0.99980 | 0.521 | 1.309 | 0.99571 |
| gradient_boosting | ~0.0 ⚠️ | ~0.0 | 1.00000 | 1.157 | 1.964 | 0.99036 |
| **SVR (RBF, strojony)** | **0.045** | 0.067 | 0.99998 | 0.614 | 0.930 | 0.99784 |
| **MLP (strojony)** | 0.056 | 0.071 | 0.99997 | **0.456** | 0.575 | 0.99917 |
| CNN1D (surowy profil) | 0.710 | 0.826 | 0.99659 | 1.322 | 1.755 | 0.99230 |
| GRU (surowy profil) | 0.322 | 0.443 | 0.99902 | 1.086 | 1.561 | 0.99391 |

**Zwycięzcy interpolacji:** SVR na T (0.045 °C), **MLP na RH (0.456 %RH)**.

⚠️ `gradient_boosting` ma T MAE ≈ 2.5·10⁻⁶ — patrz §6 (memoryzacja dyskretnych poziomów, **nie** wyciek).

### 4.2 Generalizacja — LOCO-CV (35 foldów, średnia ± odch. std (max))

| Model | T MAE [°C] | T max | RH MAE [%RH] | RH max |
|---|---:|---:|---:|---:|
| ridge | 0.139 ± 0.090 | 0.450 | 1.672 ± 1.277 | 6.51 |
| lasso | 0.139 ± 0.099 | 0.446 | 1.644 ± 1.274 | 6.51 |
| random_forest | 1.836 ± 2.362 | 9.70 | **8.878 ± 2.190** | 12.60 |
| gradient_boosting | ~0.0 ⚠️ | ~0.0 | 5.994 ± 3.140 | 13.29 |
| **SVR (RBF, strojony)** | **0.099 ± 0.092** | 0.475 | **1.127 ± 1.430** | 7.73 |
| MLP (strojony) | 0.100 ± 0.086 | 0.423 | 1.631 ± 2.051 | 9.38 |
| CNN1D (surowy profil) | 0.976 ± 0.664 | 2.39 | 4.309 ± 3.605 | 13.97 |
| GRU (surowy profil) | 0.624 ± 0.460 | 2.16 | 2.867 ± 2.234 | 8.99 |

**Zwycięzca generalizacji: SVR na obu targetach** (T 0.099 °C, RH 1.127 %RH).

> Pliki źródłowe: `loco_all_models_summary.csv` (ridge/lasso/RF/GB) + `{svr,mlp,cnn1d,gru}_loco_summary.csv`.

### 4.3 Najlepsze hiperparametry (strojone)
- **SVR:** `kernel=rbf, C=1000, gamma=0.01, epsilon=0.01` (GridSearchCV, 36 kombinacji × 3-fold).
- **MLP:** `solver=lbfgs, hidden_layer_sizes=(64, 64), alpha=1e-6` (100 kombinacji Adam+L-BFGS × 3-fold). Zwyciężył L-BFGS — typowe dla małych zbiorów.

---

## 5. Analiza

### 5.1 T jest łatwe, RH jest trudne
Temperatura jest odtwarzana niemal idealnie (R² > 0.999, MAE < 0.15 °C dla modeli liniowych) — czujnik światłowodowy ma silną, monotoniczną i niemal liniową odpowiedź termiczną. Wilgotność to słabszy, nieliniowy sygnał wtórny: cały „wyścig" modeli toczy się na RH. **Każde porównanie modeli należy prowadzić po RH MAE.**

### 5.2 Dlaczego SVR jest defensywnym headline'em
SVR jest najlepszy **lub bliski najlepszego w obu reżimach** — nie tylko w jednym. MLP wygrywa interpolację RH (0.456), ale jego RH przy LOCO degraduje się do 1.63 z bardzo wysoką wariancją (std 2.05, max 9.38) — model dopasowuje się do znanych stanów, ale jest niestabilny na nieznanych kombinacjach. SVR utrzymuje RH MAE 1.13 przy LOCO. **Zastrzeżenie:** nawet SVR ma skok do **7.73 %RH w narożniku siatki (T=35, RH=20)** — patrz §5.5.

### 5.3 Zapadanie się modeli drzewiastych (czysty wynik negatywny)
RF i GB świetnie interpolują (RF RH 0.52, najlepszy na replicate split po MLP), ale **nie ekstrapolują**:
- RF: RH MAE **0.52 → 8.88** (interpolacja → LOCO), T MAE 0.09 → 1.84.
- GB: RH MAE 1.16 → 5.99.

Drzewa przewidują przez średnie w liściach — dla nieobecnego w treningu stanu zwracają wartość najbliższego *widzianego* liścia, co przy brzegach siatki daje duży błąd. To najczytelniejsza ilustracja „drzewa nie ekstrapolują" w całym zestawieniu.

### 5.4 Modele głębokie poniżej klasyki
CNN1D i GRU na surowym profilu 2×140 wypadają wyraźnie gorzej (GRU > CNN1D, ale oba > SVR/MLP pod względem błędu). Przyczyny: ~560 próbek treningowych to za mało dla sieci, brak strojenia hiperparametrów, a ręcznie zaprojektowane cechy agregowane już kompresują istotną informację. Prezentujemy je jako **uczciwy baseline / wynik negatywny**, nie ukrywamy.

### 5.5 Efekty brzegowe (gdzie modele się mylą)
Największe błędy RH koncentrują się w **narożnikach siatki** — niskie T (35, 45) i skrajne RH (20, 30, 80):

| Model | Najgorszy fold (T, RH) | RH MAE |
|---|---|---:|
| ridge | (35, 20) | 6.51 |
| SVR | (35, 20) | 7.73 |
| MLP | (35, 20) | 9.38 |

Dla ridge **8 z 10 najgorszych foldów** leży na brzegach RH (20/30/70/80). To spójne z §3.2: narożniki to stany najbliższe prawdziwej ekstrapolacji, więc tam błąd rośnie.

### 5.6 Interpretowalność
Modele liniowe (ridge/lasso) są zaskakująco mocne i **stabilne** (RH LOCO 1.64–1.67, najniższa wariancja po SVR) — dowód, że proste cechy agregowane niosą większość sygnału. Kluczowa cecha to **`diff_mean` = mean(CH2) − mean(CH1)**: residuum kanału poliimidowego względem referencyjnego, fizyczny kandydat na sygnał wilgotności (CH1 kompensuje wkład temperaturowy). Daje to czytelną narrację „trusted AI": wynik jest tłumaczalny, nie czarnoskrzynkowy.

---

## 6. Ograniczenia i zastrzeżenia (slajd uczciwości)

1. **Co naprawdę testuje LOCO** (§3.2): usuwana jest kombinacja (T, RH), nie poziom. **Nie przeprowadzono** prawdziwego testu ekstrapolacji poziomu (leave-one-T-level-out / leave-one-session-out — ten drugi to świadomy `NotImplementedError` w `splits.py`, wymaga `session_id` w manifeście). Twierdzenia o generalizacji ograniczamy do nieznanej *kombinacji*.
2. **„Idealne" T u gradient_boosting (MAE ≈ 2.5·10⁻⁶) to nie wyciek danych** — T nie jest cechą. To efekt memoryzacji: T jest dyskretnym targetem 5-poziomowym, a każdy poziom jest zawsze obecny w treningu (LOCO usuwa parę, nie poziom), więc ensemble drzew odczytuje zapamiętaną średnią kubełka. Model liniowy tego nie potrafi (stąd uczciwe 0.139). Nie traktować jako sukcesu fizycznego.
3. **CNN1D / GRU nie były strojone** i trenowane na ~560 próbkach — ich słaby wynik to ograniczenie setupu, nie dowód bezużyteczności architektury.
4. **Open Question #1** (sparse spectral shift) został **rozwiązany** w `channels.py` (zakresy mapują się na `length_2_m`, nie `length_1_m`) — raport jest wewnętrznie spójny z `CONTEXT.md`.

---

## 7. Rekomendacja i kolejne kroki

- **Model produkcyjny: SVR (RBF, C=1000, gamma=0.01).** Najlepszy kompromis interpolacja/ekstrapolacja, w pełni interpretowalny pipeline (StandardScaler → SVR), tani w treningu.
- **Do prezentacji:** prowadzić narrację po **RH MAE** i pokazać oba protokoły obok siebie (interpolacja vs LOCO) — to najmocniej różnicuje modele.
- **Dalej:**
  - leave-one-T-level-out i leave-one-session-out (prawdziwa ekstrapolacja poziomu),
  - feature importance / współczynniki ridge dla potwierdzenia roli `diff_mean`,
  - większy zbiór lub augmentacja dla modeli głębokich, jeśli mają konkurować.

---

## Załącznik — mapa plików

| Plik | Zawartość |
|---|---|
| `reports/metrics/baseline_per_target.csv` | ridge/lasso/RF/GB — replicate split |
| `reports/metrics/{svr,mlp,cnn1d,gru}_per_target.csv` | pozostałe modele — replicate split |
| `reports/metrics/loco_all_models_summary.csv` | ridge/lasso/RF/GB — LOCO |
| `reports/metrics/{svr,mlp,cnn1d,gru}_loco_summary.csv` | pozostałe modele — LOCO |
| `reports/metrics/*_per_condition.csv` | rozkład błędu po 35 stanach |
| `reports/metrics/{svr,mlp}_best_params.txt` | wybrane hiperparametry |
| `reports/figures/eda/*.png` | EDA: pokrycie stanów, profile kanałów, powierzchnia odpowiedzi, wariancja replik, jakość |
| `scripts/04–08_train_*.py` | skrypty treningowe |
