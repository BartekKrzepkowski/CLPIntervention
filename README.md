# CLPIntervention

Repozytorium badawcze do analizy asymetrii uczenia dwóch gałęzi sieci wizyjnej oraz krytycznych okresów plastyczności. Obraz jest dzielony na lewy i prawy obszar widzenia, a jakość prawej modalności i dostępność lewej gałęzi zmieniają się w czterech fazach treningu.

Kod jest przeznaczony do eksperymentów badawczych, a nie do wdrożenia produkcyjnego. Aktywna implementacja znajduje się w `src/` oraz `scripts/python_new/`.

Identyfikatory najważniejszych eksperymentów W&B, odpowiadające im joby Slurm
i warianty protokołu są utrzymywane w [`docs/WANDB_RUN_INDEX.md`](docs/WANDB_RUN_INDEX.md).

Najkrótsza ścieżka po dokumentacji: ten plik opisuje cel i obsługę kodu,
[`docs/INTERVENTION_EXPERIMENTS.md`](docs/INTERVENTION_EXPERIMENTS.md)
wyjaśnia aktualne pytania naukowe i wyniki, a
[`docs/STATUS.md`](docs/STATUS.md) podaje bieżący stan prac i jobów.

## Protokół eksperymentalny

Dla obrazu o szerokości `W` każda modalność ma szerokość `ceil(W * (0.5 + overlap / 2))`. `overlap=0.0` daje dwie połówki, a większa wartość zwiększa wspólny obszar obu pól.

| Faza | Lewa gałąź | Prawa gałąź | Dane prawej modalności | Cel |
|---|---|---|---|---|
| 1 | aktywna | aktywna | rozmyte, poza opcjonalnym `subset` | wytworzenie początkowej przewagi lewej modalności |
| 2 | aktywna | aktywna | poprawne | obserwacja, czy prawa gałąź nadrabia po przywróceniu informacji |
| 3 | wyłączona przez `deactivation` | aktywna | poprawne | interwencja plastyczności wymuszająca uczenie prawej gałęzi |
| 4 | aktywna ponownie | aktywna | poprawne | pomiar trwałości interwencji i ponownej konkurencji gałęzi |

Testy ewaluacyjne obejmują wariant `test_proper` oraz `test_blurred`. Główne miary asymetrii obejmują dokładność, normy gradientów i wag, długość trajektorii parametrów, odległość między gałęziami, martwe jednostki ReLU, ślad FIM oraz RSV. Całe repozytorium używa jednej konwencji Kleinmana: `RSV=(SV_left-SV_right)/(SV_left+SV_right)`, więc `+1` oznacza zależność od lewej modalności, a `-1` od prawej.

`phase4` jest maksymalnym budżetem fazy 4. Opcjonalne `phase4_target_train_acc=0.x` kończy ją po pierwszej epoce, w której treningowa accuracy osiągnie końcową accuracy właściwego runu bazowego, zgodnie z metodą artykułu.

## Profil walidacyjny CIFAR-10

Nowy, jawnie wybierany profil `cifar10_stratified_44k_5k_1k_seed83_v1` używa stałego podziału 44k train / 5k validation / 1k FIM, osobnych statystyk `proper_left`, `proper_right` i `blurred_right` policzonych wyłącznie na train 44k oraz oryginalnego testu 10k. Indeksy, targety i pliki datasetu są weryfikowane przez SHA-256. Profil historyczny 50k pozostaje dostępny i nie jest zmieniany po cichu.

`phase2_stopping` i `phase3_stopping` obsługują `disabled`, `observe_only` i `enforce`. Eksperymentalny `decision_rule=local_accuracy` łączy sparowany cel weak≥dominant, lokalne odwrócenie Pareto w przestrzeni accuracy `(weak, full, dominant)` oraz futility wymagające jednocześnie plateau weak i szkody dla full/dominant albo konfliktu gradientów na `validation_proper`. Nie używa bezwzględnego spadku względem stanu sprzed Phase 3. Weak-only loss i stały train probe pozostają diagnostyczne. W profilu walidacyjnym każda faza dostaje świeży optimizer i scheduler; checkpoint zachowuje ich stan wyłącznie do resume w tej samej fazie.

Profil kalibracyjny Phase 3 stosuje jawny, czteroepokowy liniowy warm-up LR
(`phase3_lr_warmup_epochs=4`, start od 10% bazowego LR). LR rośnie po każdym
kroku optymalizatora i osiąga wartość bazową po ostatnim kroku czwartej epoki.
Po warm-upie LR pozostaje stałe: Phase 3 nie używa schedulera epokowego.
Warm-up jest częścią schedulera, więc jego stan jest zachowywany przez resume.
Przy wartości zero Phase 3 od początku używa stałego bazowego LR i nadal nie
uruchamia schedulera epokowego.

Opt-in `phase4_diagnostics` mierzy wczesny compatibility drift po reaktywacji
obu gałęzi. Zachowuje anchor z końca Phase 3 i porównuje weak-only dla
`current_right+anchor_shared` oraz `anchor_right+current_shared`, dzięki czemu
oddziela dryf prawego encodera od dryfu wspólnego downstream i klasyfikatora.
Opcjonalny `phase4_lr_warmup_epochs` używa tego samego schedulera per-step co
warm-up Phase 3; wartość zero zachowuje dotychczasową P4. Metoda i pierwszy
sweep są opisane w
[docs/PHASE4_COMPATIBILITY_DIAGNOSTIC_2026-08-02.md](docs/PHASE4_COMPATIBILITY_DIAGNOSTIC_2026-08-02.md).

Dla metodologicznej kontroli stoppera można ustawić
`observe_phase4_transition=hypothetical_selected`: Phase 3 nadal wykonuje cały
shadow trace, ale Phase 4 startuje z checkpointu wskazanego przez pierwszą
zamrożoną decyzję. Milestone checkpointy służą następnie do osobnych porównań
z ustalonymi długościami interwencji.

Test nigdy nie jest wejściem stoppera ani selektora. `phase4_test_policy=final_only` ocenia wybrane checkpointy Phase 4 dopiero po treningu. Opcjonalne `phase2_test_policy=posthoc_final` po zakończeniu wszystkich faz odtwarza zachowane minimum-loss i maksimum-accuracy Phase 2, loguje `posthoc_test/phase2/*`, a następnie przywraca końcowy checkpoint. Wgląd w te wyniki jest diagnostyczny i nie może zmieniać Phase 3/4 ani progów. Szczegółowa metodologia, ranking, fallbacki i zasady kalibracji są w [docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md).

Post-hoc porównanie dwóch seed-paired clean unimodal references można wykonać
na compute nodzie przez `scripts.python_new.evaluate_unimodal_ensemble`.
Wynik główny to equal-weight mean posterior probabilities; skrypt raportuje
też mean logits jako analizę czułości, pojedyncze modele oraz clean-gold full
na `validation_proper` i końcowym `test_proper`. Szczegóły i aktualne wyniki:
[docs/UNIMODAL_ENSEMBLE_EVALUATION_2026-08-02.md](docs/UNIMODAL_ENSEMBLE_EVALUATION_2026-08-02.md).

## Struktura repozytorium

```text
src/
  configs/                 konfiguracje modeli, strat i środowiska
  data/                    datasety bimodalne i transformacje
  modules/architectures/   MLP, CNN, ResNet, EfficientNet, ConvNeXt i wrapper UMT
  modules/                 straty, regularizatory, hooki i metryki
  trainer/                 standardowy trener CLP i trener UMT
  utils/                   rejestry, budowanie obiektów, checkpointy
  visualization/           backendy W&B, TensorBoard i ClearML
scripts/python_new/         aktywne entrypointy eksperymentów
scripts/python_old/         kod archiwalny, nieutrzymywany
scripts/python_backup/      kopie archiwalne, nieutrzymywane
tests/                      testy regresyjne CPU
notebooks/                  archiwalne notebooki analiz i wizualizacji
docs/papers/                artykuły stanowiące podstawę metodologiczną
docs/STATUS.md              bieżący stan, aktywne joby, decyzje i następne kroki
docs/AUDIT.md               wyniki audytu i otwarte ograniczenia
docs/CONCEPTS.md            słownik i dziennik pojęć
docs/VALIDATION_PROTOCOL.md wersjonowany split i walidacyjne sterowanie fazami
docs/INTERVENTION_EXPERIMENTS.md aktualna narracja eksperymentalna i decyzje
docs/work_logs/             krótki chronologiczny dziennik prac
```

Ważne rozróżnienie: `scripts/python_new/run_all_at_once.py` uruchamia standardowy protokół z diagnostyką FIM, natomiast główny plik `run_all_at_once.py` jest wariantem eksperymentu z `balance_loss`.

## Instalacja

Na PLGrid repozytorium ma dwa odseparowane środowiska:

```text
/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-local-cpu
/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-gh200
```

Lokalne środowisko CPU tworzy i weryfikuje:

```bash
bash scripts/bash/create_local_env.sh
scripts/bash/clp_python_local.sh -c "import torch; print(torch.__version__)"
```

Środowisko GPU musi powstać natywnie na compute nodzie GH200 (`aarch64`). Skrypt `scripts/bash/run_gpu_tests.sh` tworzy je przy pierwszym zadaniu i od razu uruchamia testy GPU:

```bash
mkdir -p slurm_logs
sbatch scripts/bash/run_gpu_tests.sh
```

Bazowe środowiska `lapsum-local-cpu` i `lapsum-gh200` służą wyłącznie jako źródła klonowania. Z tego powodu lokalny prefiks zawiera obecnie 316 pakietów i zajmuje około 3,0 GB, a prefiks GH200 około 8,0 GB. Aktywny kod nie używa JupyterLab/IPython, pandas, statsmodels, torchaudio, pyarrow, Hugging Face `datasets` ani seaborn. Minimalny zestaw znajduje się w `environment.yml`; TensorBoard, ClearML, Captum i matplotlib należą do `environment-optional.yml`, ponieważ są potrzebne tylko alternatywnym loggerom lub wizualizacji wag.

Nie należy usuwać pakietów pojedynczo z działającego prefiksu, bo solver może usunąć zależności PyTorch. Bezpieczna procedura odchudzenia to utworzenie nowych prefiksów z `environment.yml`, wykonanie pełnych bramek CPU/GPU i dopiero potem przełączenie launcherów. Obecne środowiska pozostają punktem odniesienia do czasu takiej migracji.

Skrypty klastrowe instalują domyślnie tylko zależności testowe wymagane przez core. Captum i TensorBoard można dołożyć jawnie:

```bash
CLP_INSTALL_OPTIONAL=1 bash scripts/bash/create_local_env.sh
```

## Konfiguracja środowiska

Ścieżek i sekretów nie należy wpisywać do repozytorium. Ustaw je w powłoce lub w prywatnym pliku poza Git:

```bash
export REPORTS_DIR=/path/to/reports
export CIFAR10_PATH=/path/to/cifar10
export FMNIST_PATH=/path/to/fashion-mnist
export SVHN_PATH=/path/to/svhn
export MNIST_PATH=/path/to/mnist
export KMNIST_PATH=/path/to/kmnist
export TINYIMAGENET_PATH=/path/to/tiny-imagenet

export WANDB_API_KEY=your-private-key
export WANDB_ENTITY=your-entity
export WANDB_PROJECT=Critical_Periods_Interventions
```

`src/configs/env_variables.sh` jest bezpiecznym, niesekretnym szablonem zachowującym wartości już ustawione w środowisku. Nigdy nie zapisuj prawdziwego `WANDB_API_KEY` w śledzonym pliku.

Pobieranie datasetów jest domyślnie wyłączone (`DOWNLOAD = False`), więc dane muszą już znajdować się pod wskazanymi ścieżkami. Tiny ImageNet powinien mieć katalogi `train/` i `val/` zgodne z `torchvision.datasets.ImageFolder`.

## Szybki start

`REPORTS_DIR` musi wskazywać trwały katalog pod `/net/storage`. Wspólny
helper submission automatycznie tworzy `$REPORTS_DIR/slurm_logs` i kieruje
tam stdout/stderr Slurma. Checkpointy, trace, lokalne dane W&B i surowe
predykcje również pozostają w `REPORTS_DIR`, a nie w worktree repozytorium.
Szczegółowy układ i zasady migracji opisuje
[`docs/ARTIFACT_STORAGE.md`](docs/ARTIFACT_STORAGE.md).
Nazwę joba i zależność można przekazać bez tworzenia nowego launchera przez
`CLP_JOB_NAME` i `CLP_DEPENDENCY`, np. `CLP_DEPENDENCY=afterok:12345`.

Standardowy pełny protokół dla CIFAR-10 przez wspólny launcher Slurm na GH200:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  frozen_config=configs/experiments/paper_cifar10_sresnet18_baseline_seed83.yaml
```

W trybie niezablokowanym launcher nadal przyjmuje argumenty `nazwa=wartość`. Zamrożony baseline celowo odrzuca takie nadpisania. Dla zmienionego wariantu fazy 4 należy po zakończeniu baseline utworzyć nowy zamrożony plik z `phase4_weight_decay=0.0005`, `phase4_lr_lambda=0.98` i liczbowym `phase4_target_train_acc` równym accuracy treningowej baseline; nie wolno używać accuracy testowej.

### Próba FIM

Historyczny profil deterministycznie wybiera z treningu klasowo zbalansowane 2% danych. Nowy profil `cifar10_stratified_44k_5k_1k_seed83_v1` używa stałego, osobnego FIM probe 1 000 przykładów (100/klasę), rozłącznego z train i validation. Hash indeksów i informacja o wykluczeniu są zapisywane w metadanych runu. FIM:

- używa wspólnych próbek etykiet modelu dla obu gałęzi;
- wyklucza parametry BatchNorm i wspólnego trzonu, zgodnie z analizą osobnych ścieżek w artykule;
- izoluje RNG diagnostyki od RNG treningu i odtwarza tryb modelu;
- historycznie jest liczony dwa razy w mierzonej epoce na batchu `0` i `floor(N/2)`; `fim_eval_interval_epochs` pozwala mierzyć tylko co K-tą lokalną epokę, a konfiguracja operacyjna PAIS używa jednego pomiaru co 10 epok z `M=5`.
- raportuje surowy TFIM i `trace1_per_parameter`, `trace2_per_parameter`, `trace_per_parameter` dla wszystkich dopuszczonych parametrów gałęzi (z wyłączeniem BatchNorm), a osobno analogiczne metryki `*_weight*` wyłącznie dla wag; oba zestawy logują jawne mianowniki `parameter_count*`.
- przetwarza probe w deterministycznych chunkach (`fim_chunk_size`, domyślnie 16), sumując wkłady względem pełnej liczby przykładów i próbek etykiet; chunkowanie ogranicza pamięć, ale nie zmienia definicji estymatora.

Opcje to `fim_measurements_per_epoch` (domyślnie `2`, `0` wyłącza pomiary), `fim_eval_interval_epochs`, `fim_sampling_seed`, `fim_samples_per_input` i `fim_chunk_size`; historyczny profil obsługuje dodatkowo `fim_probe_fraction`, `fim_probe_seed` i `fim_exclude_from_training`. Dla wartości `K` runner wybiera dokładnie `min(K, N)` deterministycznych pozycji w każdym `N`-batchowym treningowym loaderze; nie używa globalnego modulo. Historyczne tensory `data/train_<dataset>_held_out_*.pt` można wykorzystać tylko jawnie przez `fim_probe_source=files`; stary notebook nie gwarantuje rozłączności ani sparowania augmentacji i nie jest zalecaną ścieżką.

### Próba RSV

RSV jest analizą post hoc wykonywaną na zapisanych checkpointach, a nie metryką w pętli treningowej. Zalecany run zapisuje checkpoint na końcu każdej z czterech faz, a następnie `scripts.python_new.compute_rsv` przetwarza wszystkie checkpointy na tej samej deterministycznej próbie: domyślnie 5 obrazów na klasę, oryginał i 99 translacji do 4 px oraz rotacji do 10° dla jednego pola przy stałym drugim polu.

Domyślnie podczas tych samych forwardów zapisywane są dwa rozłączne punkty:

- `stage3_avgpool`: analiza wyjścia `main_branch.0` (torchvision `layer3`, klasyczne `conv4_x`) po dodatkowym, wyłącznie analitycznym `AdaptiveAvgPool2d(1)`;
- `stage4_avgpool`: natywne wyjście `avgpool` po `main_branch.1` (`layer4` / `conv5_x`), bezpośrednio przed `fc`, zgodne z reprezentacją Kleinmana.

Pooling po stage 3 nie zmienia forwardu ani treningu modelu. Jest wykonywany wyłącznie na tensorze przechwyconym przez hook. Oba punkty mają po jednej wartości na kanał, ale nie należy łączyć ich rozkładów w jednej analizie.

```bash
scripts/bash/submit_experiment.sh scripts.python_new.compute_rsv \
  --checkpoint phase1=/path/to/phase1.pth \
  --checkpoint phase2=/path/to/phase2.pth \
  --checkpoint phase3=/path/to/phase3.pth \
  --checkpoint phase4=/path/to/phase4.pth \
  --output-dir /path/to/results/rsv \
  --model-name mm_resnet --dataset-name mm_cifar10
```

Każdy plik `.pt` zawiera surowe macierze `rsv`, `source_variance_left` i `source_variance_right`, indeksy oraz etykiety. Towarzyszący `.manifest.json` zapisuje hash surowego pliku, checkpoint i jego hash, punkt pomiaru, pooling, kształt aktywacji, seed, augmentacje, normalizację i kanoniczną konwencję Kleinmana `RSV=(SV_left-SV_right)/(SV_left+SV_right)`. Historyczne wyniki zapisane odwrotnie należy przeliczyć jako `RSV_Kleinman=-RSV_historyczne`.

Porównanie interwencji i kontroli wykonuje hierarchiczny bootstrap sparowanych różnic. Nazwy seedów oraz wybrane indeksy obrazów muszą być zgodne:

```bash
scripts/bash/clp_python_local.sh -m scripts.python_new.bootstrap_rsv \
  --control seed1=/results/control/seed1.phase4.stage4_avgpool.pt \
  --control seed2=/results/control/seed2.phase4.stage4_avgpool.pt \
  --intervention seed1=/results/intervention/seed1.phase4.stage4_avgpool.pt \
  --intervention seed2=/results/intervention/seed2.phase4.stage4_avgpool.pt \
  --output /results/bootstrap/phase4.stage4_avgpool.json
```

Skrypt najpierw agreguje jednostki wewnątrz obrazu, następnie resampluje obrazy klasowo i modele/seedy, po czym raportuje różnicę `intervention-control`, przedział ufności i prawdopodobieństwo bootstrapowe wartości dodatniej. Do wyniku publikacyjnego należy użyć co najmniej pięciu sparowanych seedów.

## Zgodność z artykułem

Historyczny S-ResNet-18 użyty w głównym badaniu odpowiada `model_name=mm_resnet` z `backbone_type=resnet18`, `modify_resnet=true` i sumowaniem gałęzi. Dla CIFAR-10 i Fashion-MNIST trening wykonuje wspólny flip poziomy z prawdopodobieństwem `0.5`, dzieli obraz, a następnie niezależnie przesuwa każdą połówkę maksymalnie o `1/8`; transformacje treningowe nie wykonują rotacji. Rozmycie prawej połówki jest redukcją rozdzielczości przez `resize_factor=0.25` i powrotem do rozmiaru pola.

Repozytorium ma kompletny generator surowego RSV dla protokołu pracy, ale nie odtwarza jeszcze automatycznie korelacji, wyszukiwania bisekcyjnego długości interwencji ani historycznych sweepów. EWMA-10 nie należy do treningu: W&B ma przechowywać surowe punkty, a wygładzanie może być włączone wyłącznie w UI lub skrypcie post hoc. Różnica warstwy pomiarowej względem Kleinman et al. jest jawna w manifeście RSV.

## Uruchamianie pojedynczych faz

Wszystkie pojedyncze fazy obsługuje `scripts.python_new.run_single`. Faza 1 startuje od zera:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  mode=phase1 model_name=mm_resnet dataset_name=mm_tinyimagenet \
  lr=0.5 wd=0.0 phase1=60
```

Kolejne fazy wymagają checkpointu i długości wcześniejszych faz:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  mode=phase2 model_name=mm_resnet dataset_name=mm_tinyimagenet \
  lr=0.5 wd=0.0 phase1=60 phase2=100 \
  model_checkpoint=/path/to/model_step_epoch_60.pth
```

Dla `mode=phase3` podaj `phase1..phase3`, a dla `mode=phase4` — `phase1..phase4`. Dawne moduły `run_phase1`–`run_phase4` pozostają cienkimi wrapperami kompatybilności, ale nie zawierają osobnych implementacji.

Nowe checkpointy są wersjonowanymi pakietami i zawierają:

- `model_state_dict`;
- `optimizer_state_dict`, w tym momentum SGD i weight decay grup parametrów;
- `scheduler_state_dict` oraz stałe mnożniki `MultiplicativeLR`;
- `next_epoch`, `global_step` i stany RNG Pythona, NumPy, CPU oraz CUDA;
- `phase_epoch`, stan stoppera/selektora oraz generatora train loadera;
- `diagnostics_state` z referencją inicjalizacji, poprzedniego pomiaru, początku bieżącej fazy i akumulatorami `RunStatsBiModal`;
- `metadata.protocol_manifest` z wersją protokołu, architekturą, datasetem i normalizacją, hashem subsetu i próby FIM, batch size, seedem, granicami faz oraz ustawieniami optimizera, schedulera, destylacji i kontroli BN.

`model_checkpoint` oznacza rozpoczęcie nowej fazy od wag modelu i domyślnie tworzy nowy optimizer oraz scheduler. Jest to rozdzielone od `resume_checkpoint`, który służy do wznowienia przerwanego runu i domyślnie odtwarza pełny stan. Eksperymentalne przeniesienie optimizera/schedulera między osobnymi fazami wymaga jawnego `transfer_training_state=true`; `restore_training_state=false` może wymusić weights-only przy wznowieniu, ale wymaga wtedy jawnego `resume_start_epoch`. Historyczny wariant `all_at_once` zachowuje ciągły optimizer. Profil walidacyjny celowo tworzy świeży optimizer i scheduler na każdej granicy faz, a ich stan odtwarza tylko przy `resume_checkpoint` wewnątrz tej samej fazy. Stary model-only `state_dict` nie może być użyty jako `resume_checkpoint`; należy przekazać go jako świadomy `model_checkpoint`, ponieważ nie zawiera epoki, optimizera, schedulera ani RNG.

`RunStatsBiModal` zachowuje ciągłość po pełnym resume. Długość trajektorii jest sumą norm rzeczywistych przesunięć parametrów rejestrowanych po każdym `optimizer.step()`, a nie przybliżeniem `lr × ||gradient||`; obejmuje więc weight decay, momentum i różne grupy optimizera. Dodatkowo loguje dla każdej fazy osobne odległości L2 od modelu na jej początku: `left_branch_distance_from_phaseN_start_l2`, `right_branch_distance_from_phaseN_start_l2`, `main_branch_distance_from_phaseN_start_l2` oraz wartość całego modelu.

Przerwany run `all_at_once` można wznowić bez powtarzania ukończonych faz:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_all_at_once \
  model_name=mm_resnet dataset_name=mm_cifar10 lr=0.6 wd=0.0 \
  phase1=80 phase2=200 phase3=96 phase4=200 \
  resume_checkpoint=/path/to/model_step_epoch_120.pth
```

Runner porównuje te własności z `protocol_manifest` checkpointu i przerywa wznowienie z listą różniących się pól. Starszy pełny checkpoint bez manifestu jest domyślnie odrzucany; po ręcznej weryfikacji można jednorazowo użyć `allow_resume_without_protocol_manifest=true`. Wznowienie odbywa się na granicy epoki, nie w środku batcha, i tworzy nowy katalog loggera.

### Kontrola BatchNorm

Główny protokół pozostawia BatchNorm bez zmian. Opcjonalna kontrola diagnostyczna resetuje i rekalibruje przed fazą 4 wyłącznie bufory `running_mean`, `running_var` i `num_batches_tracked` we wspólnym `main_branch`, używając poprawnych danych treningowych obu modalności:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  mode=phase4 model_checkpoint=/path/to/phase3.pth \
  model_name=mm_resnet dataset_name=mm_cifar10 lr=0.6 wd=0.0 \
  phase1=80 phase2=200 phase3=96 phase4=200 \
  phase4_bn_recalibration_batches=50 \
  phase4_bn_recalibration_scope=main_branch
```

Domyślne `phase4_bn_recalibration_batches=0` wyłącza kontrolę i zachowuje dotychczasowy wynik. Rekalibracja działa bez gradientów i kroków optimizera/schedulera, nie zmienia afinicznych parametrów BN ani wag modelu, izoluje RNG i nie jest powtarzana przy wznowieniu wewnątrz fazy 4. Logger zapisuje liczbę modułów/batchy oraz normy zmian buforów. Porównanie `native_bn` i `recalibrated_shared_bn` powinno wychodzić z tego samego checkpointu po fazie 3 i używać tych samych seedów.

## Pretrening i UMT

Pretrening pojedynczej modalności korzysta z tego samego runnera. Dostępne tryby to `left_proper`, `right_proper` i `right_blurred`:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  mode=right_blurred model_name=mm_resnet dataset_name=mm_cifar10 \
  lr=0.5 wd=0.0 epochs=300
```

Historyczne moduły `run_pretrain_modality*` pozostają wrapperami kompatybilności.

Zbiorczy opis wszystkich dotychczasowych wariantów Phase-3 intervention,
wyników, kontroli negatywnych i aktualnego oracle coarse-to-fine znajduje się
w `docs/INTERVENTION_EXPERIMENTS.md`.

Publikacyjne referencje dla stoppera relative-unimodal-parity używają
wersjonowanego splitu 44k/5k/1k, sparowanej inicjalizacji oraz wyboru best
validation accuracy:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  config=configs/experiments/cifar10_unimodal_reference.yaml \
  mode=left_proper seed=83
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  config=configs/experiments/cifar10_unimodal_reference.yaml \
  mode=right_proper seed=83
```

Po wskazaniu obu wybranych checkpointów można uruchomić właściwy protokół:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  config=configs/experiments/cifar10_relative_unimodal_parity_p1_40.yaml \
  unimodal_references.left_checkpoint=/path/to/left.pth \
  unimodal_references.right_checkpoint=/path/to/right.pth
```

`phase3_stopping.recovery_fraction_threshold=1.0` odtwarza exact parity.
Niższy globalny próg, np. `0.90` lub `0.95`, zatrzymuje interwencję po
odzyskaniu odpowiedniej części deficytu weak branch względem stanu `e3=0`.
Próg jest potwierdzany w dwóch kolejnych walidacjach; nie wolno dobierać go
osobno dla seedu ani na zbiorze testowym.

Szczegóły metody i fallbacków opisuje
`docs/PHASE3_RELATIVE_UNIMODAL_PARITY.md`.

UMT korzysta z tego samego wspólnego runnera, dołącza zamrożone gałęzie nauczycieli i dopasowuje reprezentacje MSE tylko dla aktualnie aktywnych gałęzi studenta:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_all_at_once_umt \
  model_name=mm_resnet dataset_name=mm_cifar10 \
  lr=0.5 wd=0.0 phase1=100 phase2=0 phase3=0 phase4=0 \
  left_branch_pretrained_path=/path/to/left_checkpoint.pth \
  right_branch_pretrained_path=/path/to/right_checkpoint.pth \
  distill=1.0
```

Ścieżki mogą wskazywać checkpoint całego modelu bimodalnego albo checkpoint samej gałęzi. Architektura i wymiary nauczyciela muszą zgadzać się ze studentem. Logger UMT zapisuje `classification_loss`, surowe `distillation/loss`, ważone `distillation/weighted_loss` oraz całkowite `loss`; ostatnia wartość jest dokładnie skalarem używanym przez `backward()`.

Eksperyment `configs/experiments/cifar10_umt_phase3_p1_120.yaml` jest pełną
alternatywną interwencją: obie modalności studenta są aktywne, oba encodery
studenta oraz shared trunk/classifier są trenowalne, a zamrożone pozostają
wyłącznie dwa sparowane modele-nauczyciele. Historyczny przebieg UMT z
deaktywacją lewej gałęzi jest traktowany jako kontrola `right-only`, nie jako
wynik tej pełnej interwencji.

Jawna ablacją Phase 3 jest `phase3_intervention.mode=frozen_left_active`.
Lewa gałąź nadal uczestniczy wtedy w forward i addytywnej fuzji, ale jej
parametry mają `requires_grad=False`, nie należą do optimizera i moduł działa
w `eval()`, więc nie zmienia buforów BatchNorm. Prawa gałąź oraz shared
trunk/classifier pozostają trenowalne. Jest to inny eksperyment niż historyczna
`deactivation` i nie należy łączyć ich wyników bez oznaczenia wariantu.

Phase 4 może logować sampled-FIM co ustaloną liczbę epok. Klucz
`trace_fim_overall_train/proper_ratio_left_to_right` oznacza
`Tr(F_L)/Tr(F_R)` dla wszystkich penalizowanych parametrów encoderów;
wariant z sufiksem `_weight` obejmuje wyłącznie wagi. Konfiguracja
`cifar10_phase4_from_frozen_left_active_p1_120.yaml` mierzy go co 10 epok na
stałym, klasowo zbalansowanym probe 1k, z `fim_chunk_size=128`.

## Modele i konfiguracje

Rejestr modeli znajduje się w `src/utils/common.py`. Dostępne nazwy:

- `mm_mlp_bn`
- `mm_simple_cnn`
- `mm_resnet`
- `mm_effnetv2s`
- `mm_resnet18`
- `mm_convnext`

Każdy wpis ma odpowiadający plik `src/configs/<model_name>.json`. Łączenie gałęzi jest domyślnie sumowaniem cech. Konkatenacja jest obsługiwana przez własny ResNet, MLP i SimpleCNN; architektury oparte bezpośrednio na torchvision jawnie ją odrzucają.

Historyczny profil 50k obejmuje `overlap=0.0` i `0.125`; jego manifest znajduje się w `docs/normalization/cifar10_train_fields.json`. Nowy profil 44k ma osobny manifest w `configs/data/` i obecnie celowo obsługuje wyłącznie `overlap=0.0`. Fashion-MNIST i Tiny ImageNet zachowują historyczne wartości bez pełnego manifestu. MNIST, KMNIST i SVHN są obecne w rejestrze, ale trening jest celowo blokowany, dopóki ich placeholdery/skopiowane statystyki nie zostaną zastąpione wynikiem skryptu.

Każdy nowy run zapisuje w hiperparametrach dokładne `mean/std` trzech transformacji, a manifest RSV zapisuje statystyki obu wejść. Historycznych i nowych trajektorii nie należy łączyć bez sprawdzenia tych wartości; korekta rozmytego CIFAR-10 zmienia wersję protokołu wejściowego.

Statystyki można policzyć strumieniowo, ale operacja iteruje pełny dataset i nie może działać na login nodzie. Należy wysłać generator manifestu na compute node przez Slurm; domyślnie liczy wyłącznie `overlap=0.0`:

```bash
scripts/bash/submit_experiment.sh \
  scripts.python_new.generate_cifar10_protocol_manifest \
  --dataset-path "$CIFAR10_PATH" --overlap 0.0 --resize-factor 0.25
```

## Wyniki i logowanie

Każdy run zapisuje dane w:

```text
$REPORTS_DIR/<experiment>/<timestamp>/
  checkpoints/model_step_epoch_<N>.pth
  wandb/
```

Timestamp zawiera mikrosekundy, aby równoległe uruchomienia nie użyły tego samego katalogu. Backend loggera jest ładowany dopiero wtedy, gdy zostanie wybrany, więc brak opcjonalnego TensorBoard nie blokuje importu rdzenia repozytorium.

Do W&B trafiają niezaokrąglone wartości punktów pomiarowych. Learning rate jest logowany bezpośrednio po kroku schedulera jako surowe `lr/training` i nie jest uśredniany przez liczbę próbek. Zaokrąglenie do czterech miejsc jest stosowane tylko w pasku postępu, a suwak smoothing w W&B zmienia wyłącznie wizualizację — nie historię runu ani checkpoint.

Historyczne konfiguracje weak-recovery wskazują `bartekk/CLPIntervention_PAIS`.
Nowa seria clean-gold, minimal-exposure, P1=40 observe-only oraz Phase-4 grid
używa osobnego projektu `bartekk/CLPIntervention_Phase3Stopping`. W Phase 3
weak-only validation loss jest dostępny pod `phase3/weak_only_val_loss` oraz
aliasem `phase3/weak_only_loss`. Techniczne smoke testy pozostają offline. Projekt
`bartekk/CLPIntervention_Phase3Stopping` został utworzony jawnie przez W&B API;
runy online trafią do niego bez tworzenia pomocniczego pustego runu.

Eksperymentalna reintegracja P4 może najpierw uczyć wyłącznie wspólny
downstream, pozostawiając oba encodery aktywne w forward, ale zamrożone w
`eval()`. Jest to jawny, domyślnie wyłączony wariant:

```yaml
phase4_staged_unfreezing:
  enabled: true
  shared_only_epochs: 4
```

Po prefiksie pełny model zostaje odblokowany bez resetu optimizera P4.
Metodologia i ograniczenia są opisane w
`docs/PHASE4_STAGED_REINTEGRATION.md`.

Osobna, również domyślnie wyłączona ablacja chroni samodzielną użyteczność
prawej gałęzi przez wspólny cel jednego modelu:

```yaml
phase4_auxiliary_loss:
  enabled: true
  weak_weight: 1.0
  dominant_weight: 0.0
```

Odpowiada to `L_full + L_weak`. Wartość `dominant_weight=0` pomija
dominant-only training forward; ten tryb jest nadal oceniany diagnostycznie na
validation proper. Konfiguracja pierwszej kontroli znajduje się w
`configs/experiments/cifar10_phase4_weak_auxiliary_loss.yaml`.

Seed runu steruje RNG Pythona, NumPy, CPU i wszystkich urządzeń CUDA; pełny checkpoint zachowuje te stany, a FIM, RSV, bootstrap i kontrola BN izolują własne losowanie. DataLoader korzysta z deterministycznego seedowania workerów PyTorch przy niezmienionej wersji biblioteki, liczbie workerów i granicy epoki. Nie jest to gwarancja bitowej zgodności GPU: `torch.use_deterministic_algorithms` i deterministyczne ustawienia cuDNN nie są wymuszane.

Każdy run tworzy atomowo `run_manifest.json` w katalogu wyniku. Manifest zapisuje status (`running`, `completed` lub `failed`), Job ID Slurma, commit i hash całego widocznego drzewa źródeł (również plików nieśledzonych), pełną konfigurację i manifest protokołu, wersje wszystkich pakietów, identyfikator oraz treściowy SHA-256 datasetu, wejściowy checkpoint i hashe wszystkich zapisanych checkpointów. Dzięki temu wynik pozostaje identyfikowalny także przy brudnym worktree i wyłączonym W&B.

Publikacyjny baseline dla seedu 83 jest zamrożony w `configs/experiments/paper_cifar10_sresnet18_baseline_seed83.yaml`. Runner odrzuca naukowe nadpisania CLI; dozwolony jest tylko `resume_checkpoint`, ponieważ nie zmienia protokołu. Wariant fazy 4 z `wd=5e-4` i `lr×0.98` należy zamrozić dopiero po uzyskaniu z baseline liczbowej `phase4_target_train_acc`; użycie wartości testowej byłoby przeciekiem metodologicznym.

Wizualizacja neuronów gałęzi wymaga Captum:

```bash
scripts/bash/submit_experiment.sh weights_visualisation /path/to/checkpoint.pth \
  --model-name mm_resnet --output-dir features
```

## Testy

Testy tworzące batche tensorowe lub iterujące loadery muszą działać na compute nodzie. Można je wysłać przez wspólny launcher (również wtedy, gdy same testy używają CPU):

```bash
scripts/bash/submit_experiment.sh pytest -q -m "not gpu" tests
```

Na login nodzie wolno wykonywać tylko kontrole statyczne oraz jawnie wybrane testy czystej logiki, które nie tworzą batchy tensorowych.

Testy pełnych modeli na GH200:

```bash
sbatch scripts/bash/run_gpu_tests.sh
```

Kontrole dodatkowe:

```bash
scripts/bash/clp_python_local.sh -m compileall -q src scripts/python_new
for file in *.sh scripts/bash/*.sh; do bash -n "$file"; done
git diff --check
```

Testy nie pobierają datasetów. Zestaw GPU wykonuje pełne forwardy wszystkich zarejestrowanych modeli oraz jeden krok optymalizacji, ale nie uruchamia długiego treningu eksperymentalnego. Zakres oraz ograniczenia audytu są opisane w `docs/AUDIT.md`.
