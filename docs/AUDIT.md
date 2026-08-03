# Audyt repozytorium CLPIntervention

Data audytu: 2026-07-27–2026-07-28

## Zakres i metoda

Audyt objął aktywny kod w `src/`, `scripts/python_new/`, główne skrypty uruchomieniowe oraz kontrakty konfiguracji i checkpointów. `scripts/python_old/`, `scripts/python_backup/` i notebooki zostały potraktowane jako archiwalne lub analityczne i nie były refaktoryzowane.

Wykonano analizę statyczną przepływu czterech faz, datasetów i transformacji, wszystkich zarejestrowanych modeli bimodalnych, strat, UMT, metryk asymetrii, loggerów i launcherów. Walidacja dynamiczna obejmowała kompilację, niezależne importy oraz testy i smoke runy wykonywane na compute node. Statystyki, loadery, operacje batchowe i pełne modele nie są uruchamiane na login nodzie.

## Audyt względem artykułu

Przeczytano pełny, 53-stronicowy artykuł `docs/papers/Analysis of neuroscience-inspired_intervention restoring the plasticity of_bimodal artificial neural networks in_the context of critical learning periods.pdf` oraz 17-stronicowy `docs/papers/Critical Learning Periods for Multisensory Integration in Deep Networks v2.pdf` i zmapowano oba warianty metody na aktywny kod.

| Element metody | Stan po audycie |
|---|---|
| S-ResNet-18: dwie gałęzie do `layer2`, sumowanie i wspólne `layer3`–`layer4` | zgodne w historycznie używanym `mm_resnet` |
| Flip `p=0.5`, podział, niezależny shift `1/8`, brak rotacji | CIFAR-10 i Fashion-MNIST dostosowane do opisu |
| Fazy `1→2→3→4`, w fazie 3 wyłączona lewa gałąź | zgodne i współdzielone przez wszystkie standardowe runnery |
| Faza 4 do accuracy bazowego runu; wariant `wd=5e-4`, mnożnik LR `0.98` tylko w fazie 4 | obsługiwane jawnymi opcjami |
| FIM co pół epoki na zbalansowanych, rozłącznych 2% treningu; bez BN i wspólnego trzonu | naprawione i generowane deterministycznie |
| RSV post hoc na checkpointach faz, 5 obrazów/klasę i 100 wariantów/pole | dwa punkty: avgpool po stage 3 oraz avgpool po stage 4; surowe SV/RSV i manifesty |
| EWMA-10 | poprawnie pozostawione jako wizualizacja post hoc; logger zapisuje niezaokrąglone punkty |
| Bootstrap sparowanych różnic / korelacje / bisekcja / sweepy | bootstrap zaimplementowany; pozostałe analizy nadal nieobecne |

Kleinman et al. definiują równaniem `(SV_left-SV_right)/sum` konwencję `+1=lewa`. Repozytorium przyjmuje ją jako jedyną kanoniczną konwencję. Histogramy pracy CLPIntervention opisane odwrotnie wymagają transformacji `RSV_Kleinman=-RSV_historyczne`. Nowy protokół porównuje kanałowe reprezentacje po `AdaptiveAvgPool2d(1)`: analizowy pooling po `main_branch.0` (`layer3`) oraz natywny `avgpool` po `main_branch.1` (`layer4`) zgodny z Kleinmanem.

Metoda doboru learning rate na zbiorze testowym i pojedynczym seedzie jest ryzykiem metodologicznym, nie błędem wykonania kodu. Nowe eksperymenty powinny stroić hiperparametry na walidacji i raportować wiele seedów; bez tego nie należy traktować korelacji z artykułu jako potwierdzonych statystycznie.

## Najważniejsze naprawione problemy

### Krytyczne

1. **Zanieczyszczenie początkowej asymetrii przez BatchNorm.** Automatyczne wyznaczanie wymiarów wykonywało losowy forward lewej gałęzi w trybie treningowym, aktualizując tylko jej statystyki BatchNorm już w konstruktorze. `infer_dims_from_blocks` działa teraz bez gradientów w trybie eval i odtwarza poprzedni tryb.
2. **Błędne kontrakty fuzji i wymiarów modeli.** W modelach opartych na torchvision konkatenacja zachodziła po wymiarze przestrzennym, a szerokość gałęzi była liczona drugi raz. W ResNet występowały błędne padding, liczba kanałów wejściowych i wymiar klasyfikatora. MLP z konkatenacją miał niezgodny wymiar pierwszej warstwy wspólnej. Wszystkie te kontrakty zostały poprawione i objęte testami forward.
3. **Nieuruchamialny pretrening jednej modalności.** Trener wyłączał gałąź bez podania wymaganej interwencji, co kończyło forward wyjątkiem. Wyłączona gałąź otrzymuje teraz `deactivation`.
4. **Nieuruchamialna ścieżka UMT.** Wrapper nie zwracał reprezentacji, używał innych nazw nauczycieli niż trener, nie zamrażał ich, a runner miał błąd składni i brak pola `distill`. Naprawiono pełny przepływ, ekstrakcję gałęzi z checkpointu i destylację wyłącznie aktywnych gałęzi.
5. **Sekret W&B w śledzonym pliku.** Literalny klucz został usunięty z `src/configs/env_variables.sh`. Plik zachowuje teraz wyłącznie wartości ustawione zewnętrznie. Ponieważ sekret pozostaje w historii Git, powiązany klucz należy unieważnić i wygenerować ponownie.
6. **Fałszywe statystyki normalizacji.** Wiele wartości oznaczonych jako `FAKE` lub skopiowanych z CIFAR-10 do innych datasetów zostało usuniętych. Nieobsługiwany `overlap` kończy się jawnym błędem zamiast treningiem z błędną normalizacją.

### Wysokie

7. **Niespójne fabryki datasetów.** `get_tinyimagenet` odwoływał się do niezdefiniowanej ścieżki, część fabryk nie przyjmowała używanego przez runner parametru `subset`, a transformacja poprawnej prawej modalności nie była dostępna poza CIFAR-10. Sygnatury i wiring zostały ujednolicone.
8. **Etykiety SVHN.** Narzędzia oczekiwały zawsze pola `targets`, podczas gdy SVHN używa `labels`. Obsługiwane są teraz oba warianty, `Subset` i zagnieżdżone wrappery.
9. **Graf autograd podczas ewaluacji.** Walidacja budowała graf i mogła uruchamiać kosztowne regularizatory drugiego rzędu. Ewaluacja działa teraz pod `torch.no_grad()`, a model i criterion przechodzą razem do eval.
10. **Błędna metryka gradientów prawej gałęzi.** `GradientsSpectralStiffness` kopiował gradienty lewej gałęzi do prawej, pomijał końcowy niepełny chunk i grupował błędy według predykcji zamiast prawdziwej klasy. Wszystkie trzy błędy zostały poprawione.
11. **Niepełne statystyki wspólnego modelu.** `RunStatsBiModal` pomijał klasyfikator `fc` lub `classifier`, a `acos` mógł produkować NaN przez błąd numeryczny. Grupa wspólna obejmuje teraz wszystkie parametry poza obiema gałęziami, argument kąta jest ograniczany do zakresu, a tryb modelu jest odtwarzany.
12. **Martwe ReLU w fazie 3.** Hook lewej gałęzi nie otrzymywał aktywacji po jej wyłączeniu, co prowadziło do dzielenia przez zero na końcu epoki. Pusty pomiar jest pomijany, a metryka nie traktuje ujemnych wartości GELU i LeakyReLU jako martwych ReLU.
13. **Ładowanie nauczycieli z pełnych checkpointów.** Próba załadowania `left_branch.*` do samej gałęzi powodowała niezgodne klucze. `load_branch` rozpoznaje pełny i gałęziowy `state_dict`.
14. **Błędne i martwe loadery rank.** Transformacja była przypisywana do `DataLoader`, a nie datasetu, i współdzieliła ten sam obiekt między wariantami. Nieużywany blok usunięto z aktywnych runnerów.

### Średnie i porządkowe

15. `BalancePenaltyLoss` przekazuje teraz wagi klas także do podstawowego CE i nie uruchamia regularizatora w eval.
16. `MSESoftmaxLoss` nie zakłada już dziesięciu klas i zwraca kontrakt zgodny z trenerem. `FisherPenaltyLoss` zwraca spójne dwa elementy.
17. Konfiguracja optymalizatora nie mutuje słownika przekazanego przez runner.
18. Opcjonalne backendy loggerów są importowane leniwie, więc brak TensorBoard lub ClearML nie blokuje importu rdzenia.
19. Dodano brakujące konfiguracje `mm_simple_cnn`, `mm_resnet18` i `mm_convnext` dla modeli obecnych w rejestrze.
20. Architektury torchvision obsługują deklarowaną liczbę kanałów wejściowych. Interwencje tworzą tensory o zgodnym device i dtype.
21. `log_multi` nie może już przyjąć zera dla małych loaderów.
22. Katalogi runów otrzymują timestamp z mikrosekundami, co usuwa kolizje równoległych startów.
23. Launchery pretreningu rozmytego i poprawnego wskazywały zamienione moduły. Moduły zostały skorygowane.
24. Dwadzieścia wariantów launcherów dla A100/RTX i starego `clpi_env` zastąpiono jednym parametryzowanym `scripts/bash/run_experiment.sh` dla środowiska GH200.
25. `weights_visualisation.py` nie zawiera już prywatnego checkpointu, duplikatów importów i stałej konfiguracji. Otrzymał jawny interfejs CLI.
26. Dodano `environment.yml`, kompletny `README.md`, właściwy `AGENTS.md` i testy regresyjne.
27. Jedenaście niemal identycznych runnerów faz, pretreningów i UMT scalono w `scripts/python_new/run_single.py`; stare moduły są wyłącznie wrapperami zgodności.
28. Standardowy trener i trener UMT współdzielą teraz przygotowanie eksperymentu, definicje etapów, pętlę treningową i obsługę artefaktów. UMT nadpisuje wyłącznie obliczenie straty destylacyjnej.
29. Audyt środowiska wykazał około 316 pakietów w lokalnym prefiksie. JupyterLab, Qt, pandas, statsmodels, torchaudio, pyarrow, datasets i seaborn nie są wymagane przez aktywny kod; pozostają skutkiem klonowania środowiska bazowego. Odchudzona deklaracja znajduje się w `environment.yml`.
30. Ujednolicono runtime protocol loggerów W&B, TensorBoard i ClearML (`log_model`, skalary, histogramy, wykresy i zamknięcie). W&B zachowuje teraz numer kroku, a opcjonalne backendy nie wymagają niezgodnych argumentów.
31. `TraceFIM` używa tych samych próbek etykiet modelu dla obu gałęzi, izolowanego RNG, buforów modelu i przywraca poprzedni tryb. Usunięto trzy redundantne warianty implementacji.
32. Próba FIM jest deterministycznie generowana z klasowo zbalansowanych 2% treningu, wykluczana z loadera i budowana z poprawnie sparowanych modalności. Metadane zawierają hash indeksów.
33. Faza 4 obsługuje zatrzymanie po referencyjnej accuracy treningowej oraz nadpisania `weight_decay` i mnożnika LR wyłącznie w tej fazie.
34. Naprawiono nieuruchamialny `RSVCallback` (brak `group_size`), przypadek zerowej wariancji, retencję grafu i orientację znaku.
35. Usunięto nieudokumentowaną rotację 5° z treningu CIFAR-10 i Fashion-MNIST; pozostał shift `1/8` zadeklarowany w artykule.
36. `resize_factor` nie jest już ignorowany, gdy trener ponownie ustawia rozmytą transformację fazy 1.
37. Skrypt statystyk normalizacji obsługuje wszystkie bimodalne datasety, działa strumieniowo i nie materializuje całego zbioru w RAM.

38. Usunięto nieużywaną klasę `RunStats` zawierającą celowe `1 / 0`; aktywne statystyki jawnie odrzucają niedokończony wariant cosinusowy zamiast bezgłośnie nie zapisywać wyników.

39. Checkpointy są wersjonowane i zawierają model, optimizer, scheduler, mnożnik `MultiplicativeLR`, następną epokę, krok globalny, RNG oraz stan `RunStatsBiModal`. Wznowienie `all_at_once` pomija zakończone fazy; stare model-only `state_dict` pozostają obsługiwane wyłącznie jako `model_checkpoint`, nie jako `resume_checkpoint`.
40. Dodano chronologiczny dziennik prac i dziennik pojęć, rozdzielone od raportu audytu.
41. Dodano kompletny pipeline RSV z deterministycznym doborem 5 obrazów/klasę, oryginałem + 99 augmentacjami, jawną warstwą i znakiem, surowymi SV/RSV oraz manifestem.
42. Logger zapisuje pełną precyzję punktów; zaokrąglanie dotyczy wyłącznie terminala, a EWMA pozostaje wizualizacją post hoc.
43. Rozdzielono `model_checkpoint` (weights-only transfer fazy) od `resume_checkpoint` (pełne wznowienie).
44. Przeliczono CIFAR-10 dla obu overlapów; błędne odchylenia rozmytej prawej modalności poprawiono i zapisano manifest datasetu. MNIST/KMNIST/SVHN failują jawnie do czasu wyznaczenia statystyk.
45. RSV jest liczony post hoc dla wielu checkpointów faz w dwóch punktach z poolingiem kanałowym podczas tych samych forwardów.
46. Dodano klasowo stratyfikowany, hierarchiczny bootstrap sparowanych różnic po modelach/seedach i obrazach, z walidacją zgodności protokołu oraz indeksów.
47. Dodano opt-in rekalibrację buforów BatchNorm wspólnego trzonu przed fazą 4, bez gradientów, zmian wag i zużycia RNG treningu.
48. `RunStatsBiModal` loguje odległość całego modelu oraz lewej, prawej i wspólnej gałęzi od początku każdej fazy; referencje i akumulatory są zachowywane przez pełny resume. Legacy model-only `resume_checkpoint` jest odrzucany, aby nie powtarzać faz od epoki zero.
49. UMT loguje teraz całkowitą optymalizowaną stratę pod `loss`, a osobno stratę klasyfikacji oraz surową i ważoną destylację.
50. Fazy 3 i 4 jawnie ustawiają prawą modalność na transformację `proper`, więc uruchomienie fazy nie zależy od stanu transformacji odziedziczonego po wcześniejszym kodzie.
51. Checkpoint v4 zawiera niezależny od ścieżek hosta manifest protokołu. Pełny `resume` waliduje rekurencyjnie architekturę, dane i normalizację, subset/FIM, seed, batch size, granice faz, optimizer, scheduler, UMT i kontrolę BN; niezgodność kończy się przed treningiem.
52. Learning rate nie trafia już do agregatora ważonego liczbą próbek; `lr/training` jest logowany bezpośrednio po kroku schedulera.
53. Manifest protokołu v2 zastępuje harmonogram FIM oparty na globalnym modulo jawnym `fim_measurements_per_epoch`; domyślnie wybiera dokładnie dwa lokalne batche każdej epoki, także dla nieparzystych loaderów.
54. `RunStatsBiModal` rejestruje po każdym `optimizer.step()` rzeczywiste przesunięcie każdego parametru i sumuje normy kroków lewej, prawej i wspólnej gałęzi oraz modelu. Metryka obejmuje weight decay, momentum i różne grupy optimizera; snapshot ostatniego kroku jest przy resume rekonstruowany z checkpointowanego modelu na granicy kroku.
55. Każdy aktywny run zapisuje atomowy `run_manifest.json` z hashem kodu, danych, pełną konfiguracją, środowiskiem, Job ID, checkpointem wejściowym i hashami artefaktów. Bazowy protokół CIFAR-10/S-ResNet-18 dla seedu 83 jest zamrożony, a runner odrzuca naukowe nadpisania CLI.
56. Sampled FIM został porównany z naiwną pętlą per-example opartą na `autograd.grad`; aktywna implementacja `vmap(grad)` daje ten sam ślad, izoluje RNG sampled labels i używa wspólnych labeli dla obu gałęzi. Dodano TFIM znormalizowany liczbą wszystkich dopuszczonych parametrów każdej gałęzi oraz osobny wariant znormalizowany liczbą wag; BatchNorm pozostaje wyłączony zgodnie z protokołem.

57. Dodano wersjonowany profil `cifar10_stratified_44k_5k_1k_seed83_v1`: dokładnie 44k train, 5k validation i 1k FIM, stratyfikowane per klasa stałym seedem splitu 83. Hash targetów, indeksów i siedmiu surowych plików CIFAR-10 jest walidowany fail-closed.
58. Statystyki `proper_left`, `proper_right` i `blurred_right` policzono na compute nodzie wyłącznie z finalnego train 44k, przed losową augmentacją. Profil 44k obsługuje overlap 0.0; historyczny profil 50k pozostaje niezmieniony.
59. Rozdzielono `CLPTrainingLoaders` od leniwie tworzonych `CLPTestLoaders`. Train ma jawny generator i seedowanie workerów; validation/FIM/test są deterministyczne, bez shuffle, drop-last, flipu i losowej augmentacji. FIM używa stałego 1k.
60. Dodano nieinwazyjną ewaluację `full`, `dominant_only`, `weak_only` i `intervention_mode`, wielosygnałowe plateau Phase 2 oraz walidacyjny stopper Phase 3 z compatibility-drift constraints, hard safety, rankingiem feasible/safe i rollbackiem.
61. Checkpoint v5 zapisuje stan kontrolerów, lokalną epokę fazy i generator loadera. W profilu walidacyjnym optimizer/scheduler są świeże na granicy faz, zachowywane tylko do resume wewnątrz fazy; `global_step` nie cofa się po wyborze wcześniejszego checkpointu.
62. Phase 4 ma dwa selektory validation proper: pełny budżet oraz `200-e3`, z kandydatem epoch 0. Test proper/blurred jest tworzony i używany dopiero po zakończeniu Phase 4.
63. Sampled FIM na stałym probe 1k jest liczony w deterministycznych chunkach. Agregacja sumuje per-example trace i dzieli przez pełne `M*N`, zachowując definicję estymatora przy ograniczonym szczycie pamięci.

64. Dodano PAIS: sparowane per-image przedziały z korektą liczby spojrzeń i rodzin metryk, lokalne slope, trend reversal, futility z krótkim optymistycznym horyzontem oraz CI-aware hard safety.
65. Dodano stały, klasowo zbalansowany weak-only train probe 1k z train 44k. Jest oceniany bez augmentacji w `eval()` co 10 epok i loguje generalization gap wyłącznie diagnostycznie.
66. Kadencję ograniczono do validation proper co 5 epok, blurred full-only, FIM i resume co 10 epok. Równoważny `intervention_mode` współdzieli forward, a kandydaci Phase 3/4 są zapisywani tylko, gdy stają się aktywnym best.

67. Kalibracja PAIS na seedach 83/184/285 wykazała dominację hard safety: hipotetyczne stopy nastąpiły w epokach 30/15/15, bez checkpointu safe lub feasible, mimo wzrostu weak-only accuracy o 52–64 p.p. Marginesy 0.05/0.20 mylą oczekiwany compatibility drift z katastrofią i nie mogą zostać zamrożone. Shadow mode zamraża pozostałe liczniki po pierwszym triggerze, więc przed następną kalibracją należy logować niezależne czasy hard safety, futility i reversal.
68. Dodano jawnie wersjonowany `decision_rule: weak_recovery`: sparowane correctness i loss weak-only, recovery plateau/reversal, numerical-only emergency stop, diagnostyczny compatibility drift, niezależne shadow timestamps oraz pięć milestone checkpointów wyłącznie w kalibracji. Legacy PAIS pozostaje dostępny do reprodukcji.
69. W&B Phase 3 loguje teraz namespaced epoch oraz oba klucze `weak_only_val_loss` i `weak_only_loss`. Nowe konfiguracje wskazują jawnie projekt `bartekk/CLPIntervention_PAIS`, eliminując zależność od odziedziczonych zmiennych środowiskowych.
70. Shadow smoke potwierdził zamrożenie pierwszej hipotetycznej decyzji przy dalszym śledzeniu niezależnych triggerów i milestone checkpointów. Końcowy pruning został poprawiony tak, aby oprócz checkpointu wybranego przez zamrożoną decyzję zachowywał także późniejsze aktywne best feasible/safe potrzebne do analizy kalibracyjnej.
71. Kalibracja weak-recovery seedów 83/184/285 (`20121024`–`20121026`) doszła do max 200 epok bez plateau ani reversal i wybrała epoki 200/185/195. Weak-only validation loss osiągał minimum już w epokach 10–15, potem rósł przy train-probe loss bliskim zeru. Obecny ranking accuracy-first oraz progi plateau są zbyt konserwatywne; przed `enforce` wymagają accuracy-noninferiority, loss-aware selection i walidacji na wynikach Phase 4 z checkpointów milestone. Szczegóły: `docs/WEAK_RECOVERY_CALIBRATION_2026-07-29.md`.

72. Sweep Phase 3/4 na seedach 83/184/285 (`20135234`, `20135235`, `20135237`–`20135255`) wskazał stabilne optimum `e3≈60` dla `P1=80, P2=200`: 89,15% mean best validation accuracy wobec 87,62% bez interwencji i 89,57% dla kontroli minimalnej ekspozycji `P1=1, P2=200`. Ta ostatnia nie jest clean gold standardem. Między `e3=60` i `e3=80` końcowa dominant-only accuracy spadła 79,59%→33,66%, mimo wzrostu weak-only 55,15%→80,10%. Ujawniło to błąd metodologiczny weak-recovery oraz rozbieżność selekcji loss/accuracy. Raport i wykresy: `docs/PHASE3_PHASE4_SWEEP_2026-07-29.md`.

73. Naprawiono kontrakt weak-recovery: `feasible` wymaga teraz jednocześnie recovery i compatibility safety. Phase 2 oraz oba budżety Phase 4 zachowują oddzielnie minimum validation loss i maksimum validation accuracy, mają wersjonowalny wybór `primary_metric` oraz zgodny wstecznie stan resume. Ewaluacja modalności loguje dodatkowo niesmoothowany NLL, multiclass Brier score, 15-bin ECE, średnią confidence i confidence błędnych predykcji dla `full`, `dominant_only`, `weak_only` i `intervention`. Konfiguracje gold-standard i Phase-4 sweep wskazują jawnie accuracy jako metrykę główną; minimum loss pozostaje zachowanym wynikiem diagnostycznym.
74. Skorygowano błędne nazewnictwo kontroli: aktywny trener wykonuje dokładnie jedną epokę dla `phase1=1`. Konfigurację przeniesiono do `cifar10_minimal_blurred_exposure_p1_1_p2_200.yaml`, a prawdziwy clean gold standard otrzymał `phase1=0, phase2=200` w `cifar10_clean_gold_standard_p1_0_p2_200.yaml`.
75. Dodano `phase2_test_policy=posthoc_final`. Po zakończeniu wszystkich faz procedura odtwarza zachowane minimum-loss i maksimum-accuracy Phase 2, oblicza test proper/blurred, loguje wyłącznie `posthoc_test/phase2/*` i zawsze przywraca końcowy checkpoint. Polityka oraz rekordy checkpointów są zachowane w manifeście i resume; test nie jest dostępny stopperom ani selektorom.
76. Nową serię eksperymentów oddzielono od historycznego projektu PAIS. Utworzono i zweryfikowano projekt `bartekk/CLPIntervention_Phase3Stopping`; aktywne konfiguracje clean gold, minimal exposure, P1=40 observe-only, techniczny smoke i Phase-4 grid wskazują ten projekt.
77. Dodano jawne przejście `observe_phase4_transition=hypothetical_selected`. Pełny shadow trace Phase 3 nadal dochodzi do maksymalnego budżetu, następnie przed Phase 4 odtwarzany jest zamrożony checkpoint rekomendowany przez stopper. Profil P1=40 używa P4=200 i zachowuje milestone `e3=40,60,80,200`, co pozwala porównać rekomendację stoppera z późniejszym fixed-e3 recovery bez mylenia jej z endpointem Phase 3.
78. Trzy ślady P1=40 (`20153353`–`20153355`) nie znalazły żadnego
    checkpointu safe/feasible, więc wszystkie P4 odtworzyły pre-Phase-3
    (`e3=0`). Weak-only loss osiągnął minimum w epokach 15/10/20 mimo dalszego
    wzrostu accuracy. Weak-recovery v3 usuwa loss z bramki plateau/reversal,
    traktuje compatibility drift jako diagnostykę/tie-breaker i dodaje
    checkpoint `e3=20`.
79. Dodano opcjonalny, checkpointowalny liniowy warm-up LR Phase 3. Profil
    P1=40 używa 4 epok od 10% bazowego LR, zwiększanego liniowo po każdym
    optimizer step; historyczne konfiguracje zachowują domyślne zero.
    Zamrażanie wspólnego trzonu pozostaje osobną ablacją, a nie częścią
    podstawowego protokołu.
80. Finalny compute gate schedulera `20257602` oraz czterofazowy smoke
    `20257606` zakończyły się `COMPLETED (0:0)`. Smoke potwierdził trajektorię
    warm-up, wybór accuracy-first `best_feasible` w `e3=3`, odtworzenie tego
    checkpointu w observe replay i start Phase 4 bez rollbacku.
81. Po zmianie protokołu anulowano trzy pełne runy uruchomione ze starą,
    epokową wersją warm-upu. Phase 3 wykonuje wyłącznie `step_batch()` podczas
    warm-upu, potem utrzymuje stałe bazowe LR; nie wykonuje `step_epoch()`.
    Scheduler zapisuje liczbę ukończonych kroków i waliduje zgodność liczby
    kroków oraz start factor przy resume.
82. Pierwsze trzy pełne wznowienia per-step przeszły P1/P2, lecz zatrzymały
    się na pierwszej epoce P3, ponieważ wspólna ścieżka logowania oczekiwała
    schedulerowego API `get_last_lr()`. Dodano ten bezstanowy interfejs do
    schedulera per-step; nie przywraca on schedulerowego kroku epokowego.
83. Naprawiono odtwarzanie schedulerów zależnych od fazy. Przy pełnym resume
    entrypoint najpierw odczytuje metadane checkpointu bez stanu optimizera,
    tworzy świeży optimizer i scheduler przez `optimizer_factory(phase)`, a
    dopiero potem ładuje ich stan. Checkpoint P3 nie jest już omyłkowo
    odtwarzany do domyślnego schedulera epokowego.
84. Po wynikach P1=40 połączono weak recovery z accuracy-based compatibility.
    `feasible` wymaga sparowanej poprawy weak accuracy oraz non-inferiority
    full i dominant-only accuracy. Ranking uwzględnia wszystkie trzy accuracy,
    a `compatibility_breach` reaguje na potwierdzone przekroczenie twardych
    marginesów. Loss pozostaje diagnostyką.
85. Uruchomiono jednorazowy sweep P4=200 dla P1=40:
    `e3={20,40,60,80,200}` × seedy 83/184/285. Selekcja używa wyłącznie
    validation proper; test proper/blurred jest liczony post hoc.
86. Bezwzględna accuracy non-inferiority została wycofana z aktywnego
    profilu po replayu, który dla wszystkich trzech seedów błędnie kończył
    interwencję w e3=10. Dodano `local_accuracy`: sparowany target
    weak≥dominant, odwrócenie Pareto `(weak, full, dominant)` oraz
    `futility_with_harm`. Gradient conflict na deterministycznym
    `validation_proper` probe może wyłącznie potwierdzać futility.
87. Punktowy replay lokalnej reguły na śladach P1=40 wskazał e3=40 dla
    seedu 83, stop e3=60 z wyborem e3=50 dla seedu 184 oraz e3=30 dla
    seedu 285. To przedziały kandydackie; historyczne W&B nie zawiera
    per-example correctness ani gradient probe dla dokładnego replayu CI.
88. Dokładny shadow seedu 83 (`20273663`, W&B `pr4byikq`) użył sparowanych
    per-example CI i wskazał `target_reached` w e3=30. Weak-only accuracy
    wynosiła wtedy 0,7934, dominant-only 0,6868, a full 0,6574. Sporadyczny
    konflikt gradientów nie uruchomił stopu, ponieważ weak accuracy nie była
    futile; potwierdza to, że gradient jest tylko bramką pomocniczą.
89. Sweep 15 runów P4=200 dla e3={20,40,60,80,200} i trzech seedów
    zakończył się poprawnie. Średnia validation full accuracy była najwyższa
    dla e3=40 (0,8912). Najlepsze milestone per seed to 60/40/40 dla
    seedów 83/184/285; replayowe wybory 40/50/30 leżą w sąsiedztwie tych
    optimów. Przy e3=60 dominant-only załamało się już dla seedów 184 i 285.
90. Gradient probe działa co 10 epok, lecz jego ostatni wynik jest zachowany
    na pośredniej walidacji co 5 epok. Regresja potwierdza, że rzadszy probe
    nie zeruje licznika futility-with-harm; konflikt nadal nie wystarcza bez
    równoczesnej futility weak accuracy.
91. Dokładne paired-CI shadow dla seedów 184 i 285 (`20281023`, `20281024`)
    zakończyły się poprawnie i bez zmiany progów wybrały odpowiednio e3=35
    i e3=30. Razem z seedem 83 daje to wybory 30/35/30. Seed 285 wykrył
    późniejsze `pareto_reversal` w e3=65; futility-with-harm nie uruchomiło
    się w żadnym z trzech runów.
92. P4=200 z dokładnych wyborów e3=30/35/30 osiągnęły średnio validation
    full 0,8865 i test proper 0,8766. Względem fixed-e3=40 oznacza to
    -0,47 pp validation full, -0,30 pp test proper, +0,42 pp dominant-only
    i -7,69 pp weak-only. Pierwszy obserwowany paired crossover
    weak≥dominant wystąpił już w e3=5 dla wszystkich seedów, więc sam warunek
    crossoveru nie wyznacza optimum. Uruchomiono kontrolny P4 seedu 184 z
    pierwszego targetu e3=30 (`20282557`).
## Otwarte ograniczenia i decyzje badawcze

1. **Pilot seed 83 zakończony.** Slurm `20051568` ukończył 80/200/96/200 w 6:21:12. Stary shadow stopper wskazał `hard_safety` i rollback pre-Phase-3, więc jego absolutnych progów nie wolno przenosić do enforce. Final proper accuracy obu selektorów Phase 4 wyniosła 0,8588; wynik służy diagnostyce implementacji, nie publikacyjnej ocenie PAIS.
2. **Pochodzenie statystyk normalizacji poza CIFAR-10.** CIFAR-10 ma manifest; Fashion-MNIST i Tiny ImageNet wymagają analogicznego przeliczenia, a MNIST/KMNIST/SVHN są blokowane do tego czasu.
3. **Warstwa analizy publikacyjnej.** Hierarchiczne przedziały bootstrapowe dla sparowanych różnic RSV są obsługiwane. Współczynniki korelacji, automatyczna bisekcja długości interwencji i rekonstrukcja figur pozostają poza aktywnym kodem. EWMA ma pozostać wyłącznie opcjonalną wizualizacją.
4. **Ryzyko metodologiczne historycznych wyników.** Learning rate dobierano na teście i z jednym seedem. Nowy split usuwa test z decyzji online, lecz właściwe wyniki nadal wymagają wielu sparowanych seedów i z góry zamrożonego protokołu analizy.
5. **Granularność wznowienia.** Checkpoint obejmuje pełny stan potrzebny między epokami, w tym globalne i fazowe referencje `RunStatsBiModal`, ale nie zapisuje pozycji wewnątrz batcha i po wznowieniu tworzy nowy katalog loggera. Dokładna kontynuacja wymaga niezmienionego datasetu, subsetu, batch size, konfiguracji i granic faz.
6. **Eksperymentalny wariant `balance_loss`.** Główny `run_all_at_once.py` pozostaje osobnym runnerem z własną konfiguracją, ponieważ zmiana znaku lub normalizacji kary wymaga decyzji badawczej. Nie jest potrzebny do odtworzenia głównego badania.
7. **Kod archiwalny i notebooki.** Mogą zawierać stare ścieżki i wcześniejsze API. Nie są częścią wspieranej ścieżki i nie zostały objęte naprawami.
8. **Eksperymentalna postać balance penalty.** Składnik `reg_part - sum_of_fims` pozostawiono bez zmiany, ponieważ jego znak i normalizacja są elementem hipotezy badawczej. Przed użyciem w publikacji należy potwierdzić zamierzoną definicję matematyczną.
9. **Brak gwarancji bitowej deterministyczności GPU.** Python, NumPy, Torch i CUDA są seedowane, stany RNG są checkpointowane, a diagnostyki izolują losowanie. Repozytorium nie wymusza jednak `torch.use_deterministic_algorithms` ani deterministycznych ustawień cuDNN; zgodność seedów oznacza odtwarzalny protokół, nie gwarantowane identyczne bity między GPU, wersjami PyTorch i architekturami.
10. **Legacy marginesy non-inferiority PAIS wymagają decyzji a priori.** Są nadal logowaną diagnostyką i kontraktem trybu legacy, ale rekomendowany weak-recovery nie przerywa treningu na podstawie oczekiwanego compatibility drift. Przed publikacją trzeba zamrozić samą regułę recovery i sprawdzić jej zgodność z checkpointami milestone na seedach metodologicznych, bez strojenia osobno dla każdego P1.
11. **Margines accuracy-noninferiority pozostaje decyzją a priori.** Implementacja zachowuje teraz zarówno maksimum validation accuracy, jak i minimum niesmoothowanego validation loss/NLL, ale nie wybiera automatycznie marginesu równoważności. Przed wynikami publikacyjnymi trzeba zamrozić jeden globalny margines na seedach metodologicznych; nie wolno stroić go osobno dla P1 ani na teście.

12. **Weak-recovery feasibility nie gwarantuje compatibility safety.** Obecny recovery candidate spełnia dolne granice poprawy weak-only, ale może jednocześnie naruszać full/dominant constraints. Ponieważ wybór preferuje `best_feasible`, konfiguracji weak-recovery nie wolno używać w `enforce` przed połączeniem recovery i safety oraz dodaniem testu regresyjnego unsafe-feasible.
13. **Local-accuracy v4 pozostaje eksperymentalny.** Trzy dokładne shadow
    runy z CI wybrały e3=30/35/30, a fixed-e3 sweep potwierdza właściwy
    obszar. Bezpośrednie P4=200 z tych checkpointów dały średnio 0,8865
    validation full i 0,8766 test proper, ale reguła pozostaje nieco bardziej
    konserwatywna od wspólnego e3=40. Następne runy mierzą e3=1,2,3,4, potem
    co cztery epoki i używają czteropomiarowego okna; przed publikacyjnym
    `enforce` ta zmiana wymaga compute smoke. Test nie może służyć do korekty
    reguły.
14. **Relative-unimodal-parity jest nowym wariantem metodologicznym.** Używa
    seed-paired clean unimodal references i podczas Phase 3 zamraża left oraz
    shared trunk. Usuwa zależność celu od bezwzględnie różnych możliwości
    modalności, lecz osiągnięcie parity nie gwarantuje najlepszego downstream
    full accuracy. Pełne trajektorie wykazały, że exact parity wybiera zbyt
    późne e3=188/112/156. Dodano recovery fraction z zachowaniem
    `threshold=1.0` jako kompatybilnego exact parity. Replay 90% wybrał
    e3=36/24/36, a replay 95% e3=56/36/76. Sześć zależnych P4=200 porównuje
    te dwa globalne progi wyłącznie przez validation proper; test jest
    wyłączony i nie może zmieniać reguły.
15. **Phase 4 natychmiast zmienia kompatybilność gałęzi.** Aktualny
    frozen-shared sweep P1=40 ma mean full accuracy 0,8833/0,8870/0,8898/
    0,8921 dla e3=20/40/60/80 i nie odtwarza historycznego załamania
    dominant-only przy e3>=60. Hybrydowa diagnostyka e3=40 wykazała jednak,
    że w pierwszej epoce P4 zwykłe weak-only spada 0,7720→0,3905. Układ
    `current_right+anchor_shared` zachowuje 0,7243, a
    `anchor_right+current_shared` tylko 0,4814, więc głównym źródłem jest
    dryf shared trunk/classifier. Czteroepokowy warm-up poprawia e1 weak o
    12,38 pp, ale w e10 tylko o 1,25 pp i nie rozwiązuje docelowej
    współadaptacji. Szczegóły:
    `docs/PHASE4_COMPATIBILITY_DIAGNOSTIC_2026-08-02.md`.
16. **Clean gold nie uzasadnia stałego oznaczenia prawej gałęzi jako
    słabszej.** Dla `P1=0/P2=200` wybrane checkpointy mają mean validation
    dominant-only `0,5648`, a weak-only `0,7487`. Asymetria obserwowana po
    `P1=40` jest więc skutkiem protokołu ekspozycji/uczenia, nie stałą różnicą
    informacyjności tych pól. Dodano opt-in prefix P4 uczący najpierw wyłącznie
    shared downstream przy dwóch zamrożonych encoderach oraz diagnostykę jego
    stabilności. Nie jest on jeszcze nowym protokołem publikacyjnym; wymaga
    porównania na tych samych seedach i checkpointach.
17. **Sama kolejność odmrażania nie gwarantuje kompatybilności.** Poprawne
    runy `20355956`–`20355958` zamroziły oba encodery przez e4=1–4; hybryda
    `current_right+anchor_shared` pozostała dokładnie na mean `0,7720`, ale
    zwykłe weak-only spadło do `0,4626`. Zmiana pochodzi zatem wyłącznie ze
    shared downstream uczonego przez `L_full`. Prefix poprawia weak-only wobec
    warm-up control o 4,47 pp w e4=4 i 2,80 pp w e4=10, lecz nie rozwiązuje
    problemu. Wymagany jest osobny eksperyment z celem chroniącym tryby
    unimodalne lub naprzemiennym modality dropout. Pierwsze trzy joby
    `20355917`–`20355919` są oznaczone nieważne, ponieważ nowa sekcja nie była
    przekazywana do `run_config`; regresja kontraktu została dodana.
18. **Repeated-look CI exact recovery jest zbyt konserwatywny jako praktyczny
    stopper.** Sparowany replay 99%/100% na 5k `validation_proper` koryguje 54
    spojrzenia i dwie rodziny progów. Żaden seed nie uzyskał dwóch dolnych
    granic powyżej zera; oba progi użyły fallbacków e3=188/176/156. Przedziały
    nie obejmują dodatkowo niepewności mianowników unimodalnych, ponieważ
    referencje nie zachowały per-image poprawności.
19. **Asymetryczny Phase-4 loss jest ablacją, nie zmianą baseline.** Sekcja
    `phase4_auxiliary_loss` jest domyślnie wyłączona. Pierwsza kontrola używa
    `L_full + L_weak`, `lambda_L=0`; dominant-only nie jest więc osobnym
    sygnałem treningowym. Wpływ drugiego forwardu na full/weak accuracy i czas
    wymaga compute gate oraz trzyseedowej diagnostyki przed P4=200.
20. **Skorygowany repeated-look CI nie jest obecnie praktycznym stopperem.**
    Fallbacki `e3=188/176/156` dały mean full `0,8973` po P4, mniej niż fixed
    `e3=140` (`0,8984`). Korekta 54 spojrzeń i dwóch progów nie uzyskała dwóch
    kolejnych dodatnich LCB w żadnym seedzie. Nie wolno usuwać korekty po
    obejrzeniu wyników; należy z góry ograniczyć liczbę spojrzeń albo użyć
    predefiniowanego alpha-spending/confidence sequence.
21. **Equal-weight UME przewyższa clean gold w trzechseedowym post hoc.**
    Mean validation/test accuracy probability-mean wynosi `0,9077/0,9001`,
    wobec clean gold `0,8979/0,8893`. Logit mean daje `0,9166/0,9073`, ale
    jest analizą czułości, a nie wynikiem wybranym po walidacji. Trzy seedy nie
    wystarczają do końcowego wnioskowania publikacyjnego; definicję ensemble
    i sposób raportowania trzeba zamrozić przed rozszerzeniem seedów.
22. **UMT prawej gałęzi nie chroni zgodności shared trunk z dominant
    encoderem.** Dla P1=120 mean validation weak-only rośnie
    `0,2251→0,8432`, lecz dominant-only spada `0,8183→0,3655`, mimo malejącej
    straty treningowej. Nauczyciel ogranicza dryf reprezentacji prawego
    encodera, ale classification loss nadal dostraja shared downstream do
    jednej aktywnej modalności. Dodano osobną ablacją `frozen_left_active`,
    która zachowuje stałą lewą reprezentację w fuzji; nie jest ona zmianą
    historycznego protokołu.
23. **Optimum długości P3 nie jest monotoniczne.** Shared-trunk P1=120 daje
    mean full validation po P4=200 równe `0,8755/0,8832/0,8865/0,8847/0,8771`
    dla e3=0/20/40/60/80. Klasyczna bisekcja nie jest uzasadniona dla samej
    accuracy; można użyć z góry ustalonego coarse-to-fine search albo
    bisekcji wyłącznie dla monotonicznego crossing criterion, np. zamrożonego
    progu TFIM ratio, a następnie niezależnie sprawdzić recovery w P4.
    Przyjęty oracle diagnostyczny działa per seed: coarse grid co 20 epok,
    rozszerzenie siatki, gdy maksimum leży na brzegu, następnie refinement co
    10 i 5 epok wewnątrz sąsiedniego przedziału. Każdy kandydat przechodzi
    pełne P4=200, a wybór używa wyłącznie validation proper. To wyznacza cel
    dla przyszłego stoppera online, ale samo nie jest tanią regułą zatrzymania.

24. **Standalone Phase 4 omijała TFIM mimo dodatniej częstotliwości.** Profil
    `MODE_SPECS["phase4"]` miał `trace_fim=False`, więc trzy joby
    `20361560`–`20361562` są poprawnymi powtórzeniami recovery, ale nie
    pomiarami Fishera. Profil naprawiono, ustawiono chunk 256 i dodano lokalny
    wersjonowany artefakt `trace_fim_train.jsonl`.

## Zalecenia metodologiczne po porównaniu obu artykułów

1. Traktować seed/model jako jednostkę replikacji. Zaimplementowany hierarchiczny bootstrap agreguje jednostki wewnątrz obrazu, resampluje klasowo obrazy i modele oraz porównuje sparowane różnice `intervention-control`. Do publikacji nadal wymagane jest co najmniej 5 sparowanych seedów.
2. Strojenie LR, weight decay, długości interwencji i progu zatrzymania wykonywać na walidacji. Zbiór testowy powinien być użyty raz dla z góry wybranego protokołu. Bisekcja mediany RSV również jest procedurą strojenia i nie może wykorzystywać końcowego testu.
3. Raportować oddzielnie `stage3_avgpool` (`main_branch.0` + analizowy pooling) oraz `stage4_avgpool` (natywny pooling po `main_branch.1`, wariant Kleinmana). Nie łączyć rozkładów między punktami.
4. Utrzymać domenową normalizację: faza 1 używa statystyk rozmytych danych treningowych, a późniejsze fazy nowych statystyk poprawnej domeny. Wariant wspólnych statystyk może być wyłącznie analizą wrażliwości.
5. Porównywać `native_bn` z opt-in `recalibrated_shared_bn` od tego samego checkpointu fazy 3. Rekalibracja wspólnego trzonu pozwala ocenić, ile efektu pochodzi z buforów BN, a ile ze zmian wag.
6. Ciągłość/reset optimizera i schedulera musi być jednym czynnikiem dla całego porównania. Przy obecnym SGD bez momentum, `wd=0` i mnożniku LR `1.0` różnica jest znikoma; po dodaniu momentum lub decay staje się istotnym confoundem.
7. Wariant fazy 4 z `wd=5e-4` i mnożnikiem LR `0.98` wymaga układu 2×2: interwencja tak/nie × zmiana optimizera tak/nie. Inaczej poprawy nie można przypisać samej interwencji.
8. Wszystkie wykresy powinny bazować na zachowanych surowych punktach. Parametr wygładzania, jeśli użyty do figury, należy podać w podpisie i nie używać wygładzonego przebiegu do testów, korelacji ani wyboru checkpointu.

## Walidacja

Najświeższe bramki nowego protokołu wykonano wyłącznie na compute node:

```text
Slurm 20077966: 35 passed — testy danych, loaderów, modalności i PAIS
Slurm 20078110: 112 passed, 4 ostrzeżenia torchvision/NumPy — pełny compute gate
Slurm 20078023: adaptacyjny czterofazowy enforce smoke COMPLETED w 2:13
Slurm 20078105: pełny FIM probe 1k, M=5, chunk=128 COMPLETED w 1:40
Slurm 20078173: resume wewnątrz Phase 3 i dokończenie Phase 4 COMPLETED w 1:47
Slurm 20119506: 40 passed — końcowy gate weak-recovery i ewaluacji modalności
Slurm 20119503: czterofazowy weak-recovery enforce smoke COMPLETED w 1:48
Slurm 20119808: observe-only shadow smoke COMPLETED w 2:00; decyzja epoka 3, milestones 2/4, dalsze śledzenie do epoki 6
Slurm 20120980: 40 passed po końcowej poprawce retencji checkpointów
Slurm 20120983: finalny observe-only shadow smoke COMPLETED w 2:41; selected 1, milestones 2/4, aktywny best 6 zachowane równocześnie
Slurm 20273546: gradient probe non-invasiveness, 1 passed w 5,90 s, COMPLETED
Slurm 20273600: czterofazowy local-accuracy shadow smoke COMPLETED w 2:29
Slurm 20273663: dokładny Phase-3 shadow z paired CI COMPLETED w 11:56; wybrano e3=30
Slurm 20273137–20273153: 15 fixed-e3 Phase-4 runów COMPLETED
Slurm 20281023: dokładny Phase-3 shadow seed 184 COMPLETED w 12:01; wybrano e3=35
Slurm 20281024: dokładny Phase-3 shadow seed 285 COMPLETED w 12:34; wybrano e3=30
Slurm 20282321/20282322/20282324: P4=200 z wyborów e3=30/35/30 COMPLETED
Slurm 20282574: regeneracja Phase 3 seed 184 e3=30 COMPLETED w 5:46
Slurm 20285723: 2 passed — parity freeze oraz kanoniczna inicjalizacja na GPU
Slurm 20285747: czterofazowy relative-parity smoke COMPLETED w 2:24
Slurm 20285790–20285795: pełne referencje left/right, 3 seedy, wszystkie COMPLETED
Slurm 20290381: RunStats z frozen left/shared, 1 passed na GH200
Slurm 20336049–20336060: 12 frozen-shared milestone P4=200, wszystkie COMPLETED
Slurm 20354203: 4 passed — hybrydowa diagnostyka i per-step warm-up P4
Slurm 20354218–20354223: sześć sparowanych diagnostyk e3=40, wszystkie COMPLETED
Slurm 20355913: 3 passed — staged Phase-4 shared-only trainability
Slurm 20355956–20355958: poprawne staged shared-only diagnostics, COMPLETED
Slurm 20357242–20357247: P4=200 dla frozen-shared e3=140/160, COMPLETED
Slurm 20357275–20357277: P4=200 z repeated-look CI fallbacków, COMPLETED
Slurm 20357331/20357332/20357334: e4<=10 `L_full+L_weak`, COMPLETED
Slurm 20358622/20359333/20359334: post-hoc UME validation/test, COMPLETED
```

Smoke `20078023` wykonał wszystkie fazy w trybie `enforce`, ewaluację modalności, sparowane CI, train probe, rolling selection Phase 2, wybór najlepszego feasible checkpointu Phase 3 oraz oba selektory Phase 4. Zachował osiem potrzebnych checkpointów: granice faz, pre-Phase-3, aktywne best oraz `resume_current`; nie zapisywał każdego pomiaru walidacyjnego. Run resume `20078173` pominął zakończone fazy 1–2, odtworzył stan kontrolera i RNG Phase 3, zachował wybór źródłowej epoki 5 i ukończył Phase 4.

Nie wykonano:

- pełnego odtworzenia historycznego eksperymentu;
- pełnego pomiaru RSV na checkpointach publikacyjnych i rekonstrukcji figur.

Ślady weak-recovery seedów 83/184/285 (`20121024`–`20121026`) zakończyły się poprawnie i wykazały brak stopu przed 200 epoką oraz późną selekcję w obszarze rosnącego validation loss. Analiza znajduje się w `docs/WEAK_RECOVERY_CALIBRATION_2026-07-29.md`; kolejną bramką są wyniki Phase 4 z zachowanych milestone checkpointów.

## Zalecany następny etap

Przed nową serią eksperymentów należy:

1. unieważnić historycznie ujawniony klucz W&B;
2. przeliczyć i zapisać pochodzenie statystyk Fashion-MNIST oraz Tiny ImageNet;
3. wykonano walidacyjnie sterowany czterofazowy smoke (`20049968`, `COMPLETED`);
4. porównać all-at-once z sekwencją osobnych faz, jeżeli oba warianty mają być interpretowane wspólnie;
5. wykonano: aktywny runner tworzy lokalny manifest datasetu, subsetu, próby FIM, kodu, środowiska, konfiguracji i artefaktów.
