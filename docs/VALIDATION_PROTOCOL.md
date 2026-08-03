# Walidacyjny protokół CIFAR-10 i sterowanie fazami

## Zakres i status

Nowy protokół jest opt-in i dotyczy obecnie standardowego, niedestylowanego
`all_at_once` dla `mm_cifar10`. Konfiguracje historyczne zachowują dawny
profil 50k oraz dawne przejścia faz. UMT nie korzysta jeszcze z kontrolerów
walidacyjnych.

Konfiguracje robocze:

```text
configs/experiments/cifar10_pais_calibration.yaml
configs/experiments/cifar10_validation_protocol_adaptive_seed83.yaml
configs/experiments/cifar10_pais_recovery_calibration.yaml
configs/experiments/cifar10_validation_protocol_recovery_seed83.yaml
configs/experiments/cifar10_pais_recovery_shadow_smoke.yaml
```

Pierwsza prowadzi trzy ślady metodologiczne w `observe_only`, bez FIM i bez
kosztownej Phase 4 recovery. Druga jest wariantem operacyjnym `enforce` z FIM
co 10 epok. Marginesy non-inferiority pozostają eksperymentalne do czasu
zamrożenia decyzji naukowej.

## Podział danych i normalizacja

Profil `cifar10_stratified_44k_5k_1k_seed83_v1` dzieli oryginalny train
CIFAR-10 na:

- train: 44 000, po 4 400 przykładów każdej klasy;
- validation: 5 000, po 500 przykładów każdej klasy;
- FIM probe: 1 000, po 100 przykładów każdej klasy.

Oryginalny test 10 000 pozostaje bez zmian. Seed podziału 83 jest stałą
protokołu i nie zależy od seedu modelu. Dla każdej klasy wykonywana jest
osobna permutacja `numpy.random.default_rng(83)`: najpierw wybierana jest
walidacja, potem FIM, a reszta trafia do treningu. Posortowane indeksy, targety
i surowe pliki CIFAR-10 mają zweryfikowane SHA-256 w
`configs/data/cifar10_stratified_44k_5k_1k_seed83_v1.json`. Niezgodność
przerywa run.

Statystyki populacyjne policzono bez augmentacji wyłącznie na finalnym train
44k. Phase 1 używa `proper_left` i `blurred_right`; Phase 2–4 używają
`proper_left` i `proper_right`. Rozmycie podczas liczenia statystyk ma tę samą
deterministyczną parę resize co bodziec treningowy. Profil publikacyjny
obsługuje obecnie wyłącznie `overlap=0.0`.

`proper_right_subset_path` nadal zawiera surowe indeksy oryginalnego train
CIFAR-10. Indeks validation, FIM albo spoza datasetu powoduje błąd.

Test nie jest tworzony przez `CLPTrainingLoaders`. `CLPTestLoaders` powstaje
dopiero po zakończeniu Phase 4 i służy wyłącznie końcowej ocenie wybranych
checkpointów.

## Wspólna ewaluacja modalności

Każda decyzja online korzysta wyłącznie z `validation_proper`. Jedna usługa
ewaluacyjna mierzy na identycznych przykładach:

- `full`: obie gałęzie aktywne;
- `dominant_only`: tylko lewa, historycznie zdrowa/dominująca gałąź;
- `weak_only`: tylko prawa, historycznie zaburzona gałąź;
- `intervention_mode`: dokładny tryb bieżącej interwencji.

Ewaluacja używa `eval()` i `no_grad()`, nie aktualizuje wag, gradientów,
Dropout ani buforów BatchNorm i odtwarza wcześniejszy tryb każdego modułu.

Marginalna użyteczność słabej modalności w przestrzeni straty to:

```text
weak_utility_loss = dominant_only_val_loss - full_val_loss
```

Wartość dodatnia oznacza, że dołączenie prawej modalności obniża loss.
`weak_only` mierzy jakość samej reprezentacji, a `weak_utility` także to, czy
wspólny trzon i klasyfikator potrafią z niej korzystać.

`validation_blurred` jest wyłącznie diagnostyką. Nie zatrzymuje fazy i nie
wybiera checkpointu.

## Phase 2: plateau wielosygnałowe

Phase 2 ma w nowym protokole budżet 0 albo 200 epok. Zero oznacza bezpośrednie
przejście z ekspozycji Phase 1 do interwencji Phase 3. Dla budżetu 200 plateau
nie jest zwykłym early stoppingiem na `full_val_loss`: pełny model może już
nie poprawiać wyniku, mimo że słaba gałąź nadal się uczy.

`Phase2PlateauDetector` wymaga jednocześnie:

1. minimalnej liczby epok;
2. braku istotnej poprawy `full_val_loss` przez ustaloną patience;
3. stabilnego nachylenia `weak_only_val_loss`;
4. stabilnego nachylenia `weak_utility_loss`;
5. kilku kolejnych potwierdzeń wszystkich warunków.

Nachylenia pochodzą z regresji liniowej względem rzeczywistych lokalnych epok,
więc mają jednostkę loss/epokę. Stabilizacja nie oznacza zrównania jakości
modalności.

Po plateau lub limicie checkpoint wybierany jest tylko z końcowego okna
ewaluacji: najniższy full loss, najwyższy weak utility, najniższy weak-only
loss, a na końcu wcześniejsza epoka.

## Phase 3: legacy PAIS, weak-recovery v2 i compatibility drift

Przed pierwszym krokiem interwencji zapisywany jest checkpoint i bazowa
ewaluacja po reaktywacji obu gałęzi. Podczas treningu lewa gałąź jest
odłączona od forwardu. Każda walidacja ocenia `full`, `dominant_only`,
`weak_only` i `intervention_mode`, ponieważ sam tryb interwencji nie wykrywa
utraty zgodności wspólnego trzonu z zamrożonym lewym encoderem. Obecnie
`intervention_mode` jest równoważny `weak_only`; implementacja współdzieli ten
sam forward, ale zachowuje oba pola wyniku.

Adaptacyjny wariant kontrolera nazywa się **PAIS — Paired Adaptive
Intervention Stop**. Dla każdego z 5 000 obrazów zachowuje straty czterech
trybów i tworzy sparowane różnice względem stanu sprzed Phase 3. Dzięki temu
nie porównuje dwóch niezależnych średnich. Normalne przedziały ufności CLT są
korygowane metodą Bonferroniego względem maksymalnej liczby spojrzeń oraz
ośmiu rodzin metryk. Surowe średnie nadal są logowane; korekta dotyczy
wyłącznie decyzji. Jest to przybliżenie asymptotyczne, które wymaga raportu
wrażliwości w eksperymencie metodologicznym.

Checkpoint jest `safe`, gdy górne granice wzrostu `full` i `dominant_only`
loss mieszczą się w limitach non-inferiority. Jest `feasible`, gdy dodatkowo
dolne granice zysku jakości `weak_only` oraz `weak_utility` przekraczają
ustalone minima. Nie zakładamy równych możliwości modalności: prawa gałąź jest
porównywana ze swoim własnym stanem początkowym.

PAIS łączy trzy mechanizmy, a nie wybiera jednego z nich:

1. **Paired validation uncertainty** rozróżnia zmianę od szumu na tych samych
   obrazach.
2. **Trend reversal** działa po znalezieniu feasible checkpointu. Jeżeli przez
   `reversal_patience` pomiarów obecny model jest wiarygodnie gorszy od
   najlepszego zarówno pod względem weak utility, jak i weak-only quality,
   interwencja kończy się i odtwarzany jest wcześniejszy najlepszy checkpoint.
3. **Futility** działa przed znalezieniem feasible checkpointu. Na ostatnich
   `trend_window` pomiarach liczony jest osobny per-image slope weak utility i
   weak quality względem rzeczywistych epok. Optymistyczna granica to górna
   granica aktualnego zysku plus dodatnia część górnej granicy slope pomnożona
   przez `futility_prediction_horizon_epochs`. Futility wymaga braku optymistycznej weak utility oraz braku dalszego
   dodatniego trendu weak-only quality (albo braku jej wymaganego zysku) przez
   `futility_patience` pomiarów. Historyczna poprawa weak-only nie blokuje więc
   stopu, jeżeli się wypłaszczyła i nadal nie jest wykorzystywana przez model.

Minimalna ekspozycja wymaga jednocześnie `min_epochs` i
`minimum_exposure_evaluations`; w konfiguracji operacyjnej pięć walidacji co
pięć epok oznacza co najmniej dwadzieścia pięć epok przed zwykłym stopem. Osobny
`hard_safety` może zadziałać wcześniej po kolejnych, statystycznie
potwierdzonych przekroczeniach twardych limitów degradacji. Kolejność decyzji
to hard safety, reversal, futility i maksymalny budżet.

Ranking feasible pozostaje leksykograficzny: największy weak utility gain,
największy weak quality gain, najmniejszy wzrost full loss, najmniejszy wzrost
dominant-only loss i wcześniejsza epoka. Wybór końcowy to najlepszy feasible,
potem najlepszy safe, a przy braku obu rollback do checkpointu sprzed Phase 3.
Naturalnie mało informacyjna prawa modalność nie jest zmuszana do osiągnięcia
jakości lewej: brak wiarygodnej użyteczności prowadzi do futility i bezpiecznego
fallbacku zamiast długiego memorization.

### Weak-recovery v3: accuracy-first

Kalibracja legacy PAIS na seedach 83/184/285 wykazała, że progi compatibility
loss 0.05/0.20 uruchamiają hard safety w epokach 30/15/15, mimo wzrostu
weak-only accuracy o 52–64 p.p. Dlatego legacy `decision_rule: legacy`
pozostaje dostępne do reprodukcji, ale nie jest rekomendowanym trybem enforce.

Nowy `decision_rule: weak_recovery` zachowuje sparowane per-image loss oraz
binarne correctness, ale w rekomendowanym
`recovery_primary_metric: accuracy` decyzję prowadzi correctness. Checkpoint
recovery jest `feasible`, gdy dolna granica zysku weak-only accuracy przekracza
minimum. Compatibility safety jest niezależną diagnostyką i tie-breakerem:
oczekiwany drift podczas pełnej deaktywacji nie eliminuje recovery checkpointu
przed sprawdzeniem go w Phase 4. Ranking wybiera najwyższą weak-only accuracy,
następnie najniższy weak-only loss, mniejszy full i dominant compatibility
drift oraz wcześniejszą epokę.

Trend jest liczony osobno dla każdego obrazu względem rzeczywistych numerów
epok. `recovery_plateau` wymaga, aby górna granica nachylenia weak-only
accuracy nie przekraczała ustalonej tolerancji przez `plateau_patience`
walidacji. `trend_reversal` wymaga statystycznie potwierdzonego pogorszenia
accuracy względem najlepszego checkpointu. Loss nadal jest logowany i
rozstrzyga remisy accuracy, ale jego wzrost nie blokuje stopu. Opcjonalny tryb
`accuracy_and_loss` zachowuje dawną dwusygnałową bramkę do analiz
wrażliwości.

`emergency_stop_mode: numerical_only` reaguje wyłącznie na NaN/Inf. Oczekiwany
compatibility drift nie jest emergency stopem. Train–validation gap pozostaje
diagnostyką i nie steruje regułą, dzięki czemu wszystkie decyzje online nadal
korzystają wyłącznie z validation proper.

### Local-accuracy v4: cel, Pareto i futility-with-harm

Eksperymentalny `decision_rule: local_accuracy` usuwa bezwzględną
non-inferiority względem checkpointu sprzed Phase 3. Taki warunek mylił
oczekiwany początkowy drift po deaktywacji z nieodwracalnym uszkodzeniem.
Reguła używa wyłącznie sparowanej poprawności na `validation_proper` i ma
trzy możliwe zakończenia:

1. `target_reached`: dolna granica CI dla `weak_accuracy -
   dominant_accuracy` jest nieujemna, weak poprawiło się względem początku
   interwencji, a lokalne trendy full i dominant nie wskazują potwierdzonego
   spadku. Warunek musi utrzymać się przez `target_patience` pomiarów.
2. `pareto_reversal`: checkpoint z lokalnego rolling window jest punktowo
   nie gorszy w
   `(weak, full, dominant)` accuracy i sparowane CI potwierdza przewagę w co
   najmniej jednym wymiarze. Dwa kolejne zdominowane checkpointy kończą
   interwencję i odtwarzają wcześniejszego dominatora.
3. `futility_with_harm`: górna granica nachylenia weak accuracy nie jest
   dodatnia, a jednocześnie full/dominant ma potwierdzony lokalny trend
   spadkowy albo validation gradient probe wykrywa konflikt. Samo plateau,
   sam konflikt gradientów ani rosnący train–validation gap nie wystarczają.

Gradient probe działa domyślnie co 10 epok na pierwszym deterministycznym
batchu `validation_proper`, w `eval()`. Używa `torch.autograd.grad`, więc nie
zmienia `.grad`, parametrów, buforów BatchNorm ani trybów modułów. Loguje
normy per sqrt(liczby parametrów) dla weak encoder i shared trunk oraz
cosinusy shared-gradient weak–dominant i weak–full. Jest sygnałem
potwierdzającym futility, nie samodzielnym stopperem. FIM i train probe
pozostają wyłącznie diagnostyczne. Ostatni dostępny wynik gradient probe
obowiązuje na kolejnych pomiarach validation; dzięki temu rzadsza diagnostyka
nie zeruje sztucznie licznika potwierdzeń stoppera.

Profil operacyjny P1=40 mierzy validation w każdej z pierwszych czterech epok,
a następnie w epokach podzielnych przez cztery; końcowa epoka fazy jest zawsze
mierzona również wtedy, gdy wypada poza rytmem. `trend_window=4`,
`minimum_exposure_evaluations=4` i `min_epochs=4` pozwalają utworzyć pierwszy
pełny trend w e3=4. Dwa potwierdzenia targetu nadal są wymagane, więc samo
pojedyncze trafienie nie zatrzymuje interwencji. NaN/Inf nadal
uruchamia `emergency_stop`. Pierwsza hipotetyczna decyzja w `observe_only`
jest zamrażana, a jej checkpoint pozostaje zachowany do późniejszej Phase 4.
Rolling Pareto frontier przechowuje najwyżej
`trend_window + pareto_patience` checkpointów, więc mechanizm nie przywraca
nieograniczonego zapisu całej trajektorii.

Profil P1=40 dodaje czteroepokowy liniowy warm-up LR Phase 3. Pierwszy update
używa 10% bazowego LR, a scheduler zwiększa LR liniowo po każdym kroku
optymalizatora. Bazowy LR zostaje osiągnięty po ostatnim kroku czwartej epoki
i jest używany od początku epoki 5. Ogranicza to gwałtowny ruch wspólnego
trzonu po deaktywacji lewej gałęzi bez zamrażania parametrów. Warm-up jest
jedynym schedulerem Phase 3; po nim LR pozostaje stałe i nie jest wykonywany
żaden krok schedulera epokowego. Stan odtwarza się przy resume. Przy długości
zero Phase 3 używa stałego bazowego LR od pierwszego update.

W kalibracyjnym `observe_only` pierwsza hipotetyczna decyzja jest zamrożona jako
wynik, lecz kontroler nadal aktualizuje niezależne timestamps emergency,
reversal, recovery plateau i max epochs. Domyślne przejście do Phase 4 używa
endpointu pełnej Phase 3. Opcjonalne
`observe_phase4_transition=hypothetical_selected` najpierw wykonuje cały shadow
trace, a następnie odtwarza zamrożony hipotetyczny checkpoint (lub pre-Phase-3
przy rollbacku) i dopiero z niego uruchamia Phase 4. Global step pozostaje
monotoniczny, a `selected_source_step` ujawnia źródło wag. Jawny zestaw
`calibration_milestone_epochs` zachowuje checkpointy fixed-e3 potrzebne do
porównania; tryb operacyjny nadal zapisuje aktywne best, granice i resume.
Profil P1=40 zachowuje `e3=20,40,60,80,200`.

W&B loguje równolegle `phase3/weak_only_val_loss` oraz czytelny alias
`phase3/weak_only_loss`, a także `phase3/phase_epoch`, sparowane CI accuracy,
nachylenia i pierwsze epoki triggerów. Nowe runy używają jawnego projektu
`bartekk/CLPIntervention_Phase3Stopping`; historyczne runy pozostają w PAIS.

Kalibracja seedów 83/184/285 nie uruchomiła plateau ani reversal przed epoką 200. Obecnych progów i rankingu accuracy-first nie należy używać w `enforce` do czasu walidacji Phase 4 z checkpointów milestone. Analiza i kandydat progów są w [WEAK_RECOVERY_CALIBRATION_2026-07-29.md](WEAK_RECOVERY_CALIBRATION_2026-07-29.md).

`compatibility_drift_loss` to wzrost dominant-only loss po interwencji.
`compatibility_drift_accuracy` i `reactivation_full_loss_gap` są diagnostyką.
Dodatkowo stały, stratyfikowany train probe 1 000 obrazów z train 44k jest co
10 epok oceniany bez augmentacji w `eval()` i trybie `weak_only`. Logowany
`weak_only_generalization_gap = validation_loss - train_probe_loss` wykrywa
memorization, ale nie wybiera checkpointu i nie uczestniczy w stopping rule.

Pełne `validation_proper` jest mierzone co pięć epok. `validation_blurred` to
osobny full-only pomiar diagnostyczny co 10 epok i nie jest jednym z czterech
trybów. W konfiguracji kalibracyjnej FIM jest wyłączony; w operacyjnej jest
mierzony co 10 epok i pozostaje diagnostyką, nie wejściem PAIS.

## Tryby i checkpointy

- `disabled`: kontroler nie zmienia historycznej trajektorii;
- `observe_only`: pierwsza hipotetyczna decyzja zostaje zamrożona, ale trening
  trwa do skonfigurowanego końca i kolejna faza dostaje końcowy stan; w trybie
  weak-recovery niezależne shadow triggery są nadal aktualizowane;
- `enforce`: faza kończy się zgodnie z kontrolerem i ładowany jest wybrany
  checkpoint.

W nowym protokole każda faza tworzy świeży optimizer i scheduler. Ich stan w
checkpointcie służy dokładnemu resume wewnątrz tej samej fazy, a nie transferowi
między fazami. Przenoszone są wybrane wagi, bufory BatchNorm, diagnostyki, RNG
i stan generatora loadera. `global_step` opisuje rzeczywiście wykonane
aktualizacje i nie jest cofany; `selected_source_step` zapisuje krok źródłowy.

Checkpoint v5 zapisuje także lokalną epokę fazy, stan detectora/stoppera/
selektora i generatora train loadera. Rzeczywiste checkpointy graniczne Phase 1–3 pozostają do analizy RSV post hoc; rolling pruning nie usuwa tych granic. `resume_current` jest pojedynczym, atomowo nadpisywanym checkpointem zapisywanym
domyślnie co 10 epok w nowych konfiguracjach; po awarii można więc powtórzyć
do dziewięciu epok, ale stan z punktu zapisu jest kompletny. Kandydaci są ograniczeni do rolling window
Phase 2, pre/best feasible/best safe Phase 3 oraz dwóch selektorów Phase 4.

## Phase 4 i izolowane testy post hoc

Phase 4 wybiera na `validation_proper`:

- najlepszy checkpoint w pełnym wykonanym budżecie do 200 epok;
- najlepszy checkpoint w budżecie `200-e3`, gdzie `e3` jest lokalną epoką
  checkpointu faktycznie wybranego w Phase 3.

Kandydatem jest także epoch 0. Dla obu budżetów zachowywane są osobno minimum
validation loss i maksimum validation accuracy. Przy
`phase4_test_policy=final_only` dopiero po zakończeniu treningu wybrane
checkpointy są oceniane na test proper i test blurred; identyczny plik jest
liczony raz.

`phase2_test_policy=posthoc_final` jest osobną diagnostyką wykonywaną jeszcze
później, po zakończeniu wszystkich faz. Procedura tworzy test loadery lokalnie,
odtwarza zachowane checkpointy `best_loss` i `best_accuracy` Phase 2, zapisuje
wyłącznie `posthoc_test/phase2/*`, a w bloku `finally` przywraca końcowy
checkpoint runu. Test loader ani wynik testowy nie są przekazywane do pętli
fazowej, stoppera lub selektora. Powtarzane oglądanie tych metryk i ręczna
zmiana protokołu nadal byłoby pośrednim przeciekiem testowym.

## Kalibracja i raportowanie

PAIS ma działać operacyjnie w pojedynczym runie na podstawie jego własnych,
sparowanych danych walidacyjnych. Nie wolno kalibrować osobnego progu dla
każdej długości Phase 1 ani wymagać wcześniejszego sweepu dla nowego przypadku.
Parametry `confidence_level`, liczba rodzin, minimalna ekspozycja, okno trendu,
horyzont i patience stanowią jeden globalny protokół.

Do walidacji metodologicznej należy użyć trzech z góry wydzielonych,
sparowanych seedów i kilku reprezentatywnych warunków P1/P2. Jej celem jest
sprawdzenie pokrycia przedziałów, opóźnienia względem post-hoc maksimum i
odsetka rollbacków, a nie dopasowanie PAIS do każdej krzywej. Po zamrożeniu
reguły wynik publikacyjny powinien korzystać z co najmniej pięciu nowych
seedów. Marginesy non-inferiority full/dominant trzeba ustalić a priori,
ponieważ kod nie może sam wywnioskować, jaka degradacja jest naukowo
akceptowalna. Test pozostaje final-only.

Logi zawierają surowe średnie, przedziały, slope i ich granice, optymistyczne
granice futility, liczniki patience, feasibility, compatibility drift, globalną
i lokalną epokę, global step, source step oraz JSON summary każdej fazy.
Wygładzanie W&B pozostaje wyłącznie wizualizacją post hoc.

Żadne obliczanie statystyk, iterowanie loaderów, forwardy ani testy na pełnym
modelu nie mogą działać na login nodzie. Należy je wysyłać przez Slurm na
compute node.
