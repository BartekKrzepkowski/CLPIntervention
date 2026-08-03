# Analiza kalibracji weak-recovery PAIS v2 — 2026-07-29

## Zakres

Analiza obejmuje trzy pełnodatasetowe ślady `observe_only` z CIFAR-10, Phase 1 = 80, Phase 2 = 200, Phase 3 = 200 i Phase 4 = 0. Walidacja proper była wykonywana co 5 epok, FIM był wyłączony, a test nie był używany. Runy:

| Seed | Slurm | W&B | Stan |
|---:|---:|---|---|
| 83 | `20121024` | [0f7ndiot](https://wandb.ai/bartekk/CLPIntervention_PAIS/runs/0f7ndiot) | `COMPLETED` |
| 184 | `20121025` | [ywho5vir](https://wandb.ai/bartekk/CLPIntervention_PAIS/runs/ywho5vir) | `COMPLETED` |
| 285 | `20121026` | [s70pb1qt](https://wandb.ai/bartekk/CLPIntervention_PAIS/runs/s70pb1qt) | `COMPLETED` |

Każdy run zachował pre-Phase-3, milestone 20/40/60/80/120, końcową granicę Phase 3 oraz aktywny best. Wszystkie wymagane pliki istnieją.

## Wynik kontrolera

| Seed | Baseline weak acc. | Baseline weak loss | Hipotetyczny stop | Wybrana epoka | Weak acc. | Weak loss | Final train–val gap |
|---:|---:|---:|---|---:|---:|---:|---:|
| 83 | 0,2182 | 6,1652 | max epochs | 200 | 0,8356 | 1,1603 | 1,1539 |
| 184 | 0,2534 | 5,4406 | max epochs | 185 | 0,8152 | 1,3469 | 1,3724 |
| 285 | 0,1530 | 7,2424 | max epochs | 195 | 0,8196 | 1,3014 | 1,3145 |

Żaden run nie uruchomił `recovery_plateau`, `trend_reversal` ani numerical emergency stop. Wszystkie doszły do limitu 200 epok. `is_safe` pozostawało fałszywe, zgodnie z oczekiwanym compatibility drift; recovery feasibility była osiągana dzięki poprawie weak-only.

## Najważniejsza obserwacja

Weak-only accuracy poprawiała się jeszcze powoli pod koniec interwencji, ale weak-only validation loss osiągał minimum bardzo wcześnie:

| Seed | Minimum validation loss | Epoka minimum | Accuracy w minimum | Najlepsza accuracy | Epoka najlepszej accuracy |
|---:|---:|---:|---:|---:|---:|
| 83 | 0,6705 | 15 | 0,7828 | 0,8356 | 200 |
| 184 | 0,7970 | 10 | 0,7356 | 0,8152 | 185 |
| 285 | 0,7477 | 15 | 0,7644 | 0,8196 | 195 |

Po minimum validation loss rósł, podczas gdy loss deterministycznego weak-only train probe spadał niemal do zera. W epokach 20 → 200 train–validation gap wzrósł odpowiednio:

- seed 83: 0,496 → 1,154;
- seed 184: 0,680 → 1,372;
- seed 285: 0,539 → 1,315.

Jest to silna diagnostyka memorization/overconfidence. Train probe nie może być wejściem stoppera, ale potwierdza, że późne drobne zyski accuracy nie są darmowe.

## Weak utility i compatibility

Największy `weak_utility_gain` względem stanu sprzed Phase 3 wyniósł:

- seed 83: +0,308 w epoce 140;
- seed 184: -0,557 w epoce 5;
- seed 285: -0,091 w epoce 90.

Dla seedów 184 i 285 pełny model w żadnym pomiarze nie odzyskał bazowej marginalnej użyteczności prawej modalności. Oznacza to, że prawa gałąź stała się samodzielnie informacyjna, ale wspólny trzon i klasyfikator nie wykorzystywały jej po reaktywacji obu gałęzi równie dobrze jak przed interwencją. Compatibility drift był duży, lecz w v2 pozostaje diagnostyką, a nie awaryjnym stopem.

Nie można jeszcze uznać tego za porażkę interwencji: właściwym endpointem pracy jest wynik po Phase 4, której te ślady celowo nie wykonywały.

## Dlaczego obecny stopper nie zadziałał

1. Plateau wymaga jednocześnie górnej granicy slope accuracy ≤ 0,0005/epokę i górnej granicy slope poprawy loss ≤ 0,001/epokę przez trzy walidacje.
2. Korekta dla powtarzanych spojrzeń i dziewięciu rodzin metryk poszerza przedziały. Seed 83 minął warunek przy epoce 80 tylko o około 0,000097 na slope accuracy; seed 184 miał maksymalnie dwa kolejne potwierdzenia; seed 285 nie miał żadnego.
3. Ranking jest leksykograficzny z accuracy na pierwszym miejscu. Poprawa accuracy większa niż `min_delta=0.001` zastępuje best nawet wtedy, gdy validation loss jest znacznie gorszy.
4. Reversal wymaga jednoczesnego, statystycznie potwierdzonego pogorszenia accuracy i loss względem ruchomego best. Drobne późne rekordy accuracy stale przesuwają punkt odniesienia, dlatego licznik reversal nie wzrósł ani razu.

## Analiza wrażliwości plateau

Post-hoc symulacja użyła wyłącznie zapisanych górnych granic slope. Nie zmienia runów ani checkpointów.

| Próg accuracy slope | Próg loss-quality slope | Patience | Seed 83 | Seed 184 | Seed 285 |
|---:|---:|---:|---:|---:|---:|
| 0,0005 | 0,001 | 3 | brak | brak | brak |
| 0,0010 | 0,005 | 3 | 90 | 115 | 115 |
| 0,0015 | 0,005 | 3 | 70 | 95 | 65 |

Wariant 0,001/0,005 ma najmniejszy rozrzut między seedami i zatrzymuje przed skrajnym memorization. Progi mają interpretację minimalnej wartości praktycznej: 0,001 accuracy/epokę odpowiada maksymalnie 0,5 p.p. możliwej poprawy między walidacjami oddalonymi o 5 epok; 0,005 loss/epokę odpowiada 0,025 loss na taki interwał. Jest to kandydat kalibracyjny, nie zamrożona reguła publikacyjna.

## Problem selekcji checkpointu

Obecny ranking accuracy-first wybiera 185–200. Lepszy kontrakt to:

1. znaleźć checkpointy, których weak-only accuracy jest nie gorsza od najlepszego o z góry ustalony margines `delta_accuracy` z wykorzystaniem sparowanej nieinferiority;
2. w tym zbiorze minimalizować weak-only validation loss;
3. następnie preferować mniejszy full/dominant drift i wcześniejszą epokę.

Surowa analiza z marginesem 1 p.p. wybrałaby epoki 60, 70 i 100 zamiast 200, 185 i 195. Jest to wyłącznie analiza wrażliwości; `delta_accuracy` musi zostać zamrożone przed finalną serią. Train-probe gap pozostaje diagnostyką i nie może wejść do rankingu.

## Wniosek metodologiczny

Obecnego `weak_recovery` nie należy jeszcze używać w `enforce`. Kalibracja wykazała dwie niezależne rzeczy:

- mechanizm poprawnie wykrywa dużą poprawę słabej gałęzi bez fałszywego hard safety;
- plateau i ranking są zbyt konserwatywne wobec późnych, małych zysków accuracy i wybierają obszar silnego overfittingu loss.

Nie wolno jednak wybierać ostatecznej długości Phase 3 wyłącznie z weak-only. Celem eksperymentu jest wynik pełnego modelu po Phase 4.

## Rekomendowany następny eksperyment

Dla każdego seedu uruchomić Phase 4 z już zachowanych checkpointów Phase 3:

- pre-intervention (`e3=0`),
- `e3=20, 40, 60, 80, 120`,
- końcowy/aktywny late best (`e3≈185–200`).

Checkpoint wybierać wyłącznie na `validation_proper`, raportując oba wcześniej uzgodnione budżety Phase 4: pełne 200 oraz `200-e3`. Test pozostaje niewykorzystany. Następnie sprawdzić, czy kandydat stopu 0,001/0,005/patience 3 znajduje się w validation-equivalent optimum końcowego wyniku Phase 4. Dopiero wtedy zamrozić stopper i uruchomić finalne wieloseedowe eksperymenty z jednokrotną oceną testu.
