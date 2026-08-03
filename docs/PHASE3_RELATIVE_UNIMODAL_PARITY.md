# Relative unimodal parity w Phase 3

## Cel i definicja

Wariant `decision_rule: relative_unimodal_parity` normalizuje jakość obu
gałęzi przez osiągalną jakość odpowiadającego im klasyfikatora unimodalnego.
Wszystkie wartości pochodzą z tego samego `validation_proper`; test nie jest
wejściem treningu referencji, stoppera ani selekcji checkpointu.

Dla accuracy lewego i prawego modelu referencyjnego `U_L`, `U_R`, bazowej
dominant-only accuracy przed interwencją `D_0` oraz weak-only accuracy po
lokalnej epoce `e`, reguła używa:

```text
dominant_ratio = D_0 / U_L
weak_ratio(e)  = W(e) / U_R
parity_gap(e)  = weak_ratio(e) - dominant_ratio
recovery_fraction(e) =
  (weak_ratio(e) - weak_ratio(0))
  / (dominant_ratio - weak_ratio(0))
```

`dominant_ratio` jest zamrażane przed pierwszym krokiem Phase 3. Aktualna
dominant-only accuracy jest nadal logowana jako kontrola niezmienności, lecz
nie przesuwa celu online.

Parametr `recovery_fraction_threshold` definiuje praktyczny cel odzyskania
deficytu. `1.0` zachowuje historyczne exact parity, `0.90` oznacza odzyskanie
90% początkowego deficytu, a `0.95` — 95%. Wartość może przekraczać 1, gdy
weak branch przekroczy parity, ale skonfigurowany próg należy do przedziału
`(0, 1]`.

## Referencje unimodalne

Dla każdego seedu trenowana jest osobna para `left_proper`/`right_proper` na
train 44k z czystymi polami. Przed utworzeniem każdego modelu seed jest
resetowany, a następnie konstruowany jest kompletny model bimodalny. Aktywny
encoder i shared trunk/classifier zachowują własne kanoniczne wagi z tej samej
inicjalizacji; prawa gałąź nie jest kopią lewej. Inaktywna gałąź jest
zamrożona.

Trening trwa 200 epok. Co pięć epok na validation proper wybierany jest
checkpoint o najwyższej accuracy, następnie najniższym loss i wcześniejszej
epoce. Checkpoint zapisuje metadane v2: modality, seed, model, split, normalizację,
politykę `canonical_bimodal_components_v2`, SHA-256 pełnego początkowego
`state_dict`, epokę i metryki. Phase 3 odrzuca referencję niezgodną z bieżącym
protokołem oraz parę pochodzącą z różnych inicjalizacji bimodalnych.

Przykład treningu pary:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  config=configs/experiments/cifar10_unimodal_reference.yaml \
  mode=left_proper seed=83
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  config=configs/experiments/cifar10_unimodal_reference.yaml \
  mode=right_proper seed=83
```

## Zamrażanie i zatrzymanie

W Phase 3 tylko `right_branch` ma `requires_grad=True` i pozostaje w
`train()`. `left_branch` oraz `main_branch` mają `requires_grad=False` i
`eval()`, więc ich wagi, gradienty, Dropout oraz bufory BatchNorm nie zmieniają
się. Gradient przechodzi przez zamrożony shared trunk do prawego encodera.
Optimizer zawiera wyłącznie parametry prawej gałęzi; czteroepokowy warm-up LR
od `lr/10` nadal działa per optimizer step.

Jeżeli weak branch nie ma bazowego deficytu względem dominant branch,
interwencja jest pomijana. W przeciwnym
razie walidacja odbywa się w e3=1,2,3,4, następnie 8,12,16,… i w końcowej
epoce. Pierwsze `recovery_fraction >= recovery_fraction_threshold` zapisuje
kandydata. Drugie kolejne trafienie potwierdza stop, ale Phase 4 rozpoczyna
się z pierwszego checkpointu tej nieprzerwanej serii. Przerwanie serii
resetuje kandydata.

Przy braku parity do e3=200 wybierany jest największy weak ratio, z
tie-breakerami: niższy weak-only loss i wcześniejsza epoka. NaN/Inf wybiera
najlepszy wcześniejszy skończony checkpoint, a przy jego braku e3=0.

Phase 4 odblokowuje cały model przed utworzeniem świeżego optimizera i
schedulera. Zachowuje limit 200 epok, wybór na validation proper i final-only
test.

## Pełna trajektoria kalibracyjna

Profil `cifar10_relative_unimodal_parity_trajectory_p1_40_v1` działa w
`observe_only`, zawsze wykonuje 200 epok Phase 3 i nie uruchamia Phase 4.
Pierwsza hipotetyczna decyzja stoppera jest zamrażana, ale nie wpływa na
trening. Surowe pomiary z e3=0,1,2,3,4,8,12,...,200 są dopisywane do
`phase3_trajectory.jsonl`. Każdy wersjonowany rekord przechowuje wszystkie
metryki i wyrównane wartości per-example, ratio, parity gap, recovery fraction
oraz stan decyzji.

Checkpointy wszystkich mierzonych epok są zachowywane jako calibration
milestones. Dzięki temu nową regułę można najpierw odtworzyć offline na
trajektorii, a następnie uruchomić wyłącznie Phase 4 z wybranego checkpointu.
Plik JSONL może zawierać powtórzony numer epoki po resume; replay musi wówczas
wybrać ostatni kompletny rekord dla danego `phase_epoch`.

Replay progów 90% i 95% wykonuje:

```bash
python -m scripts.python_new.replay_unimodal_recovery_fraction \
  --threshold 0.90 --threshold 0.95 \
  --trajectory 83=/path/to/phase3_trajectory.jsonl \
  --output-dir analysis/results/unimodal_recovery_fraction
```

## Interpretacja i ograniczenia

Exact parity oznacza równy ułamek jakości osiągalnej przez każdą modalność, a
nie równe surowe accuracy. Recovery fraction dodatkowo mierzy część deficytu
usuniętą względem własnego stanu weak branch sprzed interwencji. Dzięki temu
naturalnie mniej informacyjna prawa modalność nie musi dogonić bezwzględnego
wyniku lewej, a stopper nie musi czekać na końcówkę asymptoty. Reguła nie gwarantuje
jednak maksymalnego full accuracy po Phase 4; jest lokalnym, operacyjnym
kryterium długości interwencji. Jej wynik należy porównać z fixed-e3 na wielu
seedach, bez używania testu do zmiany reguły.
