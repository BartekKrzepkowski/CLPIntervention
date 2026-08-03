# Analiza kalibracji PAIS — 2026-07-29

## Zakres

Trzy runy `observe_only` używały P1=80, P2=200, P3=200, P4=0, splitu 44k/5k/1k i pełnego `validation_proper` co 5 epok. FIM był wyłączony. Runy są dostępne w W&B:

- seed 83: https://wandb.ai/bartekk/multi/runs/dxrlm258
- seed 184: https://wandb.ai/bartekk/multi/runs/15za2h9g
- seed 285: https://wandb.ai/bartekk/multi/runs/mztkajaf

## Phase 2

Żaden seed nie spełnił wielosygnałowego plateau. Wszystkie wykonały 200 epok; kandydaci końcowego okna pochodzili odpowiednio z epok 170, 180 i 170. Full loss był od dawna bez poprawy, lecz weak-only loss nie był stabilny: końcowe nachylenia wynosiły +0.0113, +0.0202 i +0.0509 loss/epokę. Plateau confirmations pozostało równe zero.

## Pierwsza decyzja PAIS w Phase 3

| seed | hipotetyczny stop | weak-only accuracy: baseline -> stop | weak quality gain | weak utility gain | full loss increase (LCB) | dominant loss increase (LCB) | wybór |
|---:|---:|---:|---:|---:|---:|---:|---|
| 83 | 30 | 0.1656 -> 0.8088 | +4.5926 | -0.6491 | +1.0580 (+0.8570) | +0.4089 (+0.2436) | rollback pre-P3 |
| 184 | 15 | 0.2444 -> 0.7646 | +5.0845 | -0.6749 | +1.1268 (+0.9276) | +0.4519 (+0.2556) | rollback pre-P3 |
| 285 | 15 | 0.1258 -> 0.7586 | +7.3250 | -0.3449 | +0.5800 (+0.3873) | +0.2351 (+0.0486) | rollback pre-P3 |

Dla seedów 184 i 285 hard safety nastąpiło przed minimalną ekspozycją 25 epok. Jest to zgodne z kodem: hard safety jest niezależnym zabezpieczeniem i nie czeka na minimum exposure. Seed 83 przekroczył twarde ograniczenie przez trzy kolejne walidacje w epokach 20, 25 i 30; pozostałe seedy w epokach 5, 10 i 15.

Żaden punkt nie był `safe`, ponieważ konserwatywna górna granica full/dominant loss increase nie mieściła się w marginesie 0.05. W konsekwencji żaden punkt nie mógł być `feasible` i wszystkie selekcje zakończyły się rollbackiem do checkpointu sprzed Phase 3.

## Trajektoria do 200 epok

Mimo zamrożenia pierwszej hipotetycznej decyzji trening w observe-only trwał do 200 epok. Końcowe weak-only accuracy wyniosło 0.8362, 0.8222 i 0.8188. Jednocześnie generalization gap weak-only wzrósł odpowiednio do 1.202, 1.321 i 1.225 loss, a train probe dochodził do niemal perfekcyjnego dopasowania. Potwierdza to późną memorization/overconfidence, choć sam gap nie wyznacza checkpointu.

Weak quality poprawiło się bardzo mocno, ale weak utility było zwykle ujemne. Oznacza to, że enkoder słabej gałęzi nauczył się klasyfikować, natomiast wspólna część dostrojona w trybie weak-only utraciła natychmiastową kompatybilność z zamrożoną gałęzią dominującą. Seed 285 osiągał chwilowo dodatni średni weak utility gain w późnych epokach, ale przy bardzo dużej degradacji full i dominant-only.

## Wnioski metodologiczne

1. Marginesy safe=0.05 i hard=0.20 nie mogą zostać zamrożone. Przy pełnej deaktywacji traktują spodziewany przejściowy compatibility drift jako katastrofię.
2. Samo poluzowanie marginesów do wartości rzędu 1–2 loss uczyniłoby pojęcie `safe` mało znaczącym. Problem dotyczy roli metryki, nie tylko liczby progowej.
3. Obecna reguła nie znajduje kompromisu z pracy: weak-only szybko się poprawia, podczas gdy wynik po reaktywacji pogarsza się jeszcze przed okresem, w którym późniejsza Phase 4 może odzyskać kompatybilność.
4. Shadow controller zamraża cały stan po pierwszym triggerze. Ponieważ hard safety zadziałało jako pierwsze, runy nie wyznaczają niezależnego czasu futility ani reversal. Kalibracja konkurujących mechanizmów wymaga osobnych liczników shadow, które po zapisaniu pierwszego triggera nadal śledzą pozostałe reguły.
5. Zredukowany zapis checkpointów zachował pre-P3 i granicę P3, ale nie pośrednie stany odrzucone jako unsafe. Ocena związku lokalnej reguły z wynikiem po Phase 4 wymaga niewielkiego zestawu milestone checkpointów kalibracyjnych albo taniego recovery lookahead na kopii modelu.
6. W&B nadpisuje generyczne `epochs/phase` wartością 0 przy Phase 4 epoch 0 na tym samym global step co ostatnia walidacja Phase 3. Metryki Phase 3 są poprawne, ale do jednoznacznych wykresów należy logować także namespaced `phase3/phase_epoch`.

## Rekomendowany następny wariant

Nie uruchamiać jeszcze publikacyjnego `enforce`. Najpierw rozdzielić:

- numeryczny emergency stop dla NaN/divergence;
- lokalny stop weak-branch recovery oparty na sparowanym plateau/reversal weak-only accuracy oraz loss;
- compatibility drift jako diagnostykę i tie-breaker, a nie natychmiastową bramkę 0.05;
- osobne shadow timestamps hard safety, futility i reversal;
- mały zestaw milestone checkpointów tylko w kalibracji, aby sprawdzić wynik po identycznej Phase 4.

Dopiero zgodność lokalnego stopu z późniejszym wynikiem recovery pozwoli zamrozić globalną regułę przed świeżymi seedami wynikowymi.

## Dalsze wdrożenie

Rekomendację wdrożono jako osobny `decision_rule: weak_recovery`, bez zmiany semantyki legacy PAIS. Konfiguracje `cifar10_pais_recovery_calibration.yaml` i `cifar10_validation_protocol_recovery_seed83.yaml` używają sparowanego correctness/loss, numerical-only emergency stop, diagnostic compatibility oraz projektu W&B `bartekk/CLPIntervention_PAIS`. Train-probe gap pozostał diagnostyczny, aby stopping nadal był validation-only.
