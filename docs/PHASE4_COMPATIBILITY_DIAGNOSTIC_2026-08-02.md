# Diagnostyka kompatybilności Phase 4

## Punkt wyjścia

Analiza dotyczy `P1=40/P2=200` i wariantu
`relative_unimodal_parity`. W Phase 3 trenowany jest wyłącznie
`right_branch`; `left_branch` oraz cały downstream współdzielony przez obie
modalności pozostają zamrożone w `eval()`. W Phase 4 cały model jest ponownie
odblokowany i otrzymuje świeży optimizer.

Wybrane przez validation proper checkpointy końca Phase 2 miały:

| seed | full accuracy | dominant-only accuracy | weak-only accuracy |
|---:|---:|---:|---:|
| 83 | 0,8788 | 0,8304 | 0,2596 |
| 184 | 0,8646 | 0,8068 | 0,1660 |
| 285 | 0,8734 | 0,8144 | 0,1450 |
| średnia | 0,8723 | 0,8172 | 0,1902 |

## Aktualny sweep milestone Phase 3 → Phase 4

Każdy milestone używa nowej zamrożonej Phase 3 i pełnej P4=200. Checkpoint
P4 wybierany jest według maksymalnej full accuracy na validation proper;
test pozostaje wyłączony.

| e3 | mean full accuracy | mean dominant-only accuracy | mean weak-only accuracy |
|---:|---:|---:|---:|
| 20 | 0,8833 | 0,7937 | 0,2733 |
| 40 | 0,8870 | 0,7843 | 0,3263 |
| 60 | 0,8898 | 0,7919 | 0,3595 |
| 80 | **0,8921** | 0,7866 | 0,3879 |

W przeciwieństwie do historycznej Phase 3, która trenowała również shared
trunk, nowy wariant nie wykazuje załamania dominant-only przy e3=60–80.
Pełna accuracy rośnie w całym zbadanym zakresie. Jednocześnie weak-only po
P4 pozostaje znacznie niższe niż na końcu P3.

| e3 | historyczne full | frozen-shared full | historyczne dominant | frozen-shared dominant | historyczne weak | frozen-shared weak |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0,8869 | 0,8833 | 0,8255 | 0,7937 | 0,2989 | 0,2733 |
| 40 | 0,8912 | 0,8870 | 0,8197 | 0,7843 | 0,4197 | 0,3263 |
| 60 | 0,8821 | 0,8898 | 0,4282 | 0,7919 | 0,7179 | 0,3595 |
| 80 | 0,8714 | 0,8921 | 0,2567 | 0,7866 | 0,8057 | 0,3879 |

Porównanie jest ablacją dwóch różnych zasad trainability Phase 3, a nie
powtórzeniem tego samego eksperymentu.

## Wczesny collapse w Phase 4

W sześciu runach progów recovery 90%/95% weak-only spadało już między
`P4 e0` i pierwszym pomiarem `P4 e5`: średnio o 42–44 pp. W tym samym czasie
full accuracy rosła o około 5 pp, a dominant-only traciła 7–8 pp. Weak-only
loss rósł z około 0,73 do 3,2–3,5. Żaden run nie odzyskał później weak-only
z końca P3. Jest to wczesna reoptymalizacja kompatybilności, nie wyłącznie
późny skutek selekcji checkpointu.

## Diagnostyka hybrydowa

Opt-in `phase4_diagnostics` zachowuje anchor z `P4 e0`, zawierający stan
prawego encodera oraz współdzielonego downstream. Dla S-ResNet-18 downstream
obejmuje `main_branch`, bufory oraz końcowy klasyfikator. Na validation proper
mierzone są trzy weak-only układy:

1. `current_right + current_shared` — zwykłe weak-only bieżącego modelu;
2. `current_right + anchor_shared` — izoluje dryf prawego encodera względem
   niezmienionego downstream z końca P3;
3. `anchor_right + current_shared` — izoluje dryf shared trunk/classifier
   względem niezmienionego prawego encodera z końca P3.

Podmiana parametrów i buforów jest tymczasowa, wykonywana w `eval()` i
`no_grad()`, a bieżący model oraz tryby modułów są odtwarzane bitowo. Surowe
punkty trafiają do W&B oraz `phase4_hybrid_trajectory.jsonl`.

Konfiguracja diagnostyczna wykonuje pomiary w P4 e0,1,2,3,4,5,10. Sparowana
kontrola porównuje brak warm-upu z czteroepokowym liniowym warm-upem P4 od
`lr/10`, aktualizowanym po każdym optimizer step. Test pozostaje wyłączony.

## Wynik diagnostyki e3=40

Sześć sparowanych runów dla seedów 83/184/285 zakończyło się poprawnie.
Wartości poniżej są średnimi z validation proper.

| warm-up | P4 epoch | full | dominant-only | current weak | current right + anchor shared | anchor right + current shared |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0,8286 | 0,8172 | 0,7720 | 0,7720 | 0,7720 |
| 0 | 1 | 0,8730 | 0,7166 | 0,3905 | 0,7243 | 0,4814 |
| 0 | 5 | 0,8770 | 0,7304 | 0,3657 | 0,6609 | 0,5011 |
| 0 | 10 | 0,8797 | 0,7451 | 0,3712 | 0,6355 | 0,5182 |
| 4 | 0 | 0,8286 | 0,8172 | 0,7720 | 0,7720 | 0,7720 |
| 4 | 1 | 0,8815 | 0,7216 | 0,5143 | 0,7743 | 0,5257 |
| 4 | 5 | 0,8731 | 0,7177 | 0,3712 | 0,7017 | 0,4623 |
| 4 | 10 | 0,8773 | 0,7480 | 0,3837 | 0,6585 | 0,4939 |

Bez warm-upu w P4 e1 aktualny right encoder z anchor shared zachowuje 0,7243
accuracy wobec 0,7720 na starcie, podczas gdy anchor right z aktualnym shared
spada do 0,4814. Oznacza to, że pierwszy collapse jest przede wszystkim
dryfem shared trunk/classifier. Aktualny model łączy oba dryfy nieliniowo i
spada jeszcze niżej, do 0,3905.

Warm-up wyraźnie łagodzi pierwszy krok: w e1 zwiększa current weak o 12,38 pp
i full o 0,85 pp względem braku warm-upu. Chroni również right encoder, bo
`current right + anchor shared` pozostaje na poziomie 0,7743. Efekt nie jest
jednak trwały. W e5 przewaga current weak wynosi tylko 0,55 pp, a w e10
1,25 pp; full accuracy w e10 jest o 0,23 pp niższa. Warm-up opóźnia gwałtowną
reoptymalizację, ale sam nie usuwa bodźca pełnego lossu do współadaptacji.

Dokładne średnie wszystkich siedmiu pomiarów znajdują się w
`analysis/results/phase4_compatibility_diagnostic_e40_summary.csv`.
