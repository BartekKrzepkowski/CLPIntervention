# Analiza lokalnego stoppera Phase 3 — 2026-08-01

## Reguła

`decision_rule: local_accuracy` korzysta wyłącznie z `validation_proper` i
łączy cztery elementy:

1. sparowany cel `weak-only accuracy >= dominant-only accuracy`, z dodatnim
   przyrostem weak accuracy, lokalną ochroną trendu full/dominant i dwoma
   potwierdzeniami;
2. lokalne odwrócenie Pareto w przestrzeni `(weak, full, dominant)` accuracy;
3. futility-with-harm: brak możliwej dodatniej poprawy weak accuracy oraz
   równoczesny spadek full/dominant;
4. gradient conflict jako alternatywne potwierdzenie szkody w punkcie 3.
   Konflikt samodzielnie nigdy nie zatrzymuje treningu.

Opisane niżej runy historyczne mierzyły walidację co 5 epok, gradient probe co
10 epok i wymagały pięciu walidacji. Po tej analizie profil następnych runów
zmieniono na gęste pomiary e3=1,2,3,4, następnie e3=8,12,16,…, czteropomiarowe
okno trendu i minimalną ekspozycję czterech pomiarów. FIM, loss, train probe i
test są diagnostyczne i nie sterują decyzją.

## Dokładne shadow z paired CI

Slurm `20273663`, seed 83, zakończył się poprawnie w 11:56. Run W&B:
`bartekk/CLPIntervention_Phase3Stopping/pr4byikq`.

- pierwsze spełnienie targetu: e3=25;
- drugie potwierdzenie i hipotetyczny stop: e3=30;
- wybrany checkpoint: e3=30;
- weak-only accuracy: 0,7934;
- dominant-only accuracy: 0,6868;
- full accuracy: 0,6574.

Gradient conflict pojawiał się sporadycznie, między innymi w e3=20, ale
`weak_futile` pozostawało fałszywe. Zgodnie z kontraktem nie spowodowało to
zatrzymania. Run działał w `observe_only`, dlatego wykonał pełne 80 epok i
zamroził pierwszą hipotetyczną decyzję e3=30.

Identyczna, niezmieniona reguła została następnie uruchomiona dla pozostałych
seedów:

| seed | Slurm | W&B | paired-CI stop/selection | weak / dominant / full accuracy w wybranym e3 |
|---:|---:|---|---:|---:|
| 83 | 20273663 | `pr4byikq` | 30 / 30 | 0,7934 / 0,6868 / 0,6574 |
| 184 | 20281023 | `rxmj419y` | 35 / 35 | 0,7952 / 0,5416 / 0,5070 |
| 285 | 20281024 | `9ghnidgh` | 30 / 30 | 0,7830 / 0,5510 / 0,4258 |

W każdym seedzie pierwszą zamrożoną decyzją był `target_reached`. Dla seedu
285 pełna obserwowana trajektoria wykryła dodatkowo `pareto_reversal` w
e3=65. Żaden run nie uruchomił `futility_with_harm`: weak accuracy nadal
miała możliwy dodatni trend, więc ani chwilowa szkoda accuracy, ani konflikt
gradientów nie mogły samodzielnie przerwać interwencji.

## Walidacja względem Phase 4

Wszystkie 15 runów P4=200 dla `e3={20,40,60,80,200}` i seedów
`83/184/285` zakończyło się poprawnie. Checkpoint Phase 4 był wybierany na
`validation_proper`; test policzono dopiero post hoc.

| e3 | mean validation full acc | mean test proper acc | mean validation dominant acc | mean validation weak acc |
|---:|---:|---:|---:|---:|
| 20 | 0,8869 | 0,8711 | 0,8255 | 0,2989 |
| 40 | **0,8912** | 0,8796 | 0,8197 | 0,4197 |
| 60 | 0,8821 | 0,8714 | 0,4282 | 0,7179 |
| 80 | 0,8714 | 0,8588 | 0,2567 | 0,8057 |
| 200 | 0,8514 | 0,8391 | 0,1309 | 0,8231 |

| seed | replay: stop / selection | exact paired-CI selection | best tested e3 | best validation / test proper acc |
|---:|---:|---:|---:|---:|
| 83 | 40 / 40 | 30 | 60 | 0,9038 / 0,8927 |
| 184 | 60 / 50 | 35 | 40 | 0,8902 / 0,8772 |
| 285 | 30 / 30 | 30 | 40 | 0,8848 / 0,8755 |

Dokładne wybory 30/35/30 leżą w przedziale między milestone e3=20 i e3=40.
Dla seedów 184 i 285 przedział zawiera indywidualne optimum e3=40. Dla seedu
83 jest o jeden przedział wcześniejszy od optimum e3=60, ale e3=40 traci do
niego tylko 0,52 punktu procentowego validation accuracy, a agregatowe
optimum trzech seedów wypada w e3=40. Nie wykonano dodatkowego P4 z dokładnie
wybranych checkpointów e3=30/35/30; ich wyniku nie należy interpolować ani
zastępować testem.

Najważniejszy sygnał metodologiczny to załamanie dominant-only już przy
e3=60 dla seedów 184 i 285, podczas gdy weak-only nadal rośnie. Dlatego
sam weak-only target lub maksimum weak accuracy byłoby niewystarczające.
Lokalne Pareto i futility-with-harm pozostają potrzebnymi wyjściami dla
trajektorii, na których crossover nie wystąpi albo nastąpi przez degradację
pozostałych trybów.

## Decyzja

Trzy dokładne shadow runy zgodnie wybierają obszar e3=30–35. Regułę należy
oceniać na downstream P4 bez używania testu do zmiany decyzji. Ewentualna
zmiana sposobu wyboru checkpointu wewnątrz okna potwierdzeń wymaga osobnej
kontroli na validation proper.

## Downstream P4 z dokładnych wyborów stoppera

P4=200 uruchomione bezpośrednio z e3=30/35/30 zakończyły się poprawnie:

| seed | wybrane e3 | validation full | test proper | validation dominant | validation weak |
|---:|---:|---:|---:|---:|---:|
| 83 | 30 | 0,8922 | 0,8839 | 0,8380 | 0,3526 |
| 184 | 35 | 0,8860 | 0,8746 | 0,7928 | 0,3544 |
| 285 | 30 | 0,8812 | 0,8713 | 0,8408 | 0,3214 |
| średnia | — | 0,8865 | 0,8766 | 0,8239 | 0,3428 |

Względem wspólnego fixed-e3=40 stopper traci średnio 0,47 pp validation
full i 0,30 pp test proper, zachowuje o 0,42 pp wyższą dominant-only
accuracy, ale ma o 7,69 pp niższą weak-only accuracy po P4. Względem e3=20
pełna accuracy jest praktycznie identyczna (-0,05 pp), przy poprawie
weak-only o 4,39 pp. Reguła jest więc oszczędna i chroni dominant branch,
ale prawdopodobnie zatrzymuje nieco za wcześnie względem najlepszego
kompromisu full/weak w e3≈40.

Pierwszy obserwowany crossover weak≥dominant wystąpił już w e3=5 we
wszystkich seedach i był od razu potwierdzony paired CI. Nie można ustalić,
czy nastąpił w e3=1–4, ponieważ walidacja jest wykonywana co 5 epok. Sam
crossover nie lokalizuje więc optimum; długość 30–35 wynika z minimalnej
ekspozycji, okna trendu i dwóch potwierdzeń targetu.

W seedzie 184 pierwszy target e3=30 miał full/dominant 0,5766/0,5962, a
checkpoint potwierdzający e3=35 tylko 0,5070/0,5416, przy wzroście weak o
0,40 pp. Pierwsza próba P4 (`20282557`) nie rozpoczęła treningu, ponieważ
rolling buffer usunął tymczasowy e3=30 po zamrożeniu e3=35. Job `20282574`
odtworzył Phase 3 do e3=30 z tego samego checkpointu sprzed interwencji.
Otrzymano weak/full/dominant accuracy 0,7916/0,5520/0,5846. Weak odtworzyło
się z różnicą 0,04 pp, natomiast full/dominant były niższe o 2,46/1,16 pp niż
w pierwotnej trajektorii, co wskazuje na niedeterministyczność GPU. Zależny P4
`20282582` sprawdza, czy patience powinno potwierdzać stop, ale selekcja
powinna odtwarzać pierwszy checkpoint nieprzerwanego okna targetu.

Artefakty:

- `analysis/results/phase3_local_accuracy_replay_p1_40.json`;
- `analysis/results/phase4_milestone_p1_40.csv`;
- `analysis/results/phase4_stopper_selected_p1_40.csv`;
- `analysis/results/phase3_first_weak_dominant_crossover_p1_40.csv`;
- `docs/figures/phase3_local_stopper_phase4_validation.png`;
- `docs/figures/phase3_local_stopper_phase4_validation.pdf`.
