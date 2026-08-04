# Eksperymenty nad interwencją w Phase 3

## Cel i aktualny priorytet

Aktualnym celem jest wyznaczenie, osobno dla każdego seedu, takiej długości
Phase 3, która po ponownym treningu w Phase 4 najwierniej odtwarza clean
accuracy gold. Kryterium jest hierarchiczne: najpierw wyznacza najmniejszą
bezwzględną lukę `validation_proper full accuracy`, po czym uznaje za
równoważnych kandydatów mieszczących się w pasie do `minimum + 0,01` (1 punkt
procentowy). Wewnątrz tego pasa minimalizuje średnią bezwzględną lukę
dominant-only/weak-only. Kolejne tie-breaki to osobno luka dominant, luka weak,
luka full i wcześniejsze e3. Jest to najpierw problem wyznaczenia
referencyjnego best recovery, a dopiero później problem skonstruowania taniego
stoppera online.

Rozróżniamy dwie referencje. `Accuracy gold` jest per-seed clean
`P1=0/P2=200` i dostarcza full/dominant/weak validation accuracy. `TFIM
dynamics gold` startuje od losowej inicjalizacji i uczy się wyłącznie na
czystych danych przez `P2=17`, mierząc TFIM w e2=5/8/11/14/17. Pomiar 17 epok
po wcześniejszym P2=200 nie jest TFIM dynamics gold.

Test set nie bierze udziału w przeszukiwaniu długości Phase 3, wyborze
checkpointu ani selekcji Phase 4. Może zostać użyty dopiero po zamrożeniu
wyboru na validation proper.

## Wariant bazowy: deactivation ze wspólnym trzonem trenowalnym

Historyczna interwencja wyłącza lewą/dominant gałąź. Prawy encoder oraz shared
trunk/classifier są trenowalne. Dla `P1=120/P2=200` pełne trajectory P3=200
zostały zapisane dla seedów 83, 184 i 285.

Najlepsze validation full accuracy po pełnym P4=200 na wspólnej coarse grid:

| e3 | mean full accuracy |
|---:|---:|
| 0 | 0,8755 |
| 20 | 0,8832 |
| 40 | 0,8865 |
| 60 | 0,8847 |
| 80 | 0,8771 |

Średnio najlepsze jest e3=40, lecz optimum nie jest wspólne dla seedów:
seed 83 osiągnął dotychczas maksimum przy e3=80 (`0,9026`), seed 184 w
lokalnym punkcie e3=44 (`0,8806`), a seed 285 przy e3=60 (`0,8878`). Funkcja
celu nie jest monotoniczna, dlatego klasyczna bisekcja po downstream accuracy
nie ma uzasadnienia.

Aktualne wyszukiwanie best recovery coarse-to-fine używa per seed:

1. coarse grid `e3=0/20/40/60/80/100`;
2. P4=40 dla każdego kandydata, validation co 5 epok;
3. lokalny refinement wokół najlepszego obszaru;
4. pełne P4=200 dopiero dla finalistów;
5. końcowe sprawdzenie wszystkich nierozstrzygniętych całkowitych e3 w
   finalnym przedziale, aż pozostanie jedna epoka;
6. test tylko dla ostatecznie wybranych checkpointów.

Krótkie P4=40 jest screeningiem i nie zastępuje finalnego P4=200. TFIM jest
mierzony co 5 epok jako diagnostyka potencjalnego monotonicznego sygnału:
`Tr(F_L)`, `Tr(F_R)` i `Tr(F_L)/Tr(F_R)`.

## Stoppery oparte na lokalnej trajektorii

Pierwsze stopery używały weak-only loss, full/dominant constraints, hard
safety, reversal i futility. Loss okazał się mylący: NLL często wzrastał mimo
poprawy accuracy z powodu rosnącej pewności błędnych predykcji. Bezwzględne
marginesy compatibility powodowały rollback do e3=0.

Wariant `local_accuracy` używał sparowanej poprawności obrazów, targetu
weak≥dominant, odwrócenia Pareto `(weak, full, dominant)` oraz futility with
harm. Point replay wybrał e3=40/50/30, a dokładne paired-CI shadow e3=30/35/30
dla seedów 83/184/285. Po P4=200 dokładne wybory dały mean validation/test
full `0,8865/0,8766`, czyli były blisko coarse optimum, ale zbyt wcześnie dla
części seedów. Sam pierwszy crossover weak≥dominant następował już w e3=5 i
nie jest wystarczającym kryterium.

## Normalizacja względem modeli unimodalnych

`relative_unimodal_parity` zamraża lewy encoder i shared trunk, trenując tylko
prawy encoder. Exact parity względem dwóch seed-paired modeli unimodalnych
wybierało zbyt późno: e3=188/112/156. Recovery fraction 90% wybierało
e3=36/24/36, a 95% e3=56/36/76. Pełna repeated-look correction dla 54
spojrzeń była zbyt konserwatywna i przechodziła do fallbacków.

Ten wariant zmienia trainability względem historycznej interwencji. Jego
wyniki nie mogą być bezpośrednio używane do wyznaczenia optimum aktualnego
shared-trunk protocol.

## Frozen-left-active

W tej kontroli lewy encoder jest aktywny w forward, ale frozen/eval. Prawy
encoder i shared trunk/classifier trenują się z pełnego wyjścia. Między e3=0
i e3=200 średnie validation accuracy zmieniły się następująco:

| tryb | e3=0 | e3=200 | zmiana |
|---|---:|---:|---:|
| full | 0,8735 | 0,8714 | −0,0021 |
| dominant-only | 0,8183 | 0,8227 | +0,0044 |
| weak-only | 0,2251 | 0,2335 | +0,0083 |

Weak-only accuracy poprawia się tylko przejściowo, a weak-only loss wyraźnie
rośnie. Stała silna lewa reprezentacja pozwala shared classifierowi nadal
ignorować prawą gałąź. Jest to kontrola negatywna, nie rekomendowana
interwencja regeneracyjna.

Artefakty:

- `analysis/results/frozen_left_active_p1_120_phase3/frozen_left_active_validation_accuracy.png`;
- `analysis/results/frozen_left_active_p1_120_phase3/frozen_left_active_validation_accuracy_per_seed.png`;
- `analysis/results/frozen_left_active_p1_120_phase3/frozen_left_active_validation_loss.png`.

## UMT

Historyczny przebieg UMT był w rzeczywistości `right-only`: lewa gałąź była
deaktywowana. Między e3=0 i e3=200 mean validation weak-only wzrosło
`0,2251→0,8432`, ale dominant-only spadło `0,8183→0,3655`, a full
`0,8735→0,6072`. Nauczyciel prawej gałęzi nie chronił kompatybilności shared
trunk z lewym encoderem.

Poprawiony pełny UMT jest osobną alternatywną interwencją: obie gałęzie są
aktywne, wszystkie parametry studenta są trenowalne, a zamrożone są dwa
seed-paired modele-nauczyciele. Smoke `20361883` przeszedł; pełne trzy seedy
to `20361885/20361888/20361889`. Wyniki nie są jeszcze dostępne.

## Reintegracja w Phase 4

Hybrydowa diagnostyka wykazała, że w pierwszej epoce P4 głównym źródłem
weak-only collapse jest zmiana shared trunk/classifier, nie samego prawego
encodera. Czteroepokowy warm-up od LR/10 poprawiał weak-only o 12,38 pp w e4=1,
ale przewaga spadała do 1,25 pp w e4=10.

Prefix `shared-only`, w którym oba encodery są początkowo frozen, zachował
prawy encoder, ale sam dryf shared obniżył mean weak-only `0,7720→0,4626`.
Sam harmonogram odmrażania nie wystarcza. Wariant `L_full + L_weak` jest
zachowaną ablacją funkcji celu, nie aktualnym baseline.

## TFIM

TFIM jest sampled Fisherem liczonym na stałym class-balanced probe 1k z tymi
samymi sampled labels dla obu gałęzi. Nie steruje aktualnie stopperem. Pierwsze
punkty P4 dla shared-trunk e3=40 wskazują silną dominację lewej gałęzi:

| seed | ratio e4=10 | ratio e4=20 |
|---:|---:|---:|
| 83 | 8,39 | 8,81 |
| 184 | 3,71 | 4,07 |
| 285 | 13,19 | 14,31 |

Pełne trajektorie co 10 epok oraz coarse-oracle P4=40 co 5 epok zakończyły się
poprawnie. Wykresy nie uśredniają seedów, a oś X jest lokalną epoką Phase 4:

- `analysis/results/phase4_tfim_oracle_p1_120/phase4_tfim_ratio_oracle_all_seeds.png`;
- osobne pliki `phase4_tfim_ratio_oracle_seed{83,184,285}.png`;
- `analysis/results/phase4_tfim_shared_p1_120_e3_40/phase4_tfim_ratio_e3_40_p4_200_all_seeds.png`.

Oracle ujawnia jakościową zmianę po interwencji. Dla seedu 83 i e3=0 ratio
spada od `0,42` w e4=5 do `0,15` w e4=40, podczas gdy e3=20 zaczyna od
`8,71` i pozostaje powyżej 9. Nie należy jeszcze traktować pojedynczego progu
ratio jako stoppera: trzeba zestawić kształt każdej krzywej z najlepszym
validation checkpointem P4 osobno dla seedu i e3.

Wcześniejsze feature search, pairwise i LOOSO korzystały z downstream oracle
do kalibracji targetu albo wyboru postaci reguły. Zostały odrzucone jako
narzędzia wyznaczania końca interwencji. Ich artefakty pozostają wyłącznie
archiwum w
`$REPORTS_DIR/analysis/tfim_stopper_development_2026-08-03_04/`; nie są
częścią aktywnego stoppera ani bieżącego commita.

### Końcowy gęsty refinement slope względem clean gold

Aby oddzielić wynik od wcześniejszych, niezależnie uruchamianych trajektorii
P3, wykonano po jednej nieprzerwanej trajektorii P3 dla każdego seedu. Zapis
checkpointów materializacyjnych nie wykonuje ewaluacji i nie zmienia RNG
treningu. Zachowano każdą całkowitą epokę w końcowych zakresach: `75..80`
dla seedu 83, `50..55` dla seedu 184 oraz `65..70` dla seedu 285. Smoke
`20393538` i trajektorie `20393540`–`20393542` zakończyły się poprawnie.

Z 18 checkpointów uruchomiono niezależne P4=17 (`20394087`–`20394126`, z
przerwami w numeracji). Każdy probe używał `fim_chunk_size=256` i mierzył
`Tr(F_L)/Tr(F_R)` w e4=5/8/11/14/17. Cechą był slope regresji liniowej
`log(Tr(F_L)/Tr(F_R))` względem lokalnej epoki P4. Clean-from-scratch P2=17
tego samego seedu służył wyłącznie jako z góry ustalony target porównawczy;
downstream P4=200 nie był wejściem decyzji.

| seed | clean-gold slope | e3 najbliższe targetowi | slope | dokładne best recovery P4=200 | błąd e3 |
|---:|---:|---:|---:|---:|---:|
| 83 | -0,009922 | 77 | -0,008891 | 80 | -3 |
| 184 | +0,000359 | 54 | -0,002453 | 57 | -3 |
| 285 | +0,004075 | 69 | +0,028296 | 74 | -5 |

Wybór najbliższego slope `77/54/69` jest taki sam dla pełnego okna
5/8/11/14/17 i kontroli 5/8/11/14. Reguła online „pierwszy slope poniżej
gold” daje dla pełnego okna `78/54/70`, czyli błędy `-2/-3/-4`, ale dla seedu
83 zmienia decyzję na 77 po usunięciu punktu e4=17. Żadna z reguł nie trafia
dokładnie w oracle `80/57/74`. Zachowują poprawną kolejność długości między
seedami, lecz systematycznie kończą interwencję za wcześnie. Średni
bezwzględny błąd wyboru najbliższego targetowi wynosi 3,67 epoki, a pierwszego
przejścia — 3 epoki.

Wynik zamyka ten eksperyment jako negatywną walidację aktualnego stoppera:
slope TFIM jest użyteczny do wskazania lokalnego obszaru, ale nie jest jeszcze
samodzielnym, zwalidowanym kryterium stopu. Ewentualnego bufora po przejściu
progu nie wolno stroić na tych samych trzech seedach; wymaga on z góry
ustalonej reguły i nowych seedów albo prospective shadow validation.
Pełna tabela 18 slope'ów i podsumowanie znajdują się w storage pod
`$REPORTS_DIR/analysis/tfim_stopper_development_2026-08-03_04/tfim_gold_slope_dense_refinement_2026-08-04/`.

### Kontrole clean P2 i unimodalne

Aby ustalić punkt odniesienia dla interpretacji ratio, ukończono
50-epokową kontrolę dla seedów 83/184/285:

- bimodalny `P1=0/P2=50`: `Tr(F_L)`, `Tr(F_R)` i ich ratio;
- clean left-only: `Tr(F_L)`;
- clean right-only: `Tr(F_R)`.

Wszystkie warunki używają tego samego splitu, normalizacji, class-balanced
probe 1k, sampled-label RNG i kadencji co 5 epok. Para unimodalna odtwarza pełną inicjalizację bimodalną
`canonical_bimodal_components_v2`; każda gałąź zachowuje własne kanoniczne
wagi z tego samego początkowego modelu. Standardowy bimodalny baseline zachowuje
swoją historyczną inicjalizację. Traces dotyczą encoderów, bez BatchNorm i
shared trunk. Test jest wyłączony.

Wszystkie dziewięć runów `20362508`–`20362516` zakończyło się
`COMPLETED (0:0)`. Surowe, nieuśrednione trajektorie mają punkty w epokach
5, 10, ..., 50.

| seed | bimodal Tr(F_L), e5→e50 | bimodal Tr(F_R), e5→e50 | ratio e5→e50 | maks. ratio |
|---:|---:|---:|---:|---:|
| 83 | 44,81→59,34 | 176,57→152,66 | 0,254→0,389 | 0,424 (e35) |
| 184 | 34,28→50,42 | 134,19→191,73 | 0,255→0,263 | 0,291 (e45) |
| 285 | 44,75→87,53 | 97,21→142,02 | 0,460→0,616 | 0,655 (e45) |

W clean bimodal P2 prawa gałąź ma większy sampled TFIM we wszystkich
30 obserwowanych punktach; ratio nie przecina 1 dla żadnego seedu. Oznacza to,
że silna dominacja lewego TFIM obserwowana po interwencji Phase 3 nie jest
prostą własnością architektury ani clean training. Nie oznacza to jednak
bezpośrednio większej przyczynowej użyteczności prawej gałęzi.

| seed | left_proper Tr(F_L), e5→e50 | right_proper Tr(F_R), e5→e50 | bimodal/uni left e50 | bimodal/uni right e50 |
|---:|---:|---:|---:|---:|
| 83 | 174,61→149,84 | 124,40→178,54 | 0,396 | 0,855 |
| 184 | 135,63→145,87 | 122,82→221,88 | 0,346 | 0,864 |
| 285 | 149,66→158,64 | 174,30→233,66 | 0,552 | 0,608 |

Raw TFIM zależy więc silnie od kontekstu funkcji celu: szczególnie lewa gałąź
ma znacznie mniejszy trace w modelu bimodalnym niż w odpowiadającym modelu
unimodalnym. Nie należy normalizować lub porównywać TFIM między kontekstami
tak, jakby był niezmienną właściwością encodera.

Wybrane na validation proper referencje są zbalansowane pod względem accuracy:

| seed | left accuracy (epoka) | right accuracy (epoka) |
|---:|---:|---:|
| 83 | 0,8312 (50) | 0,8260 (45) |
| 184 | 0,8318 (50) | 0,8380 (50) |
| 285 | 0,8420 (50) | 0,8418 (35) |

Średnio left/right accuracy wynosi 0,8350/0,8353. Walidator pełnych par
`20362599` potwierdził policy `canonical_bimodal_components_v2` i wspólny
hash źródła wewnątrz każdego seedu: seed 83
`f5aa5f4553a4422abce0aafbb01ba33bb364101ad4887d27d338f9d7d58fa97e`,
seed 184
`c274b4eaf49a419545715f8bb5cad43aa77369f440928d1179bbba65dcd0d33c`,
seed 285
`f0a66dd5eca9dfe995b6f4887a3a32e53dc2c7e2f0c428743e04fde3f60aa52b`.

Endpoint clean bimodal P2=50 osiągnął validation full/dominant-only/weak-only:
seed 83 `0,8738/0,5058/0,7080`, seed 184
`0,8650/0,3290/0,7166`, seed 285 `0,8596/0,5268/0,6700`; średnio
`0,8661/0,4539/0,6982`.

Artefakty:

- `analysis/results/tfim_clean_controls_50_v2/tfim_clean_controls_by_seed.png`;
- `analysis/results/tfim_clean_controls_50_v2/tfim_clean_controls_branch_context.png`;
- osobne `tfim_clean_controls_seed{83,184,285}.png`;
- surowe dane `tfim_clean_controls_raw.csv` i podsumowanie
  `tfim_clean_controls_summary.json`.

Wykresy nie uśredniają seedów ani nie wygładzają trajektorii. Ograniczenia:
sampled TFIM pochodzi z jednego ustalonego probe i sampled-label RNG, ma tylko
trzy seedy i nie ma przedziałów niepewności. Clean bimodal baseline zachowuje
historyczną ścieżkę inicjalizacji, natomiast hash-pairing v2 dotyczy par
unimodalnych; raw magnitudes tych kontekstów są kontrolą diagnostyczną, a nie
paired causal contrast.

## Stan decyzji

- Primary protocol: historyczne `deactivation`, right + shared trainable.
- Primary target: per-seed downstream oracle długości P3.
- Frozen-left-active: kontrola negatywna.
- Relative parity/frozen-shared: osobna metodologia, zbyt długa interwencja.
- Right-only UMT: kontrola z katastrofalnym compatibility drift.
- Full UMT: ukończony eksperyment alternatywnej interwencji, oczekuje analizy.
- TFIM: diagnostyka lokalizująca obszar; gęsty refinement wykazał bias
  stoppera opartego wyłącznie na gold slope w stronę zbyt krótkiej P3.

Oracle P4=200 i gęsty refinement TFIM są zakończone. Dalsza praca nad
automatycznym stopperem wymaga nowych seedów lub zamrożonej prospective shadow
validation; nie należy dalej dostrajać reguły na oracle 80/57/74.
