# Bieżący status

Ostatnia aktualizacja: 2026-08-03.

## Aktualny stan

- Najwyższy priorytet to per-seed best recovery długości Phase 3 dla
  `P1=120/P2=200` i historycznej trainability:
  trainability Phase 3: lewa gałąź jest deaktywowana, a prawy encoder oraz
  shared trunk/classifier pozostają trenowalne. P3 jest pełną trajektorią
  observe-only do 200 epok; P4 i test są wyłączone.
- Coarse oracle P4=40 wybrał e3=60/40/60 dla seedów 83/184/285.
  Retrospektywny TFIM probe wskazuje slope log-ratio jako lepszy predyktor niż
  endpoint/EMA; wynik LOOSO jest obiecujący, ale eksploracyjny.
- True clean gold standard ma `P1=0/P2=200/P3=0/P4=0`; `P1=1/P2=200` jest kontrolą minimalnej jednoepokowej ekspozycji.
- P2=200 jest accuracy gold. Osobny TFIM dynamics gold ma start od losowej
  inicjalizacji i wyłącznie clean P2=17 z pomiarami e2=5/8/11/14/17.
- Trzy ślady P1=40 zakończyły się rollbackiem `e3=0`, ponieważ stara bramka compatibility odrzuciła wszystkie 120 checkpointów Phase 3.
- `local_accuracy` pozostaje zachowanym wariantem eksperymentalnym. Aktywnie
  wdrażany `relative_unimodal_parity` normalizuje obie gałęzie przez ich
  seed-paired clean unimodal reference i zamraża w Phase 3 cały model poza
  prawym encoderem.
- Punktowy replay wskazał stop/selection: seed 83 `40/40`, seed 184
  `60/50`, seed 285 `30/30`.
- Dokładne shadow z paired CI wybrały e3=30/35/30 dla seedów 83/184/285.
  Sweep P4 wskazał najlepsze milestone 60/40/40, a średnio najlepsze e3=40.
- Pełna analiza i wykres są w
  `docs/PHASE3_LOCAL_STOPPER_ANALYSIS_2026-08-01.md`.
- Nowy wariant i jego referencje trafiają do projektu
  `bartekk/CLPIntervention_UnimodalParity`.
- Exact relative parity okazało się zbyt późne. Zaimplementowany
  `recovery_fraction_threshold` zachowuje `1.0` jako exact parity, a replay
  porównuje globalne progi 90% i 95%.
- Aktualny frozen-shared sweep e3=20/40/60/80/100/120/140/160 daje mean
  validation full 0,8833/0,8870/0,8898/0,8921/0,8941/0,8957/0,8984/0,8981.
  Trajektoria osiąga plateau około e3=140; nie ma historycznego załamania
  dominant-only.
- Hybrydowa diagnostyka P4 wskazuje shared trunk/classifier jako główne źródło
  natychmiastowego weak-only collapse. Warm-up pomaga w e1, ale nie zachowuje
  przewagi do e5–10.
- Clean gold `P1=0/P2=200` nie ma naturalnie słabszej prawej gałęzi: mean
  validation weak-only `0,7487` przewyższa dominant-only `0,5648` o 18,39 pp.
- Dodano opt-in czteroepokowy prefix P4 uczący wyłącznie shared downstream;
  oba encodery pozostają wtedy aktywne w forward, ale frozen/eval.
- Replay sparowanych CI 99%/100% po korekcie 54 spojrzeń nie potwierdził
  crossingów; oba progi wybrały fallbacki e3=188/176/156.
- Dodano opt-in Phase-4 loss `L_full + L_weak` z `lambda_L=0`; pierwszy test
  izoluje funkcję celu bez staged unfreezing. Do e4<=10 daje mean
  full/dominant/weak `0,8682/0,8001/0,7053`.
- Post-hoc equal-weight UME daje mean validation/test `0,9077/0,9001`, wobec
  clean gold `0,8979/0,8893`. Mean logits `0,9166/0,9073` jest analizą
  czułości, nie wynikiem głównym.
- UME rozszerzone o gold bimodalny, jako ustalona średnia trzech
  prawdopodobieństw, daje mean validation/test `0,9183/0,9100`.
- Shared-trunk P1=120 po P4=200 ma mean full accuracy dla
  e3=0/20/40/60/80: `0,8755/0,8832/0,8865/0,8847/0,8771`; w zbadanej
  wspólnej siatce maksimum przypada na e3=40.
- Historyczna kontrola right-only UMT Phase 3 silnie poprawia weak-only, ale
  nie chroni compatibility; pełny UMT jest osobnym eksperymentem:
  mean validation e0→e200 full `0,8735→0,6072`, dominant
  `0,8183→0,3655`, weak `0,2251→0,8432`.

## Joby bieżącej analizy

- Stara seria nie jest źródłem wyników: `20362098` i `20362101` ukończyły
  się, ale używały błędnej polityki `canonical_left_encoder`; `20362099`
  zakończył się statusem `FAILED`, a `20362102`–`20362112` anulowano.
- Nowa seria v2: GPU gate `20362364` `COMPLETED`, `1 passed`; smoke
  `20362375` (P2=2), `20362376` (left) i `20362377` (right), wszystkie
  `COMPLETED`. Walidacja metadanych `20362496` `COMPLETED`: version 2,
  policy `canonical_bimodal_components_v2`, wspólny hash
  `f5aa5f4553a4422abce0aafbb01ba33bb364101ad4887d27d338f9d7d58fa97e`.
  Smoke gate `20362506` także `COMPLETED`.
- Po `afterok:20362506` uruchomiono dziewięć pełnych jobów 50-epokowych:
  P2: seed83=`20362516`, seed184=`20362509`, seed285=`20362511`; left:
  seed83=`20362515`, seed184=`20362512`, seed285=`20362510`; right:
  seed83=`20362508`, seed184=`20362513`, seed285=`20362514`. Wszystkie
  zakończyły się `COMPLETED (0:0)`; TFIM co 5 epok,
  `fim_chunk_size=256`, test wyłączony, projekt W&B
  `CLPIntervention_TFIM_CleanControls_50`.
  Walidacja pełnych par `20362599` także zakończyła się `COMPLETED`;
  wykresy i surowe dane są w
  `analysis/results/tfim_clean_controls_50_v2/`.

- `20361984`–`20362004` z przerwami w numeracji: 18 runów coarse oracle,
  `e3=0/20/40/60/80/100 × seed=83/184/285`, każdy P4=40, validation i TFIM
  co 5 epok, chunk 256, test wyłączony; W&B
  `CLPIntervention_P3Oracle_TFIM_P4_40`; wszystkie `COMPLETED (0:0)`.

- `20361540`: frozen-left-active dwuepokowy GPU smoke, `COMPLETED (0:0)`;
  lewa gałąź była aktywna, a dominant-only pozostało `0,888→0,888`.
- `20361565`–`20361567`: pełne frozen-left-active P3=200 dla seedów
  184/83/285; W&B `CLPIntervention_FrozenLeftActive_P1_120`.
- `20361560`–`20361562`: ukończone kontrolne P4=200 z shared-trunk e3=40;
  nie są ważnymi pomiarami TFIM, ponieważ profil standalone P4 blokował moduł.
- `20361883`: GPU smoke pełnego UMT (dwa nauczyciele, obie gałęzie aktywne,
  cały student trenowalny), `COMPLETED (0:0)`; oba składniki MSE były liczone.
- `20361885`, `20361888`, `20361889`: pełne UMT P3=200 dla seedów
  83/184/285; wszystkie `COMPLETED (0:0)`; W&B
  `CLPIntervention_FullUMT_P1_120`.
- `20361878`–`20361880`: poprawione P4=200 z TFIM co 10 epok,
  `fim_chunk_size=256`, seedy 83/184/285, wszystkie `COMPLETED (0:0)`;
  projekt W&B
  `CLPIntervention_SharedTrunk_P1_120_TFIM`.
- `20361501`: post-hoc UMT validation/test co 20 epok i dwa wykresy,
  `COMPLETED (0:0)`.
- `20360975`–`20360991`: 17 P4=200 z P1=120 shared-trunk milestone i
  wyborów stoppera, wszystkie `COMPLETED (0:0)`.
- `20359797`, `20359798`, `20359799`: pełne trajektorie
  `P1=120/P2=200/P3=200/P4=0`, seedy 83/184/285, shared trunk odmrożony,
  wszystkie `COMPLETED (0:0)` w 1:05–1:07, W&B
  `CLPIntervention_SharedTrunk_P1_120`.
- `20357242`–`20357247`: P4=200 dla milestone e3=140/160, trzy seedy,
  wszystkie `COMPLETED (0:0)`.
- `20357275`–`20357277`: P4=200 z trzech unikalnych fallbacków CI
  99%/100% e3=188/176/156, wszystkie `COMPLETED (0:0)`.
- `20357290`: pierwszy compute test asymetrycznego lossu, `COMPLETED`, ale
  ujawnił, że prototyp wykonywał encoder weak dwa razy.
- `20357305`, `20357307`, `20357308`: anulowane po 0:48/0:48/0:00; nie są
  prawidłowymi runami naukowymi z powodu podwójnego forwardu encodera.
- `20357316`: poprawiony compute gate cache’owania `h_L/h_R`, `COMPLETED`,
  4 passed.
- `20357331`, `20357332`, `20357334`: poprawione trzyseedowe e4<=10 dla
  `L_full+L_weak`, pojedyncze kodowanie gałęzi, dwa przebiegi shared trunk,
  wszystkie `COMPLETED (0:0)`.
- `20358622`, `20359333`, `20359334`: post-hoc UME validation/test dla
  seedów 83/184/285, wszystkie `COMPLETED (0:0)` i zsynchronizowane z W&B.
- `20359340`: compute test definicji UME.

- `20355172`, `20355173`, `20355175`–`20355178`: sześć P4=200 z aktualnych
  frozen-shared milestone e3=100/120, wszystkie `COMPLETED (0:0)`.
- `20355913`: compute gate staged unfreezing, `COMPLETED (0:0)`, 3 passed.
- `20355917`–`20355919`: `COMPLETED`, ale nieważne jako staged control;
  sekcja nie dotarła do trenera i runy zreplikowały zwykły warm-up.
- `20355956`–`20355958`: poprawne e3=40 `shared_only=4 -> full`, wszystkie
  `COMPLETED (0:0)`; prefix ogranicza, ale nie usuwa weak-only collapse.

- `20331013`: rzeczywisty replay kontrolera 90%/95%, `COMPLETED (0:0)`.
  Stop/selection: 90% — `40/36`, `28/24`, `40/36`; 95% — `60/56`,
  `40/36`, `80/76` dla seedów 83/184/285.
- `20331032`–`20331037`: sześć P4=200 z checkpointów wybranych przez 90%
  i 95%, test wyłączony, W&B `CLPIntervention_UnimodalParity`.
- `20331441`: test GPU niezmienności left/shared i aktualizacji wyłącznie
  right encoder po rozszerzeniu stoppera, `COMPLETED (0:0)`, 1 passed.
- `20336049`–`20336060`: 12 aktualnych frozen-shared milestone P4=200,
  wszystkie `COMPLETED (0:0)`, test wyłączony.
- `20354203`: test GPU diagnostyki hybrydowej i per-step warm-up P4,
  `COMPLETED (0:0)`, 4 passed.
- `20354218`–`20354223`: sześć e3=40 diagnostyk warm-up 0/4,
  `COMPLETED (0:0)`; surowe punkty w W&B i lokalnych JSONL.

- `20285723`: testy GPU zamrażania parity i sparowanej inicjalizacji,
  `COMPLETED (0:0)`, 2 testy przeszły.
- `20285728`, `20285729`: poprawione smoke right/left unimodal reference,
  `COMPLETED (0:0)`; oba wybrały checkpoint z epoki 2.
- `20285734`: pierwsza próba czterofazowego smoke `FAILED (1:0)` przed
  Phase 2, ponieważ skrócony limit P2 wymaga jawnego `protocol_smoke=true`.
- `20285747`: poprawiony czterofazowy smoke relative parity,
  `COMPLETED (0:0)` w 2:24. Zweryfikował manifesty referencji, skip P3 przy
  parity spełnionym w e3=0, odtworzenie checkpointu oraz final-only test P4.
- `20285762`–`20285767`: anulowane po wykryciu błędu rolling bufferu, który
  przy niepoprawiającym pomiarze mógł usunąć wcześniejszy best checkpoint.
- `20285790`–`20285795`: poprawione pełne referencje left/right dla seedów
  83/184/285, wszystkie `COMPLETED (0:0)`. Validation accuracy left/right:
  seed 83 `0,8594/0,8506`, seed 184 `0,8644/0,8556`, seed 285
  `0,8612/0,8564`.
- `20290229`, `20290227`, `20290228`: `FAILED (1:0)` na pierwszym kroku P3
  po poprawnym ukończeniu P1/P2. RunStats nie obsługiwał zamrożonych grup;
  wybrane checkpointy P2 e165/e195/e200 zostały zachowane.
- `20290381`: ukierunkowany test GPU poprawki RunStats, `COMPLETED (0:0)`,
  1 passed. Zamrożone grupy raportują zerowe gradienty, a tryby modułów są
  dokładnie odtwarzane.
- `20290388`, `20290387`, `20290386`: standalone Phase-3 `observe_only` do
  e3=200 dla seedów 83/184/285. Używają zachowanych checkpointów P2, zapisują
  `phase3_trajectory.jsonl` i wszystkie 53 checkpointy pomiarowe; P1/P2 nie
  są ponownie trenowane, P4 jest wyłączona.
- `20285726`, `20285727`: pierwsze smoke referencji `FAILED (1:0)`; ujawniły
  niekompatybilną historyczną diagnostykę trajektorii zamrożonej gałęzi,
  wyłączoną następnie wyłącznie dla nowego wariantu.
- `20282574`: odtworzenie Phase 3 seedu 184 do e3=30, `COMPLETED (0:0)` w
  5:46; W&B `absavsy8`.
- `20282582`: kontrolny P4=200 seedu 184 z e3=30, `COMPLETED (0:0)` w 28:18;
  validation/test proper 0,8836/0,8728, W&B `7org63if`.
- `20282557`: kontrolny P4 nie rozpoczął treningu (`FAILED 1:0`), ponieważ
  tymczasowy checkpoint e3=30 został wcześniej usunięty przez rolling buffer.
- `20282321`: P4=200 z dokładnego wyboru e3=30, seed 83,
  `COMPLETED (0:0)` w 29:04.
- `20282322`: P4=200 z dokładnego wyboru e3=35, seed 184,
  `COMPLETED (0:0)` w 28:08.
- `20282324`: P4=200 z dokładnego wyboru e3=30, seed 285,
  `COMPLETED (0:0)` w 28:18.
- `20281023`: dokładny paired-CI Phase-3 shadow, seed 184,
  `COMPLETED (0:0)` w 12:01, wybrano e3=35, W&B `rxmj419y`.
- `20281024`: dokładny paired-CI Phase-3 shadow, seed 285,
  `COMPLETED (0:0)` w 12:34, wybrano e3=30, W&B `9ghnidgh`.
- `20273546`: compute test gradient probe, `COMPLETED (0:0)`, 1 passed.
- `20273600`: czterofazowy full-model local-accuracy shadow smoke,
  `COMPLETED (0:0)` w 2:29; hipotetyczny target w e3=5.
- `20273663`: dokładny standalone Phase-3 local-accuracy shadow, seed 83,
  `COMPLETED (0:0)` w 11:56, paired-CI target e3=30, W&B `pr4byikq`.
- P4 milestone P1=40, seed 83: `20273138` e20, `20273152` e40,
  `20273143` e60, `20273146` e80, `20273140` e200 — wszystkie ukończone.
- P4 milestone P1=40, seed 184: `20273145` e20, `20273141` e40,
  `20273147` e60, `20273153` e80, `20273139` e200 — wszystkie ukończone.
- P4 milestone P1=40, seed 285: `20273137` e20, `20273148` e40,
  `20273149` e60, `20273142` e80, `20273150` e200 — wszystkie ukończone.
- `20258725`: P1=40 stopper observe/replay, seed 83, limit 2 h.
- `20258727`: P1=40 stopper observe/replay, seed 184, limit 2 h.
- `20258726`: P1=40 stopper observe/replay, seed 285, limit 2 h.
- `20258291`: finalny czterofazowy smoke schedulera wyłącznie per-step,
  `COMPLETED (0:0)` w 2:00.
- `20257703`, `20257704`, `20257705`: anulowane po zmianie specyfikacji
  warm-upu z epokowego na czteroepokowy per-step.
- `20257994`: compute test nowego czteroepokowego schedulera per-step,
  `COMPLETED (0:0)`, 2 testy przeszły.
- `20258142`: wcześniejszy smoke pośredniej wersji, `COMPLETED (0:0)`.
- `20257602` i `20257606`: zakończone testy poprzedniej wersji epokowej;
  nie stanowią bramki akceptacyjnej dla obecnej implementacji per-step.
- `20153336`, `20153344`, `20153350`–`20153355`: zakończone `COMPLETED (0:0)`.
- Poprzednie `20152478`, `20152529`–`20152534` anulowano przed startem (`0:00`).

## Ostatnie decyzje

- Relative parity używa pary referencji na seed, best validation accuracy po
  maksymalnie 200 epokach, dwóch potwierdzeń z wyborem pierwszego checkpointu
  serii oraz fallbacku best weak ratio.
- Następne runy lokalnego stoppera mierzą e3=1,2,3,4, a potem co 4 epoki;
  minimalna ekspozycja i okno trendu obejmują cztery pomiary.
- Samo P4=200 z endpointu P3=200 nie testowałoby stoppera. Phase 4 musi startować z jego zamrożonej rekomendacji.
- Trafność stoppera oceniamy na validation proper, porównując jego P4 z osobnymi P4 od milestone 40/60/80/200; test nie wybiera zwycięzcy.
- Poprzedni sweep etapu 1 dotyczył `P1=80/P2=200`, więc nie jest dowodem trafności dla P1=40.
- Gold zachowuje `P4=0`, ponieważ mierzy czyste P2; P4=200 dotyczy eksperymentów interwencyjnych P1=40.
- Stopper nie używa już rosnącego weak-only loss jako bramki plateau/reversal. Compatibility drift w Phase 3 nie powoduje rollbacku recovery checkpointu.
- Bezwzględny spadek full/dominant względem początku P3 nie jest już
  stopperem; wszystkie trzy accuracy są chronione lokalnie.
- Warm-up LR jest minimalną stabilizacją; zamrożenie wspólnego trzonu pozostaje osobną przyszłą ablacją.

## Następne kroki

1. Uruchomić wspólne P4=200 z checkpointów e3=20/40/60/80 dla trzech seedów.
   P3 pokazuje szybki compatibility shock i minimum weak loss około e16–24;
   shadow local-accuracy wybrał e3=32/44/20, lecz jego trafność można ocenić
   dopiero po P4. Test pozostaje wyłączony.
2. Sweep `lambda_R` dla `L_full + lambda_R L_weak` jest odłożony. Jeżeli do
   niego wrócimy, należy użyć pełnej P4=200 z `e3=140` albo z zamrożonego
   wyboru stoppera, a nie krótkiego diagnostycznego e3=40.
3. Rozstrzygnąć, czy celem Phase 4 jest wyłącznie full accuracy, czy również
   trwała samodzielna użyteczność weak branch; obecny full-only loss nie
   chroni tej drugiej własności.
4. Przed zamrożeniem stoppera rozszerzyć aktualny frozen-shared sweep co
   najmniej o e3=100/120 albo uzasadnić a priori ograniczenie do e3<=80,
   ponieważ full accuracy nadal rosła do końca sprawdzonego zakresu.
5. Jeśli weak-only ma być chronione, porównać mniejszy LR dla shared
   downstream albo jawny pomocniczy loss unimodalny. Sam warm-up nie wystarcza.
6. Test wykorzystać dopiero po zamrożeniu wariantu treningu i reguły stopu.

## Blokery

- Kontrola e3=30 czeka na regenerację checkpointu; nie blokuje pozostałej
  analizy.
- Historycznie ujawniony klucz W&B musi zostać unieważniony poza repozytorium.

## TFIM Phase-3 oracle refinement — 2026-08-03

- Ranking best recovery używa teraz pasa równoważności full-gap
  `minimum + 0,01`; wewnątrz pasa rozstrzyga średnia luka
  dominant-only/weak-only. Po pełnym skanie e3=78..82 seed 83 nadal wybiera
  e3=80. Po pełnym skanie e3=50..60 seed 184 wybiera e3=57.
- P4=200 dla brakujących punktów seed 83 zakończyły joby `20388014`–
  `20388015`; skan seed 184 uruchomiono jako `20388156`–`20388163`, a skan
  seed 285 jako `20388720`–`20388725`.
- Test wznowienia P3 e54→e59 `20388166` nie odtworzył nieprzerwanej
  trajektorii: full validation e59 wyniosło 0,6824 zamiast 0,6892, a
  dominant-only 0,5932 zamiast 0,6200. Po przeniesieniu odtworzenia RNG za
  inicjalizację loggera uruchomiono kontrolę `20388773`. Job zakończył się
  `COMPLETED (0:0)`, ale nadal nie odtworzył nieprzerwanej trajektorii:
  endpoint full accuracy był równy `0,6892`, natomiast dominant-only/weak-only
  wyniosły `0,6058/0,7818` zamiast `0,6200/0,7832`, a full loss
  `2,304526` zamiast `2,232441`. Sama kolejność odtworzenia RNG nie wystarcza;
  resume nadal nie wolno używać do materializacji publikacyjnej.

- Pełne P4=200 dla seedu 184 i e3=47 (`20384859`) zakończyło się
  `COMPLETED (0:0)`. Najlepsze validation proper full accuracy wyniosło
  `0,8842` w e4=185; dominant-only `0,7652`, weak-only `0,4390`.
- W aktualnie sprawdzonych punktach oracle validation wybiera e3=80/47/70
  odpowiednio dla seedów 83/184/285. Dla seedu 184 przedział został
  zawężony do `[47, 50]`.
- Materializacje P3 `20384893` (seed 83: e3=78/82) i `20384894`
  (seed 285: e3=68/72) zakończyły się poprawnie.
- Pełne P4=200 dla tych checkpointów uruchomiono jako `20386248`–`20386251`.
  Wszystkie zakończyły się `COMPLETED (0:0)`. Hierarchiczny ranking względem
  clean accuracy gold nadal wybiera e3=80 dla seedu 83 i e3=70 dla seedu 285.
- Finalny refinement do jednej epoki wymaga e3=79/81, 45/46/48/49 oraz
  69/71. Materializacje tych checkpointów uruchomiono jako
  `20387410`–`20387412`.
- Poprawny TFIM dynamics gold jest clean-from-scratch P2=17. Smoke P2=2 ma
  job `20387413`; pełne seedy `20387443`–`20387445` zależą od niego przez
  `afterok`.
- Pierwszy poziom retrospektywnego stoppera opartego na nachyleniu log TFIM
  nie generalizuje LOOSO: dla seedów 184 i 285 wskazał przedziały
  niezawierające oracle. Nie uruchomiono post-hoc drugiego poziomu dla tej
  odrzuconej reguły.
- Pairwise replay po dokładnym refinement dla seedów 83/184 poprawił wynik:
  pełne okno TFIM e4=5/8/11/14/17 wskazuje e3=79/58 wobec best recovery
  e3=80/57, a oba przedziały zawierają referencję. Okno 5/8/11/14 wskazuje
  e3=78/59 i oba przedziały mijają referencję. Wynik opiera się tylko na
  wzajemnej kalibracji dwóch seedów i nie unieważnia negatywnego
  trzyseedowego LOOSO.
- Po ukończeniu `20388720`–`20388725` dokładny ranking wybiera seed 285
  e3=74; komplet best recovery wynosi 80/57/74. Ponowiony trzyseedowy LOOSO
  pełnego okna wskazuje 81/59/73. Przedziały zawierają optimum dla seedów
  83 i 285, ale nie dla 184 (`[58,59]` wobec e3=57), więc stopper nadal nie
  przechodzi pełnej walidacji retrospektywnej.

## Końcowy refinement TFIM gold-slope — 2026-08-04

- Compute smoke materializacji checkpointów `20393538` zakończył się
  `COMPLETED (0:0)`. Trzy nieprzerwane trajektorie P3 `20393540`–`20393542`
  zapisały bez ewaluacji wszystkie checkpointy odpowiednio e3=75..80,
  50..55 i 65..70.
- Wszystkie 18 niezależnych probe'ów P4=17 (`20394087`–`20394126`, z
  przerwami) zakończyło się `COMPLETED (0:0)`. TFIM mierzono w
  e4=5/8/11/14/17 z `fim_chunk_size=256`.
- Najmniejsza odległość pięciopunktowego slope `log(Tr(F_L)/Tr(F_R))` od
  per-seed clean P2=17 gold wybiera e3=77/54/69. Kontrola czteropunktowa
  5/8/11/14 wybiera te same epoki.
- Niezależny downstream oracle P4=200 wynosi 80/57/74. Stopper myli się o
  -3/-3/-5 epok i ma MAE 3,67. Pierwsze przejście poniżej gold daje
  78/54/70 i MAE 3, ale jest niestabilne dla seedu 83 po usunięciu e4=17.
- Eksperyment jest zakończony. Aktualny gold-slope stopper lokalizuje obszar,
  lecz systematycznie zatrzymuje P3 za wcześnie i nie przechodzi walidacji
  jako gotowa reguła online. Dalszego bufora nie wolno kalibrować na tych
  samych trzech seedach.
