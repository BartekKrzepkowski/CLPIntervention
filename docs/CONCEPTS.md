# Dziennik pojęć CLPIntervention

Dokument utrwala znaczenie terminów używanych w kodzie, artykule i raportach.

## Pojęcia eksperymentalne

- **Critical learning period / okres krytyczny** — przedział treningu, w którym wcześniejsza przewaga jednej modalności ogranicza późniejszą plastyczność drugiej.
- **Modalność / pole widzenia** — lewa albo prawa część tego samego obrazu, przetwarzana przez osobną gałąź.
- **FIM / ślad Fisher Information Matrix** — miara lokalnej wrażliwości rozkładu predykcji na parametry danej gałęzi. W repozytorium estymowana z etykiet próbkowanych z modelu.
- **SV / source variance** — wariancja reprezentacji wywołana zmianami tylko jednego źródła/modalności.
- **RSV / relative source variance** — znormalizowana różnica wariancji źródeł. Repozytorium używa wyłącznie konwencji Kleinmana `(SV_left-SV_right)/(SV_left+SV_right)`: `+1=lewa`, `-1=prawa`. Historyczne wartości o przeciwnym znaku należy zanegować.
- **stage3_avgpool** — RSV kanałów wyjścia `main_branch.0` po dodatkowym poolingu przestrzennym wykonywanym wyłącznie przez analizę post hoc.
- **stage4_avgpool** — RSV natywnego wyjścia `avgpool` po `main_branch.1`, bezpośrednio przed klasyfikatorem; punkt zgodny z Kleinmanem.
- **Długość trajektorii parametrów** — suma norm rzeczywistych wektorów przesunięcia po kolejnych `optimizer.step()`. Różni się od odległości od początku, która jest pojedynczą normą wektora między dwoma stanami, i pozostaje poprawna przy weight decay oraz momentum.
- **Seed runu a generator** — seed jest zapisaną liczbą definiującą jeden replikat eksperymentu; generator jest stanowym źródłem losowości zainicjalizowanym tym seedem. Lokalne generatory izolują FIM, RSV i bootstrap, ale nie zastępują identyfikatora seedu runu.
- **Hierarchiczny bootstrap sparowanych różnic** — resampling dopasowanych par tego samego seedu/modelu i tych samych obrazów; jednostki są najpierw agregowane w obrazie, następnie resamplowane są obrazy i modele, a statystyką jest `intervention-control`.
- **EWMA-10** — wykładnicza średnia ruchoma używana wyłącznie do wizualnego wygładzenia surowego przebiegu. Nie jest elementem treningu ani zapisywanej metryki.
- **Bisekcja mediany RSV** — wyszukiwanie binarne długości interwencji: kolejne długości zawężają przedział tak, aby mediana końcowego RSV zbliżyła się do zera (w pracy próg `|mediana| <= 0.15`). Nie oznacza podziału obserwacji poniżej/powyżej mediany.
- **Korelacja** — współzmienność dwóch wielkości, np. RSV po fazie 1 i przyrostu dokładności prawej gałęzi. Nie dowodzi przyczynowości.

## Trening i infrastruktura

- **Learning rate (LR)** — wielkość kroku aktualizacji parametrów.
- **Mnożnik LR** — czynnik stosowany przez scheduler; `0.98` oznacza pomnożenie LR przez `0.98` przy każdym kroku schedulera.
- **Weight decay** — kara ograniczająca wzrost wag; w użytym SGD jest realizowana jako składnik proporcjonalny do parametrów.
- **Checkpoint treningowy** — wersjonowany pakiet z modelem, optimizerem, schedulerem, następną epoką, krokiem globalnym i stanami RNG. Pakiet może zostać wczytany weights-only przy transferze fazy albo w całości przy wznowieniu runu.
- **Resume checkpoint** — okresowo nadpisywany pełny checkpoint do kontynuacji przerwanego runu; nie jest tym samym co checkpoint wybrany naukowo ani weights-only `model_checkpoint`.
- **Smoke run** — krótki test całego przepływu na prawdziwych danych, którego celem jest wykrycie błędów integracyjnych, a nie uzyskanie wyniku naukowego.
- **Statystyki normalizacji** — średnie i odchylenia standardowe kanałów liczone na odpowiedniej domenie treningowej. Faza 1 normalizuje rozmyte prawe pole jego własnymi statystykami; po udostępnieniu poprawnych danych stosowane są ich nowe statystyki. Wartości zależą od geometrii pola, overlapu, resize factor, interpolacji i wersji biblioteki.
- **Rekalibracja BatchNorm** — przeliczenie wyłącznie buforów statystyk BN na ustalonych danych, bez gradientów i zmiany wag. W kontroli fazy 4 domyślnym zakresem jest wspólny `main_branch`.
- **Manifest runu** — lokalny, wersjonowany opis pochodzenia wyniku: commit, konfiguracja, seed, dataset/subset, checkpointy, środowisko i pliki wynikowe.
- **Deselected** — test znaleziony przez pytest, ale celowo niewykonany przez filtr markerów.
- **RSS / Resident Set Size** — ilość fizycznej pamięci RAM zajmowanej przez proces; nie obejmuje pamięci GPU.
- **Validation proper** — stały zbiór 5k z oryginalnego train CIFAR-10 używany wyłącznie do decyzji online i wyboru checkpointów. Nie jest testem.
- **Clean gold standard** — kontrola bez ekspozycji na rozmytą prawą modalność: `P1=0, P2=200, P3=0, P4=0`. `P1=1` nie jest gold standardem, lecz kontrolą minimalnej jednoepokowej ekspozycji.
- **Phase-2 post-hoc test** — końcowa diagnostyka zachowanych checkpointów minimum-loss i maksimum-accuracy Phase 2. Jest wykonywana dopiero po wszystkich fazach, nie jest wejściem żadnej decyzji i po pomiarze przywraca finalny checkpoint runu.
- **Weak utility loss** — `dominant_only_loss - full_loss`; marginalna redukcja straty po dołączeniu słabszej modalności.
- **Plateau Phase 2** — jednoczesna stabilizacja full loss, weak-only loss i weak utility, a nie zwykły brak poprawy pełnego modelu.
- **Feasible checkpoint Phase 3** — checkpoint spełniający minimalne zyski słabej gałęzi oraz constraints zachowania full i dominant-only performance.
- **Safe checkpoint Phase 3** — checkpoint spełniający constraints full/dominant, nawet jeśli nie osiąga minimalnych zysków słabej gałęzi; pierwszy fallback przed rollbackiem.
- **Compatibility drift** — utrata zgodności zamrożonego dominującego encodera ze zmienionym wspólnym trzonem i klasyfikatorem, mierzona po reaktywacji dominant-only.
- **PAIS (Paired Adaptive Intervention Stop)** — lokalna reguła długości Phase 3 łącząca sparowane przedziały walidacyjne, wykrycie odwrócenia trendu, futility i hard safety. Działa na jednym runie i nie używa testu.
- **Weak-recovery PAIS v3** — eksperymentalna lokalna, validation-only reguła Phase 3 oparta przede wszystkim na sparowanej per-image weak-only accuracy. Zatrzymuje po potwierdzonym accuracy plateau lub reversal; emergency stop reaguje tylko na NaN/Inf. Loss oraz compatibility drift są diagnostyką i tie-breakerami, a nie bramkami recovery.
- **Local-accuracy stopper v4** — validation-only reguła bez absolutnych progów spadku względem początku Phase 3. Kończy interwencję po statystycznym dogonieniu dominant przez weak, potwierdzonym odwróceniu Pareto albo plateau weak współwystępującym z lokalną szkodą accuracy lub konfliktem gradientów wspólnego trzonu.
- **Pareto reversal** — sytuacja, w której wcześniejszy checkpoint jest co najmniej równie dobry w weak-only, full i dominant-only accuracy oraz wiarygodnie lepszy w co najmniej jednym z tych wymiarów.
- **Gradient conflict** — ujemny cosinus pomiędzy gradientami wspólnego trzonu liczonymi dla weak-only i dominant-only albo full na stałym `validation_proper` probe. Jest wyłącznie potwierdzeniem futility; bez plateau weak nie zatrzymuje interwencji.
- **Recovery checkpoint** — kandydat ze statystycznie potwierdzoną poprawą weak-only accuracy względem stanu sprzed Phase 3. `safe` opisuje niezależnie zgodność reaktywowanego full/dominant modelu i pozostaje diagnostyką; nie eliminuje kandydata recovery przed Phase 4.
- **Overconfidence** — nadmierna pewność rozkładu predykcji, szczególnie dla błędnych klas. Może zwiększać NLL mimo rosnącej accuracy; należy ją diagnozować NLL, Brier score, ECE oraz confidence błędnych predykcji, a nie utożsamiać z samą liczbą błędów.
- **Podwójna selekcja checkpointów** — równoległa retencja minimum validation loss oraz maksimum validation accuracy. `primary_metric` określa wagi przekazywane dalej, ale drugi checkpoint pozostaje dostępny do jawnego porównania. Nie jest to strojenie na teście.
- **Niesmoothowany NLL** — ujemny logarytm prawdopodobieństwa prawdziwej klasy liczony bez label smoothing i wag klas; służy do rozpoznania overconfidence niezależnie od kryterium optymalizowanego w treningu.
- **Brier score / ECE** — diagnostyki kalibracji prawdopodobieństw. Multiclass Brier score mierzy średni kwadratowy błąd całego wektora prawdopodobieństw, a ECE porównuje confidence z accuracy w 15 przedziałach. Nie sterują obecnie stopperem.
- **Recovery plateau** — kolejne walidacje, w których górna granica lokalnego nachylenia weak-only accuracy nie przekracza tolerancji dalszego praktycznego zysku. Trend loss pozostaje logowany, lecz nie jest bramką stopu w trybie accuracy-first.
- **Phase-3 LR warm-up** — opcjonalne liniowe zwiększanie LR od skonfigurowanego ułamka do bazowego LR na początku interwencji, wykonywane po każdym kroku optymalizatora. Ogranicza gwałtowny pierwszy ruch wspólnego trzonu po deaktywacji dominant branch; nie zamraża parametrów i jest checkpointowalną częścią schedulera.
- **Minimal exposure** — minimalna liczba epok i pomiarów, które muszą upłynąć przed reversal lub futility; hard safety pozostaje niezależne.
- **Futility stop** — zakończenie interwencji przed znalezieniem feasible checkpointu, gdy nawet optymistyczna granica aktualnego zysku i krótkiego lokalnego trendu nie osiąga wymaganego minimum.
- **Weak-only train probe** — stałe 1 000 obrazów z train 44k, oceniane bez augmentacji w `eval()` wyłącznie do diagnostyki train-validation gap; nie steruje treningiem.
- **Observe-only** — tryb shadow stoppera: zamraża pierwszą hipotetyczną decyzję, ale nie skraca treningu ani nie ładuje wybranego kandydata na granicy faz.
- **Shadow replay do Phase 4** — `observe_only` wykonuje pełny trace Phase 3 bez wcześniejszego stopu, po czym opcjonalnie odtwarza zamrożony hipotetyczny checkpoint stoppera i z niego uruchamia Phase 4. Pozwala ocenić rekomendację stoppera bez utraty późniejszego przebiegu Phase 3.
- **Selected source step** — globalny krok, z którego pochodzi wybrany checkpoint; jest logowany osobno, ponieważ monotoniczny global step wykonanego runu nie jest cofany.
- **FIM chunk size** — liczba przykładów różniczkowanych jednocześnie przez `vmap(grad)`; wpływa na pamięć i kolejność sumowania, nie na definicję estymatora.
- **Weak utility (loss)** — `dominant_only_loss - full_loss`, czyli marginalna redukcja loss po dodaniu weak branch. Może sztucznie rosnąć, gdy dominant-only loss eksploduje, więc nie jest samodzielną miarą jakości.
- **Weak utility (accuracy)** — `full_accuracy - dominant_only_accuracy`. Również wymaga ochrony compatibility, ponieważ może rosnąć wskutek załamania dominant-only.
- **Accuracy-based compatibility constraint** — sparowana bramka non-inferiority względem początku Phase 3. Kandydat jest dopuszczalny tylko wtedy, gdy dolne granice CI zmian full i dominant-only accuracy pozostają powyżej ustalonych ujemnych marginesów.
- **Unimodal reference** — klasyfikator z jednym aktywnym encoderem oraz własnym shared trunk/classifier, trenowany na proper train i wybierany wyłącznie przez validation proper. Referencje left/right są parowane po seedzie i inicjalizacji.
- **Relative unimodal parity** — kryterium `weak_accuracy/unimodal_right_accuracy >= dominant_baseline_accuracy/unimodal_left_accuracy`. Porównuje ułamek osiągalnej jakości modalności zamiast surowych accuracy.
- **Relative recovery fraction** — część początkowego znormalizowanego deficytu weak branch, która została usunięta w Phase 3: `(weak_ratio(e)-weak_ratio(0))/(dominant_ratio-weak_ratio(0))`. Wartość `1` oznacza exact parity; próg poniżej `1` zatrzymuje przed asymptotyczną końcówką uczenia.
- **Parity confirmation window** — dwa kolejne pomiary spełniające relative parity. Drugi potwierdza stop, ale wybierany jest pierwszy checkpoint nieprzerwanej serii.
- **Phase-4 anchor** — niezmieniony stan z `P4 e0`, przed pierwszym krokiem ponownego treningu obu gałęzi. W diagnostyce zawiera prawy encoder oraz cały współdzielony downstream, łącznie z `main_branch`, buforami i klasyfikatorem.
- **Current right encoder** — prawa/weak gałąź z aktualnie ocenianego checkpointu Phase 4, po tylu aktualizacjach P4, ile wskazuje `phase_epoch`; nie jest to prawa gałąź z końca Phase 3.
- **Phase-4 hybrid compatibility diagnostic** — dwie tymczasowe ewaluacje weak-only: `current_right+anchor_shared` mierzy dryf encodera względem końca P3, a `anchor_right+current_shared` mierzy dryf wspólnego downstream względem encodera z końca P3. Żadna podmiana nie zmienia zapisanego modelu.
- **Phase-4 LR warm-up** — opt-in liniowy wzrost LR od skonfigurowanego ułamka do bazowej wartości, wykonywany po każdym optimizer step. Wartość zero zachowuje historyczną Phase 4 bez warm-upu.
- **Phase-4 shared-only prefix** — opt-in etap reintegracji po Phase 3. Oba encodery uczestniczą w forward, ale mają zamrożone parametry i bufory BatchNorm; uczy się wyłącznie wspólny downstream (`main_branch` i classifier). Po ustalonej liczbie epok wszystkie parametry są odblokowywane bez resetu optimizera.
- **Frozen-left-active Phase 3** — alternatywa dla deaktywacji. Lewy encoder
  uczestniczy w forward, ale pozostaje frozen/eval; prawy encoder i shared
  downstream uczą się z pełnego wyjścia. Pozwala sprawdzić, czy stała kotwica
  dominant representation ogranicza compatibility drift.
- **TFIM left/right ratio** — `Tr(F_L)/Tr(F_R)` sampled Fishera liczony na
  tym samym probe, tych samych sampled labelach i dla parametrów obu
  encoderów. Wartość nie jest porównywalna między runami bez identycznego
  probe, RNG, liczby próbek i definicji zbioru parametrów.
- **Asymetryczny pomocniczy loss Phase 4** — opt-in cel `L_full + lambda_R L_weak + lambda_L L_dominant` liczony przez jeden model. W aktualnej kontroli `lambda_R=1`, `lambda_L=0`, więc full i weak-only aktualizują te same wagi, a dominant-only training forward nie jest wykonywany.
- **Sparowany CI recovery gap** — przedział z per-image poprawności tego samego `validation_proper`, skorygowany na wszystkie spojrzenia i wspólnie badane progi. Bieżąca wersja traktuje accuracy modeli unimodalnych jako stałe i jest warunkowa względem wybranych referencji.

- **Per-seed coarse-to-fine Phase-3 best recovery** — retrospektywne
  wyznaczanie czasu interwencji osobno dla seedu. Kandydaci są oceniani
  hierarchicznie względem per-seed clean P2=200 accuracy gold: najpierw luka
  full, potem średnia luka dominant/weak. Coarse grid jest lokalnie zagęszczany
  aż do wyboru jednej całkowitej epoki. Best recovery jest etykietą celu i
  analizą krajobrazu, nie operacyjnym stopperem, ponieważ wymaga wielu pełnych
  przebiegów Phase 4.
- **TFIM dynamics gold** — trajektoria modelu uczonego od losowej inicjalizacji
  wyłącznie na clean P2=17, z TFIM w e2=5/8/11/14/17. Nie jest to krótki P4
  uruchomiony po clean P2=200.
