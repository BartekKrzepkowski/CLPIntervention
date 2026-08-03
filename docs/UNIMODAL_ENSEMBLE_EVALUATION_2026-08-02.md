# Post-hoc UME i rozszerzony sweep Phase 3

## Ustalona definicja UME

Skrót UME nie występuje w kodzie ani w dwóch pracach przechowywanych w tym
repozytorium. Dlatego analiza nie przypisuje mu nieudokumentowanej definicji.
Wynik główny `ume_probability_mean` jest równoważnym wagowo soft votingiem
dwóch niezależnych modeli unimodalnych:

\[
p_{UME}(y\mid x)=\tfrac12\left[p_L(y\mid x_L)+p_R(y\mid x_R)\right].
\]

Wagi `1/2,1/2` ustalono przed pomiarem i nie dobierano ich na walidacji.
`ume_logit_mean_sensitivity` uśrednia logity przed softmaxem i jest wyłącznie
analizą czułości. Modele zachowują własne encodery, shared trunk i klasyfikator;
nie składamy ich warstw w jeden nowy model. Test jest liczony post hoc po
zamrożeniu definicji ensemble.

## Wyniki UME

Accuracy na `validation_proper`:

| seed | left | right | UME probability | UME logits (sensitivity) | clean gold full |
|---:|---:|---:|---:|---:|---:|
| 83 | 0,8594 | 0,8506 | 0,9058 | 0,9170 | 0,9088 |
| 184 | 0,8644 | 0,8556 | 0,9092 | 0,9182 | 0,8934 |
| 285 | 0,8612 | 0,8564 | 0,9080 | 0,9146 | 0,8914 |
| **średnia** | **0,8617** | **0,8542** | **0,9077** | **0,9166** | **0,8979** |

Accuracy na oryginalnym `test_proper`:

| seed | left | right | UME probability | UME logits (sensitivity) | clean gold full |
|---:|---:|---:|---:|---:|---:|
| 83 | 0,8466 | 0,8519 | 0,9037 | 0,9080 | 0,8982 |
| 184 | 0,8468 | 0,8459 | 0,8971 | 0,9068 | 0,8847 |
| 285 | 0,8487 | 0,8496 | 0,8996 | 0,9072 | 0,8850 |
| **średnia** | **0,8474** | **0,8491** | **0,9001** | **0,9073** | **0,8893** |

Główne UME poprawia średnią względem clean gold o `+0,98 pp` na walidacji i
`+1,08 pp` na teście. Średnia logitów daje `+1,87 pp` i `+1,80 pp`, lecz nie
wolno po tych wynikach zamienić jej w wynik główny bez jawnej, z góry
ustalonej zmiany protokołu. UME jest też wyraźnie lepsze od obu pojedynczych
referencji. Wskazuje to na komplementarne błędy pól, a nie na przewagę jednego
modelu unimodalnego.

Poza accuracy zapisano NLL, Brier, ECE-15, mean confidence i mean incorrect
confidence. Probability mean ma mean test NLL `0,4281` i ECE `0,0437`, wobec
clean gold `0,7149` i `0,0839`; jest więc również lepiej skalibrowane. Pełne
JSON-y są w `${REPORTS_DIR}/posthoc_ume/`, a surowa tabela w
`analysis/results/unimodal_ensemble_2026-08-02.csv`.

## Repeated-look CI

Trajektoria ma 54 możliwe momenty podjęcia decyzji: `e3=0`, gęste pomiary
`1,2,3,4`, następnie co cztery epoki do 200 i pomiar końcowy. Dla obrazu `i`
i progu odzyskania `q` replay liczy sparowany gap:

\[
g_i(q,e)=\frac{w_{e,i}}{U_R}
 -(1-q)\frac{w_{0,i}}{U_R}
 -q\frac{d_{0,i}}{U_L}.
\]

Dolna granica to lower confidence bound średniej `G_q=mean_i(g_i)`. Trafienie
oznacza `LCB >= 0`. Stop wymaga dwóch trafień w kolejnych zaplanowanych
pomiarach, a wybrany zostaje pierwszy checkpoint tej serii.

Pełna korekta repeated-look kontroluje błąd wynikający z wielokrotnego pytania
o ten sam warunek. Bonferroni rozdziela `alpha=0,05` przez `54` momenty i dwie
rodziny progów (`q=0,99/1,00`), czyli daje około `0,000463` na test i
`z≈3,50` dla dwustronnego przedziału. Jest to konserwatywne: dobrze ogranicza
fałszywy wczesny stop, ale silnie poszerza przedziały i obniża moc. W żadnym
seedzie LCB nie przekroczyło zera w dwóch kolejnych pomiarach. Fallbacki
wyniosły `e3=188/176/156`.

Przykładowe najlepsze późne punkty nadal obejmują zero:

| seed | e3 | q | mean gap | skorygowany CI |
|---:|---:|---:|---:|---:|
| 83 | 188 | 0,99 | 0,00485 | [-0,02237; 0,03207] |
| 184 | 176 | 0,99 | 0,01629 | [-0,01139; 0,04397] |
| 285 | 156 | 0,99 | 0,00279 | [-0,02531; 0,03089] |

## Rozszerzony sweep i loss Phase 4

| wariant | wybrane e3 | mean full | mean dominant | mean weak |
|---|---:|---:|---:|---:|
| fixed milestone | 140 | **0,8984** | 0,7826 | 0,4145 |
| fixed milestone | 160 | 0,8981 | 0,7903 | 0,4604 |
| repeated-look CI fallback | 188/176/156 | 0,8973 | 0,7855 | 0,4585 |
| `L_full+L_weak`, tylko P4 do e4=10 | 40 | 0,8682 | 0,8001 | **0,7053** |

W frozen-shared Phase 3 nie występuje historyczne załamanie w okolicy 80
epok. Full accuracy rośnie do e3=140 i następnie osiąga plateau; późny stopper
CI nie poprawia e140. `L_full+L_weak` silnie chroni samodzielną prawą gałąź,
ale krótki probe traci około `0,91 pp` full wobec odpowiadającej mu kontroli
P4 e4=10. Wymaga P4=200 i/lub strojenia `lambda_R` wyłącznie na walidacji.

Checkpointy `e3=40` w tym probe nie były wyborem stoppera. To wspólny,
sparowany benchmark izolujący wpływ funkcji celu P4: te same checkpointy i
ta sama długość interwencji dla wszystkich wariantów. Osobne joby
`20357275`–`20357277` uruchomiły P4 dokładnie z checkpointów wskazanych przez
fallback repeated-look CI.
