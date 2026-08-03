# Phase 3 z trenowalnym shared trunk, P1=120/P2=200

Joby `20359797/20359798/20359799` zakończyły się poprawnie dla seedów
83/184/285. Phase 3 wykonała pełne 200 epok w observe-only, z wyłączoną lewą
gałęzią oraz trenowalnym prawym encoderem i shared trunk/classifier. P4 i test
były wyłączone.

| e3 | mean full acc | mean dominant acc | mean weak acc | mean weak loss |
|---:|---:|---:|---:|---:|
| 0 | 0,8735 | 0,8183 | 0,2251 | 5,2914 |
| 1 | 0,6235 | 0,6307 | 0,5401 | 1,3105 |
| 4 | 0,5910 | 0,6183 | 0,6303 | 1,0630 |
| 20 | 0,5475 | 0,6131 | 0,7688 | **0,7882** |
| 40 | 0,5934 | 0,6379 | 0,7882 | 0,9639 |
| 60 | 0,5801 | 0,6149 | 0,7965 | 1,0488 |
| 80 | 0,5884 | 0,6177 | 0,7971 | 1,1396 |
| 120 | 0,6369 | 0,6304 | 0,8064 | 1,2248 |
| 160 | 0,6359 | 0,6238 | 0,8109 | 1,2999 |
| 200 | 0,6489 | 0,6271 | 0,8140 | 1,3121 |

Shared trunk daje szybkie odzyskanie weak-only, lecz już pierwsza epoka
obniża full o 25,01 pp i dominant-only o 18,75 pp. Czteroepokowy warm-up nie
zapobiega compatibility shock. Weak accuracy po około 40–60 epokach rośnie
już bardzo wolno, podczas gdy weak loss po minimum około e16–24 systematycznie
się pogarsza. To sygnał rosnącej nadmiernej pewności, nie dalszej dużej poprawy
jakości klasyfikacji.

Pierwsze `weak >= dominant` występuje w e3=3/8/4 dla seedów 83/184/285.
Obecny shadow local-accuracy wybrał e3=32/44/20. Nie należy jeszcze uznawać
tych epok za optymalne: w P3 pełny i dominant mode są już silnie uszkodzone, a
celem jest accuracy po ponownej wspólnej adaptacji w P4.

Następna bramka powinna uruchomić identyczną P4=200 z fixed milestone
e3=20/40/60/80 dla wszystkich trzech seedów. Dopiero ich validation proper
po P4 pozwoli ocenić, czy lokalny stopper trafia w przedział optimum. Test
pozostaje wyłączony.
