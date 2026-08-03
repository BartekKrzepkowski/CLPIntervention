# Sparowane przedziały ufności dla znormalizowanego odzyskania weak branch

Dla checkpointu `e` oznaczamy przez `w_e(i)` poprawność weak-only na obrazie
walidacyjnym `i`, przez `w_0(i)` weak-only przed Phase 3, a przez `d_0(i)`
dominant-only przed Phase 3. Accuracies sparowanych modeli unimodalnych to
`U_R` i `U_L`. Dla wymaganej części odzyskania `q` badamy:

```text
G_q(e) = mean(w_e)/U_R
         - (1-q) mean(w_0)/U_R
         - q mean(d_0)/U_L
```

`G_q(e) >= 0` oznacza odzyskanie co najmniej części `q` początkowego
znormalizowanego deficytu; `q=1` to exact relative parity. Przedział jest
sparowany, bo wszystkie trzy poprawności dotyczą tego samego obrazu z
`validation_proper`. Dwustronny normalny CI 95% ma korektę Bonferroniego na
54 spojrzenia trajektorii i dwie wspólnie badane rodziny `q=0.99/1.0`.
Dwa kolejne pomiary z dolną granicą co najmniej zero potwierdzają pierwszy.
Bez potwierdzenia fallback maksymalizuje dolną granicę CI.

## Wynik P1=40/P2=200

| seed | e3 99% | CI 99% | e3 100% | CI 100% |
|---:|---:|---:|---:|---:|
| 83 | 188 | [-0,02237; 0,03207] | 188 | [-0,02912; 0,02560] |
| 184 | 176 | [-0,01139; 0,04397] | 176 | [-0,01892; 0,03672] |
| 285 | 156 | [-0,02531; 0,03089] | 156 | [-0,03322; 0,02327] |

Żaden seed nie uzyskał dwóch potwierdzonych crossingów. Oba progi wybrały te
same fallbacki. Artefakty są w
`analysis/results/unimodal_recovery_ci_q99_q100_2026-08-02/`.

CI jest warunkowy względem wybranych referencji: trajektoria zachowała ich
scalar accuracy, ale nie per-image predykcje, więc niepewność mianowników
`U_R/U_L` nie jest uwzględniona. Pełny CI ratio wymaga wspólnego bootstrapu
predykcji referencji. Mimo tego repeated-look correction jest już zbyt
konserwatywna jako praktyczny wczesny stopper.

