# Etapowa reintegracja modalności w Phase 4

## Motywacja

Hybrydowa diagnostyka P4 dla `P1=40, P2=200, e3=40` wskazała, że pierwszy
krok wspólnego treningu niszczy przede wszystkim zgodność prawego encodera ze
zmieniającym się shared downstream. Bez warm-upu mean weak-only spadło z
`0,7720` w e4=0 do `0,3905` w e4=1. Podmiana `current_right+anchor_shared`
zachowała `0,7243`, podczas gdy `anchor_right+current_shared` osiągnęła tylko
`0,4814`.

Pierwszą kontrolą jest dlatego etap `shared_only`: oba encodery biorą udział w
forward na proper input, ale pozostają w `eval()`, mają `requires_grad=False`
i nie aktualizują wag ani buforów BatchNorm. Aktualizowany jest wyłącznie cały
shared downstream, czyli `main_branch`, jego bufory i oddzielny klasyfikator
`fc`. Po skonfigurowanej liczbie lokalnych epok P4 wszystkie moduły zostają
odblokowane. Optimizer od początku zawiera wszystkie parametry P4, więc
przejście nie resetuje stanu już uczonego shared downstream i jest zgodne z
resume.

Konfiguracja jest jawna i domyślnie wyłączona:

```yaml
phase4_staged_unfreezing:
  enabled: true
  shared_only_epochs: 4
```

Metryki `phase4/shared_only_active`, `phase4/trainable_left`,
`phase4/trainable_right` i `phase4/trainable_shared` dokumentują granicę
etapu. Hybrydowe pomiary w e4=0,1,2,3,4,5,10 pozwalają rozdzielić dryf
encodera od dryfu shared downstream.

## Aktualnie badana hipoteza: asymetryczny loss

Najpierw izolujemy `L_P4 = L_full + lambda_R L_weak` z `lambda_R=1` i
`lambda_L=0`. Jest to jeden model z tymi samymi wagami. Full forward przekazuje
do shared downstream sumę reprezentacji, a weak-only forward tylko prawą.
Oba lossy aktualizują ten sam prawy encoder, `main_branch` i klasyfikator.
Przy `lambda_L=0` dominant-only training forward nie jest wykonywany;
dominant-only pozostaje diagnostyką walidacyjną. Pierwsza kontrola używa
pełnego odmrożenia i dotychczasowego czteroepokowego per-step warm-upu.

## Odłożony plan czterech kroków

1. Zmierzyć asymetryczny loss przy pełnym odmrożeniu w e4=10, a dopiero przy
   poprawie weak bez szkody full uruchomić P4=200.
2. Na tym samym lossie włączyć `shared_only=4`, aby shared downstream najpierw
   uczył się przy stałych encoderach.
3. Następnie odblokować prawy encoder z normalnym LR, a lewy z mniejszym LR
   albo później; jest to osobna ablacja grup parametrów.
4. Dopiero osobna ablacja może sprawdzić czasowy ramp reintegracji lewej
   reprezentacji `h = h_R + alpha(e) h_L`, gdzie `alpha` rośnie do `1` i
   pozostaje równe `1` przed selekcją checkpointu. Stałe tłumienie lewej
   gałęzi w treningu i inferencji zmieniałoby model docelowy i utrudniało
   przypisanie efektu samej interwencji.

## Związek z literaturą

Gradient Blending wskazuje, że modalności przeuczają się i generalizują w
różnym tempie, więc jeden wspólny sygnał optymalizacji bywa suboptymalny:
[Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html).
OGM adaptacyjnie osłabia aktualizację modalności dominującej według jej
bieżącego wkładu:
[Peng et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html).
Conditional learning speed i ReconBoost dostarczają odpowiednio uzasadnienia
dla monitorowania nierównego tempa uczenia oraz dla naprzemiennego trenowania
modalności:
[Wu et al., ICML 2022](https://proceedings.mlr.press/v162/wu22d.html),
[Hua et al., ICML 2024](https://proceedings.mlr.press/v235/hua24a.html).
Gradual unfreezing z ULMFiT jest analogią optymalizacyjną, ale nie jest
bezpośrednim dowodem dla tej architektury wizyjnej:
[Howard i Ruder, ACL 2018](https://aclanthology.org/P18-1031/).

Literatura wspiera zatem rozdzielanie tempa i kolejności aktualizacji, lecz
nie wyznacza za nas liczby epok ani stałej skali dla pól CIFAR-10. Te elementy
pozostają ablacjami wybieranymi wyłącznie na validation proper.

## Wynik pierwszej kontroli

Pierwsze joby `20355917`–`20355919` były nieważne jako staged control:
wejściowy manifest zawierał `enabled=true`, ale sekcja nie została przekazana
do `run_config` trenera. Błąd wykryto z `phase_summaries.jsonl`, poprawiono i
dodano regresję kontraktu. Poprawne runy `20355956`–`20355958` zakończyły się
`COMPLETED (0:0)`.

Mean validation proper:

| e4 | full | dominant-only | weak-only | current-right + anchor-shared |
|---:|---:|---:|---:|---:|
| 0 | 82,86% | 81,72% | 77,20% | 77,20% |
| 1 | 88,02% | 71,97% | 52,55% | 77,20% |
| 4 | 87,83% | 71,43% | 46,26% | 77,20% |
| 5 | 87,49% | 71,55% | 38,00% | 75,49% |
| 10 | 87,87% | 74,05% | 41,17% | 71,55% |

W e4=1–4 oba encodery są niezmienne, czego bezpośrednim potwierdzeniem jest
stałe `current-right + anchor-shared = 77,20%`. Spadek weak-only do 46,26%
pochodzi zatem wyłącznie ze zmian shared downstream. Wobec zwykłego warm-upu
weak-only jest lepsze o 4,47 pp w e4=4 i 2,80 pp w e4=10, ale prefix nie usuwa
załamania. Sam `L_full` nie wymaga, aby shared downstream pozostał dobrym
klasyfikatorem obu pojedynczych reprezentacji. Następny test dodaje dlatego
jawny sygnał weak-only z `lambda_L=0`;
wydłużanie samego shared-only prefix nie ma obecnie uzasadnienia.
