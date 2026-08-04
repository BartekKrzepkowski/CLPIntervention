# Indeks runów W&B

Kanoniczne lokalne mapowanie identyfikatorów W&B na job Slurm i wariant
eksperymentu znajduje się w
[`analysis/results/wandb_run_index.csv`](../analysis/results/wandb_run_index.csv).
Nie zawiera tokenów, ścieżek maszynowych ani danych uwierzytelniających.

Każdy wpis podaje projekt, identyfikator runu, job Slurm, profil protokołu,
seed, długości faz, status i krótką uwagę. Nowe istotne runy należy dopisywać
po wysłaniu joba, a status aktualizować po jego zakończeniu.

Replay nowego lokalnego stoppera z pobranego wcześniej cache można odtworzyć
bez kolejnego skanowania W&B:

```bash
scripts/bash/clp_python_local.sh scripts/python_new/replay_phase3_accuracy_stopper.py \
  --input-json analysis/results/phase3_accuracy_compatibility_replay_p1_40.json \
  --output-prefix analysis/results/phase3_local_accuracy_replay_p1_40
```

Gotowe wyniki replayu są zapisane w plikach JSON i CSV o powyższym prefiksie.
Opcje `--run SEED=entity/project/run_id` nadal pozwalają pobrać nową
trajektorię, gdy nie ma jej jeszcze w lokalnym cache.

## Bieżąca seria TFIM clean controls v2

Po bramce `afterok:20362506` ukończono dziewięć jobów:

- P2: seed83 `20362516/99h4r4m9`, seed184
  `20362509/m77b0ipa`, seed285 `20362511/d3fztzra`;
- left: seed83 `20362515/a53x3v38`, seed184
  `20362512/wqas8hr5`, seed285 `20362510/vkw1rd9g`;
- right: seed83 `20362508/4y2przjj`, seed184
  `20362513/dryoqryt`, seed285 `20362514/f5rn25am`.

Wszystkie runy mają status `completed`; lokalna analiza jest w
`analysis/results/tfim_clean_controls_50_v2/`.

## Phase-3 oracle TFIM refinement

- `20388014`–`20388015`: pełne P4=200 seed 83 dla e3=79/81, `completed`;
  pas full-gap 1 pp nadal wybiera e3=80 w skanie 78..82.
- `20388156`–`20388163`: pełne P4=200 seed 184 dla brakujących całkowitych
  e3=51..59; pas 1 pp wybiera e3=57 w skanie 50..60.
- `20388720`–`20388725`: pełne P4=200 seed 285 dla e3=73/74/76/77/78/79.
- `20388720`–`20388725`: wszystkie `completed`; pas 1 pp na gęstym skanie
  seedu 285 wybiera e3=74.
- `20388166`: diagnostyczne resume P3 e54→e59, `completed`, ale niezgodne z
  trajektorią nieprzerwaną. Powtórzenie `20388773` po zmianie kolejności
  odtwarzania RNG także jest `completed`, lecz nadal nie odtwarza dokładnie
  nieprzerwanego przebiegu; resume pozostaje wyłączone z materializacji.

- `20384859` (`inoeeaco`): seed 184, e3=47, pełne P4=200, `completed`;
  najlepsze validation proper full accuracy `0,8842` w e4=185.
- `20384893` (`uaqe96h4`): seed 83, materializacja P3 e3=78/82,
  `completed`.
- `20384894` (`5zx7ob3t`): seed 285, materializacja P3 e3=68/72,
  `completed`.
- `20386248`–`20386251`: pełne P4=200 odpowiednio dla seed/e3
  `83/78`, `83/82`, `285/68`, `285/72`; wszystkie `completed`.
- `20387410`–`20387412`: materializacje dokładnych checkpointów P3 dla
  finalnego skanu e3=79/81, 45/46/48/49 i 69/71.
- `20387413`: smoke clean-from-scratch TFIM dynamics gold P2=2.
- `20387443`–`20387445`: właściwy TFIM dynamics gold P2=17 dla seedów
  83/184/285, zależny `afterok:20387413`.

## Końcowy gęsty refinement TFIM gold-slope

- `20393538`: compute smoke zapisu checkpointów materializacyjnych,
  `completed`.
- `20393540`: seed 83, jedna trajektoria P3 do e80, checkpointy e75..80,
  `completed`.
- `20393541`: seed 184, jedna trajektoria P3 do e55, checkpointy e50..55,
  `completed`.
- `20393542`: seed 285, jedna trajektoria P3 do e70, checkpointy e65..70,
  `completed`.
- `20394087`–`20394095`: sześć P4=17 seed 83 dla e3=75..80, wszystkie
  `completed`.
- `20394096`, `20394097`, `20394103`, `20394116`–`20394118`: sześć P4=17
  seed 184 dla e3=50..55, wszystkie `completed`.
- `20394119`–`20394122`, `20394124`, `20394126`: sześć P4=17 seed 285 dla
  e3=65..70, wszystkie `completed`.

Najbliższy per-seed clean P2=17 gold slope wybiera e3=77/54/69 wobec
downstream oracle 80/57/74. Jest to końcowa negatywna walidacja bieżącej
reguły stopu, nie nowy wynik oracle.
