# Przechowywanie artefaktów eksperymentalnych

Wszystkie nowe artefakty treningu i obliczeń muszą trafiać do trwałego
`$REPORTS_DIR`, który ma wskazywać katalog pod `/net/storage`. Do repozytorium
nie zapisujemy checkpointów, surowych trajektorii, predykcji per-example,
lokalnych danych W&B ani logów Slurm.

Wymagany układ jest następujący:

- `$REPORTS_DIR/slurm_logs/` — stdout/stderr Slurma;
- `$REPORTS_DIR/<run>/checkpoints/` — checkpointy i stan treningu;
- `$REPORTS_DIR/<run>/trace_fim_train.jsonl` — surowa trajektoria TFIM;
- `$REPORTS_DIR/<run>/wandb/` — lokalne dane klienta W&B;
- repozytorium `analysis/results/` — wyłącznie małe, oczyszczone ze ścieżek
  absolutnych tabele, manifesty podsumowujące i wybrane wykresy.

Przed submission należy sprawdzić lokalizację:

```bash
printf '%s\n' "$REPORTS_DIR"
```

Joby uruchamiamy przez:

```bash
scripts/bash/submit_experiment.sh scripts.python_new.run_single \
  config=configs/experiments/<experiment>.yaml
```

Helper odrzuca `REPORTS_DIR` poza `/net/storage` i zapisuje logi w
`$REPORTS_DIR/slurm_logs`. Starych artefaktów w worktree nie wolno
automatycznie usuwać ani przenosić: migracja musi mieć jawny zakres i zachować
powiązania między logami, checkpointami oraz analizami.
