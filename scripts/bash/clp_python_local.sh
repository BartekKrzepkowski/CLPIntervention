#!/bin/bash

set -euo pipefail

CLP_LOCAL_PREFIX="${CLP_LOCAL_PREFIX:-/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-local-cpu}"

if [[ ! -x "${CLP_LOCAL_PREFIX}/bin/python" ]]; then
  echo "Brak środowiska lokalnego: ${CLP_LOCAL_PREFIX}" >&2
  echo "Uruchom: bash scripts/bash/create_local_env.sh" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CLP_LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
exec "${CLP_LOCAL_PREFIX}/bin/python" "$@"
