#!/bin/bash

set -euo pipefail

CLP_CONDA="${CLP_CONDA:-/net/storage/pr3/plgrid/plggdnnp/apps/miniforge3/bin/conda}"
CLP_BASE_PREFIX="${CLP_BASE_PREFIX:-/net/storage/pr3/plgrid/plggdnnp/conda_envs/lapsum-local-cpu}"
CLP_LOCAL_PREFIX="${CLP_LOCAL_PREFIX:-/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-local-cpu}"
CLP_CACHE_ROOT="${CLP_CACHE_ROOT:-/net/storage/pr3/plgrid/plggdnnp/cache/plgkrzepk}"
CLP_TMP_ROOT="${CLP_TMP_ROOT:-/net/storage/pr3/plgrid/plggdnnp/tmp/plgkrzepk}"

mkdir -p "${CLP_CACHE_ROOT}/clpintervention-conda-pkgs" "${CLP_CACHE_ROOT}/clpintervention-pip" "${CLP_TMP_ROOT}/clpintervention-tmp"

if [[ ! -x "${CLP_LOCAL_PREFIX}/bin/python" ]]; then
  CONDA_PKGS_DIRS="${CLP_CACHE_ROOT}/clpintervention-conda-pkgs" TMPDIR="${CLP_TMP_ROOT}/clpintervention-tmp" "${CLP_CONDA}" create --yes --prefix "${CLP_LOCAL_PREFIX}" --clone "${CLP_BASE_PREFIX}"
fi

LD_LIBRARY_PATH="${CLP_LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PIP_CACHE_DIR="${CLP_CACHE_ROOT}/clpintervention-pip" TMPDIR="${CLP_TMP_ROOT}/clpintervention-tmp" "${CLP_LOCAL_PREFIX}/bin/python" -m pip install pytest==9.1.1
if [[ "${CLP_INSTALL_OPTIONAL:-0}" == "1" ]]; then
  LD_LIBRARY_PATH="${CLP_LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PIP_CACHE_DIR="${CLP_CACHE_ROOT}/clpintervention-pip" TMPDIR="${CLP_TMP_ROOT}/clpintervention-tmp" "${CLP_LOCAL_PREFIX}/bin/python" -m pip install tensorboard==2.21.0 captum==0.9.0
fi
LD_LIBRARY_PATH="${CLP_LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" "${CLP_LOCAL_PREFIX}/bin/python" -c "import pytest, torch, torchvision; print(torch.__version__, torchvision.__version__)"
