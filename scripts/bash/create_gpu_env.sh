#!/bin/bash
# Uruchamiaj wyłącznie jako zadanie Slurm na węźle GH200/aarch64.

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Ten skrypt wymaga węzła GH200/aarch64." >&2
  exit 1
fi

CLP_CONDA="${CLP_CONDA:-/net/storage/pr3/plgrid/plggdnnp/apps/miniforge3-gh200/bin/conda}"
CLP_BASE_PREFIX="${CLP_BASE_PREFIX:-/net/storage/pr3/plgrid/plggdnnp/conda_envs/lapsum-gh200}"
CLP_GPU_PREFIX="${CLP_GPU_PREFIX:-/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-gh200}"
CLP_CACHE_ROOT="${CLP_CACHE_ROOT:-/net/storage/pr3/plgrid/plggdnnp/cache/plgkrzepk}"
CLP_TMP_ROOT="${CLP_TMP_ROOT:-/net/storage/pr3/plgrid/plggdnnp/tmp/plgkrzepk}"

mkdir -p "${CLP_CACHE_ROOT}/clpintervention-gh200-conda-pkgs" "${CLP_CACHE_ROOT}/clpintervention-pip" "${CLP_TMP_ROOT}/clpintervention-gh200-tmp"

if [[ ! -x "${CLP_GPU_PREFIX}/bin/python" ]]; then
  CONDA_PKGS_DIRS="${CLP_CACHE_ROOT}/clpintervention-gh200-conda-pkgs" TMPDIR="${CLP_TMP_ROOT}/clpintervention-gh200-tmp" "${CLP_CONDA}" create --yes --prefix "${CLP_GPU_PREFIX}" --clone "${CLP_BASE_PREFIX}"
fi

LD_LIBRARY_PATH="${CLP_GPU_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PIP_CACHE_DIR="${CLP_CACHE_ROOT}/clpintervention-pip" TMPDIR="${CLP_TMP_ROOT}/clpintervention-gh200-tmp" "${CLP_GPU_PREFIX}/bin/python" -m pip install pytest==9.1.1
if [[ "${CLP_INSTALL_OPTIONAL:-0}" == "1" ]]; then
  LD_LIBRARY_PATH="${CLP_GPU_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PIP_CACHE_DIR="${CLP_CACHE_ROOT}/clpintervention-pip" TMPDIR="${CLP_TMP_ROOT}/clpintervention-gh200-tmp" "${CLP_GPU_PREFIX}/bin/python" -m pip install tensorboard==2.21.0 captum==0.9.0
fi
LD_LIBRARY_PATH="${CLP_GPU_PREFIX}/lib:${LD_LIBRARY_PATH:-}" "${CLP_GPU_PREFIX}/bin/python" -c "import platform, pytest, torch, torchvision; assert platform.machine() == \"aarch64\"; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name())"
