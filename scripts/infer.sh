#!/usr/bin/env bash
set -euo pipefail

python src/main.py --mode infer "$@"

EXP_NAME="${EXPERIMENT_NAME:-}"
NEXT_IS_EXP=0
for arg in "$@"; do
	if [[ "$NEXT_IS_EXP" -eq 1 ]]; then
		EXP_NAME="$arg"
		NEXT_IS_EXP=0
		continue
	fi
	case "$arg" in
		--experiment-name)
			NEXT_IS_EXP=1
			;;
		--experiment-name=*)
			EXP_NAME="${arg#--experiment-name=}"
			;;
	esac
done
EXP_NAME=${EXP_NAME:-${EXPERIMENT_NAME:-polysegnet}}
INFER_DIR="experiments/${EXP_NAME}/inference"
FIG_DIR="outputs/figures"

if [[ -d "${INFER_DIR}" ]]; then
	mkdir -p "${FIG_DIR}"
	cp -f "${INFER_DIR}"/*_overlay.png "${FIG_DIR}"/ 2>/dev/null || true
fi
