#!/bin/bash
# Best test accuracy per run, one row each, sorted best first.
#
# Usage: bash AO_QAT/resnet_cifar10/results.sh [glob]
#   e.g. bash AO_QAT/resnet_cifar10/results.sh                  # every run
#        bash AO_QAT/resnet_cifar10/results.sh '*qvr*'          # QVR arms only
#        LOG=/content/drive/MyDrive/aoq/log bash .../results.sh # logs elsewhere
#
# Reads the "test epoch N ... acc@1 A" lines. Epoch -2 is the full-precision
# teacher, evaluated once before training, so it is reported separately as
# "fp" rather than counted as a result.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG=${LOG:-log}
PATTERN=${1:-*}

shopt -s nullglob
FILES=("${LOG}"/${PATTERN}/training.txt)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "no logs at ${LOG}/${PATTERN}/training.txt" >&2
    exit 1
fi

{
printf 'best@1   epoch  last@1  done  fp@1    run\n'
printf -- '-------  -----  ------  ----  ------  ---\n'
for f in "${FILES[@]}"; do
    run=$(basename "$(dirname "${f}")")
    # Field positions shift with the log prefix, so locate the labels by name.
    awk -v run="${run}" '
        /test/ && /epoch/ && /acc@1/ {
            ep = ""; a1 = ""
            for (i = 1; i <= NF; i++) {
                if ($i == "epoch") ep = $(i + 1)
                if ($i == "acc@1") a1 = $(i + 1)
            }
            if (ep == "" || a1 == "") next
            if (ep < 0) { fp = a1; next }     # the teacher, not a result
            n++
            last = a1; last_ep = ep
            if (best == "" || a1 + 0 > best + 0) { best = a1; best_ep = ep }
        }
        END {
            if (n == 0) { printf "%-7s  %-5s  %-6s  %-4d  %-6s  %s\n",
                          "-", "-", "-", 0, (fp == "" ? "-" : fp), run; exit }
            printf "%-7s  %-5s  %-6s  %-4d  %-6s  %s\n",
                   best, best_ep, last, n, (fp == "" ? "-" : fp), run
        }
    ' "${f}"
done
} | { read -r h1; read -r h2; printf '%s\n%s\n' "$h1" "$h2"; sort -gr; }
