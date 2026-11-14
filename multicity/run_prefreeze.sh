#!/usr/bin/env bash
# Plan 5 (pre-freeze wrap-up) stage runner on the compute server: the loop's environment, no LLM variables.
# Usage (from anywhere): setsid nohup bash scripts/multicity/run_prefreeze.sh STAGE [ARGS] > /dev/null 2>&1 < /dev/null & disown
# STAGE: reforms | rank | discovery | banded [CITY ...] | report. Log: results/multicity/logs/prefreeze.log
set -uo pipefail
cd "${ODSR_ROOT:?set ODSR_ROOT to the data root; this repository lives in $ODSR_ROOT/scripts}"
export ODSR_ROOT ODSR_LINE=M ODSR_PY=${ODSR_PY:-python}
ODSR_CITIES=$($ODSR_PY -I -c "import pandas as pd; print(','.join(pd.read_csv('scripts/multicity/config/cities.csv', dtype=str)['city']))")
export ODSR_CITIES SR_DROP_VARS=s_health LOOP_PROCS=${LOOP_PROCS:-48} LOOP_OUTER_PROCS=${LOOP_OUTER_PROCS:-24}
LOG=results/multicity/logs/prefreeze.log
echo "== $(date -u '+%F %T') UTC prefreeze $* start" >> "$LOG"
$ODSR_PY -I -u scripts/multicity/prefreeze.py "$@" >> "$LOG" 2>&1
rc=$?
echo "== $(date -u '+%F %T') UTC prefreeze $1 exit $rc" >> "$LOG"
exit $rc
