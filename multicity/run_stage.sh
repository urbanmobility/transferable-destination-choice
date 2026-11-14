#!/usr/bin/env bash
# Generic stage runner (compute server) with the loop's environment (no LLM variables): run_stage.sh SCRIPT STAGE [ARGS]
# e.g. setsid nohup bash scripts/multicity/run_stage.sh lawtest_extra.py register > /dev/null 2>&1 < /dev/null & disown
# Log: results/multicity/logs/<script stem>.log
set -uo pipefail
cd "${ODSR_ROOT:?set ODSR_ROOT to the data root; this repository lives in $ODSR_ROOT/scripts}"
export ODSR_ROOT ODSR_LINE=${ODSR_LINE:-M} ODSR_PY=${ODSR_PY:-python}
ODSR_CITIES=$($ODSR_PY -I -c "import pandas as pd; print(','.join(pd.read_csv('scripts/multicity/config/cities.csv', dtype=str)['city']))")
REQ_OUTER=${LOOP_OUTER_PROCS:-24}
# outer workers (one city dataset each, ~3 GB) capped by the memory available now (20 GB kept free)
LOOP_OUTER_PROCS=$($ODSR_PY -I scripts/multicity/memprocs.py "$REQ_OUTER")
export ODSR_CITIES SR_DROP_VARS=s_health LOOP_PROCS=${LOOP_PROCS:-48} LOOP_OUTER_PROCS
SCRIPT=${1:?script}; STAGE=${2:?stage}; shift 2
SUF=$([ "$ODSR_LINE" = M ] || echo "_$ODSR_LINE")      # plan 7: other data lines log beside the M line
LOG=results/multicity/logs/${SCRIPT%.py}${SUF}.log
echo "== $(date -u '+%F %T') UTC $SCRIPT $STAGE $* start (outer workers $LOOP_OUTER_PROCS of $REQ_OUTER requested)" >> "$LOG"
$ODSR_PY -I -u "scripts/multicity/$SCRIPT" "$STAGE" "$@" >> "$LOG" 2>&1
rc=$?
echo "== $(date -u '+%F %T') UTC $SCRIPT $STAGE exit $rc" >> "$LOG"
exit $rc
