#!/bin/bash
# Queue runner for the compute server (plan 2 Task 2): runs the SR jobs of QUEUE (one "SCALE VARIANT TEMPLATE SEED NITER N_ORIG" per
# line), at most MAXP at a time, SR_THREADS threads each. Pooled regions = the search cities of config/cities.csv
# (role == search, file order); s_health is dropped from the search; wall clock 43200 s for city<N> runs, 28800 s for
# the rest. Logs: results/sr/M/logs/run_<tag>.log; the last line of the queue log is QUEUE_DONE.
# Usage: setsid nohup bash scripts/multicity/run_queue.sh scripts/multicity/config/queue_search.txt 4 \
#          > results/sr/M/logs/queue.log 2>&1 < /dev/null & disown
set -u
export ODSR_ROOT=${ODSR_ROOT:?set ODSR_ROOT to the data root; this repository lives in $ODSR_ROOT/scripts}
cd "$ODSR_ROOT"
Q=$1; MAXP=${2:-4}
export ODSR_PY=${ODSR_PY:-python}
export ODSR_LINE=M
C=scripts/multicity/config/cities.csv
export ODSR_CITIES=$("$ODSR_PY" -I -c "import pandas as pd; print(','.join(pd.read_csv('$C', dtype=str)['city']))")
export SR_REGIONS=$("$ODSR_PY" -I -c "import pandas as pd; c = pd.read_csv('$C', dtype=str); print(','.join(c[c['role'] == 'search']['city']))")
export SR_DROP_VARS=s_health SR_NEG_RATE=0.1 SR_TURBO=1 SR_THREADS=${SR_THREADS:-16}
export SR_BLOCK=${SR_BLOCK:-search}     # destination block of the search: search = folds 2-4; agent loop = search23
export SR_GUESSES=${SR_GUESSES-$ODSR_ROOT/scripts/multicity/config/guesses.json}   # seeds (round 2); SR_GUESSES= disables
export SR_TIMEOUT_POOLED=${SR_TIMEOUT_POOLED:-43200} SR_TIMEOUT_SINGLE=${SR_TIMEOUT_SINGLE:-28800}
mkdir -p results/sr/M/logs
echo "$(date '+%F %T') QUEUE_START $Q regions=$SR_REGIONS block=$SR_BLOCK maxp=$MAXP threads=$SR_THREADS"
grep -v '^#' "$Q" | grep -v '^\s*$' | xargs -P "$MAXP" -L 1 bash -c \
  'tag="$0_$1_${2//:/-}_s$3"; case "$2" in city*) t=$SR_TIMEOUT_POOLED;; *) t=$SR_TIMEOUT_SINGLE;; esac;
   g="$ODSR_ROOT/scripts/multicity/config/guesses/$tag.json"; [ -f "$g" ] && export SR_GUESSES="$g";
   echo "$(date "+%F %T") start $tag timeout $t seeds ${SR_GUESSES##*/}";
   SR_TIMEOUT=$t "$ODSR_PY" -u scripts/sr/run_sr.py "$0" "$1" "$2" "$3" "$4" "$5" > "results/sr/M/logs/run_$tag.log" 2>&1;
   rc=$?; echo "$(date "+%F %T") end $tag rc $rc"'
echo "$(date '+%F %T') QUEUE_DONE $Q"
