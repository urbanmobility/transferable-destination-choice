#!/bin/bash
# Front snapshots on the compute server (plan 2 Task 2): saves each run's PySR front checkpoint (pysr/hall_of_fame.csv) to
# runs/<tag>/snapshots/hof_<mmdd_HHMM>.csv whenever it changed, every 20 min, while a run_sr.py process of the odsr
# environment or run_queue.sh is alive; then one last pass and SNAP_DONE. PySR keeps only the lowest-training-loss
# equation per complexity, so a candidate with a better validation score can leave the front; the selection pools
# all snapshots (SELECT_SNAPSHOTS=1).
# Usage: setsid nohup bash scripts/multicity/snap_hof.sh > results/sr/M/logs/snap_hof.log 2>&1 < /dev/null & disown
export ODSR_ROOT=${ODSR_ROOT:?set ODSR_ROOT to the data root; this repository lives in $ODSR_ROOT/scripts}
mkdir -p "$ODSR_ROOT/results/sr/M/runs"
cd "$ODSR_ROOT/results/sr/M/runs"
snap() {
  for f in */pysr/hall_of_fame.csv; do
    [ -e "$f" ] || continue
    t=${f%%/*}; mkdir -p "$t/snapshots"
    last=$(ls "$t"/snapshots/hof_*.csv 2>/dev/null | tail -1)
    if [ -z "$last" ] || ! cmp -s "$f" "$last"; then cp "$f" "$t/snapshots/hof_$(date +%m%d_%H%M).csv"; echo "$(date '+%F %T') $t"; fi
  done
}
alive() {
  ps -eo args | awk '$1 ~ /envs\/odsr\/bin\/python$/ && /run_sr.py/' | grep -q . \
    || ps -eo args | awk '$1 ~ /bash$/ && /run_queue.sh/' | grep -q .
}
sleep 120                      # let the queue start its first runs
while alive; do snap; sleep 1200; done
snap; echo "$(date '+%F %T') SNAP_DONE"
