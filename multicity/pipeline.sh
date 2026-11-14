#!/bin/bash
# Multi-city data pipeline on the compute server (plan 1 Task 11). Runs the named stages in order, one log per stage, a
# <STAGE>_DONE / <STAGE>_FAILED line in results/multicity/logs/pipeline.log and a ledger entry per stage.
# Stages: shards homes trips poi attrs units qc osm units2 tiers form refs
# Usage: setsid nohup bash scripts/multicity/pipeline.sh STAGE [STAGE ...] > /dev/null 2>&1 < /dev/null & disown
set -u
export ODSR_ROOT=${ODSR_ROOT:?set ODSR_ROOT to the data root; this repository lives in $ODSR_ROOT/scripts}
export ODSR_LINE=${ODSR_LINE:-M}          # plan 7: M (2024), M_nw (2024 non-work), M19 (2019-04 non-work)
SUF=$([ "$ODSR_LINE" = M ] || echo "_$ODSR_LINE")   # log suffix of the other lines
if [ "$ODSR_LINE" != M ]; then   # plan 7 final review: only line-aware stages may run on another line
  for s in "$@"; do
    case $s in homes|trips|poi|units|qc|roles19init|roles19|refs) ;;
      *) echo "stage $s is not line-aware: refused on line $ODSR_LINE" >&2; exit 2 ;;
    esac
  done
fi
cd "$ODSR_ROOT"
ulimit -n 1048576
P=${ODSR_PY:-python}
M=scripts/multicity
L=results/multicity/logs
mkdir -p "$L"
CITIES_UNITS=$(grep -v '^\s*$' $M/config/cities_units.txt | paste -sd' ')
export ODSR_CITIES=$($P -I -c "import pandas as pd; print(','.join(pd.read_csv('$M/config/cities.csv', dtype=str)['city']))")

eligible() {  # cities with eligible == True in the QC table, optionally excluding a role
  $P -I - "$@" <<'EOF'
import sys, pandas as pd
sys.path.insert(0, "scripts/multicity")
import config
q = pd.read_csv(config.RESULTS / "qc/cities_qc.csv", dtype={"city": str})
c = pd.read_csv(config.CITIES_CSV, dtype=str).set_index("city")
ok = q[q["eligible"]]["city"]
if len(sys.argv) > 1:
    if "role" not in c:
        sys.exit(f"{config.CITIES_CSV.name} has no role column: run the tiers / roles stage first")
    ok = [x for x in ok if c.loc[x, "role"] != sys.argv[1]]
print(" ".join(ok))
EOF
}

states_of() {  # FIPS state codes of the cities given
  $P -I - "$@" <<'EOF'
import sys, pandas as pd
sys.path.insert(0, "scripts/multicity")
import config
c = pd.read_csv(config.CITIES_CSV, dtype=str).set_index("city")
print(" ".join(sorted({s for x in sys.argv[1:] for s in c.loc[x, "states"].split(";")})))
EOF
}

stage() {
  local name=$1; shift
  echo "$(date '+%F %T') $name start" >> $L/pipeline$SUF.log
  if "$@" > "$L/$name$SUF.log" 2>&1; then
    echo "$(date '+%F %T') ${name^^}_DONE" >> $L/pipeline$SUF.log
    $P -I -c "import sys; sys.path.insert(0, '$M'); import ledger; ledger.add('pipeline ($ODSR_LINE): stage $name done (log $L/$name$SUF.log)')"
  else
    echo "$(date '+%F %T') ${name^^}_FAILED" >> $L/pipeline$SUF.log
    $P -I -c "import sys; sys.path.insert(0, '$M'); import ledger; ledger.add('pipeline ($ODSR_LINE): stage $name FAILED (log $L/$name$SUF.log)')"
    return 1
  fi
}

for s in "$@"; do
  case $s in
    shards) stage shards $P -I -u $M/shards.py ;;
    homes)  stage homes  $P -I -u $M/homes.py ;;
    trips)  stage trips  $P -I -u $M/trips.py ;;
    poi)    stage poi    $P -I -u $M/poi.py ;;
    attrs)  stage attrs  $P -I -u $M/attrs.py build ;;
    units)  stage units  $P -I -u $M/units.py $CITIES_UNITS ;;
    qc)     stage qc     $P -I -u $M/qc.py $CITIES_UNITS ;;
    osm)    EL=$(eligible); stage osm_extract $P -I -u $M/osm_city.py extract $(states_of $EL) \
              && stage osm_features $P -I -u $M/osm_city.py features $EL ;;
    units2) stage units2 $P -I -u $M/units.py $(eligible) && stage qc2 $P -I -u $M/qc.py $CITIES_UNITS ;;
    tiers)  stage tiers  $P -I -u $M/tiers.py ;;
    roles19init) stage roles19init $P -I -u $M/roles_m19.py init ;;
    roles19)     stage roles19     $P -I -u $M/roles_m19.py assign ;;
    form)   stage form   $P -I -u $M/form_metrics.py $(eligible) ;;
    refs)   EL=$(eligible confirm) || exit 1; stage refs $P -I -u $M/refs.py $EL ;;
    *) echo "unknown stage $s" >> $L/pipeline$SUF.log; exit 2 ;;
  esac || exit 1
done
echo "$(date '+%F %T') PIPELINE_DONE $*" >> $L/pipeline$SUF.log
