"""Plan 2 Task 2: the 12-run search queue for the compute server and the queue / snapshot scripts (static checks)."""
import subprocess
from pathlib import Path

Q = Path(__file__).resolve().parents[1] / "config/queue_search.txt"
SEARCH = ["35620", "16980", "26420", "31080", "12060", "BAY9", "14460", "42660"]


def test_queue_lines():
    rows = [l.split() for l in Q.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(rows) == 12 and all(len(r) == 6 for r in rows)
    pooled = [r for r in rows if r[2] == "city8"]
    assert sorted((r[1], r[3]) for r in pooled) == [("A", "0"), ("A", "1"), ("A", "2"), ("B", "0")]
    assert all(r[5] == "50" and r[4] == "40" for r in pooled)
    singles = [r for r in rows if r[2].startswith("single:")]
    assert sorted(r[2].split(":")[1] for r in singles) == sorted(SEARCH)
    assert all(r[1] == "A" and r[3] == "0" and r[5] == "100" and r[4] == "40" for r in singles)
    assert rows[:4] == pooled                                     # pooled runs are queued first
    tags = {f"{r[0]}_{r[1]}_{r[2].replace(':', '-')}_s{r[3]}" for r in rows}
    assert len(tags) == 12 and all(r[0] == "tract" for r in rows)


def test_queue_header_records_the_environment():
    head = "\n".join(l for l in Q.read_text().splitlines() if l.startswith("#"))
    for s in ("SR_NEG_RATE=0.1", "SR_TURBO=1", "SR_DROP_VARS=s_health", "43200", "28800"):
        assert s in head, s


def test_scripts_parse():
    for s in ("run_queue.sh", "snap_hof.sh"):
        assert subprocess.run(["bash", "-n", str(Q.parents[1] / s)]).returncode == 0
    rq = (Q.parents[1] / "run_queue.sh").read_text()
    assert "SR_REGIONS" in rq and "SR_DROP_VARS=s_health" in rq and "results/sr/M/logs" in rq and "QUEUE_DONE" in rq
    assert "43200" in rq and "28800" in rq
    sh = (Q.parents[1] / "snap_hof.sh").read_text()
    assert "results/sr/M/runs" in sh and "SNAP_DONE" in sh and "run_sr.py" in sh


def test_queue_r2_seeded_and_parallel():
    q2 = Q.parent / "queue_search_r2.txt"
    rows = [l.split() for l in q2.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(rows) == 12 and all(len(r) == 6 for r in rows)
    pooled = [r for r in rows if r[2] == "city8"]
    assert sorted((r[1], r[3]) for r in pooled) == [("A", "10"), ("A", "11"), ("A", "12"), ("B", "10")]
    singles = [r for r in rows if r[2].startswith("single:")]
    assert sorted(r[2].split(":")[1] for r in singles) == sorted(SEARCH) and all(r[3] == "10" and r[5] == "100" for r in singles)
    head = "\n".join(l for l in q2.read_text().splitlines() if l.startswith("#"))
    assert "guesses.json" in head and "MAXP 12" in head and "43200" in head and "28800" in head
    assert "SR_GUESSES" in (Q.parents[1] / "run_queue.sh").read_text()


def test_queue_r3_and_per_run_seed_files():
    q3 = Q.parent / "queue_search_r3.txt"
    rows = [l.split() for l in q3.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(rows) == 12 and sorted({r[3] for r in rows}) == ["20", "21", "22"]
    assert sorted(r[2].split(":")[1] for r in rows if r[2].startswith("single:")) == sorted(SEARCH)
    rq = (Q.parents[1] / "run_queue.sh").read_text()
    assert "config/guesses/$tag.json" in rq and 'export SR_GUESSES="$g"' in rq


def test_queue_r4_one_structural_class_per_single_run():
    """Round 4 (11-04): single-city runs only, seeds 40/41/42 = one structural class each (seed_agent.FAMILY_SEEDS)."""
    q4 = Q.parent / "queue_search_r4.txt"
    rows = [l.split() for l in q4.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(rows) == 24 and all(r[2].startswith("single:") and r[1] == "A" and r[4] == "40" and r[5] == "100" for r in rows)
    assert sorted((r[2].split(":")[1], r[3]) for r in rows) == sorted((c, s) for c in SEARCH for s in ("40", "41", "42"))
    head = "\n".join(l for l in q4.read_text().splitlines() if l.startswith("#"))
    for s in ("opportunity", "gravity_opportunity", "kernel_other", "MAXP 24", "SR_THREADS=5", "10800"):
        assert s in head, s
    rq = (Q.parents[1] / "run_queue.sh").read_text()
    assert "SR_TIMEOUT_SINGLE=${SR_TIMEOUT_SINGLE:-28800}" in rq and "SR_TIMEOUT_POOLED=${SR_TIMEOUT_POOLED:-43200}" in rq


def test_run_queue_passes_sr_block():
    """Plan 4 Task 2: the destination block of a search is SR_BLOCK (default search = folds 2-4; the agent loop's
    rounds use search23), exported by run_queue.sh and recorded by run_sr.py in config.json."""
    rq = (Q.parents[1] / "run_queue.sh").read_text()
    assert "export SR_BLOCK=${SR_BLOCK:-search}" in rq and "block=$SR_BLOCK" in rq
    rs = (Q.parents[2] / "sr/run_sr.py").read_text()
    assert 'os.environ.get("SR_BLOCK", "search")' in rs and '"block": dest_block' in rs


def test_run_queue_logs_the_real_exit_status():
    """11-06: `echo "$(date ...) end $tag rc $?"` printed date's status (always 0) — the run's status must be
    captured before the echo."""
    rq = (Q.parents[1] / "run_queue.sh").read_text()
    assert 'rc=$?;' in rq and 'end $tag rc $rc' in rq and 'rc $?"' not in rq
