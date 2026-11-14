"""Plan 2 Task 8: deterministic orchestrator — stage DAG with input / output hashes in the ledger, skipping of
finished stages, stop on failure, GATE 1 gate, dry run, CLI parsing."""
import importlib

import pytest


def _stages(tmp_path, orch, fail=None, gate_s3=None):
    a, b, c = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.txt"

    def mk(name, inputs, outputs, text, needs_gate=None):
        def run():
            if fail == name:
                raise RuntimeError("boom")
            for o in outputs:
                o.write_text(text)
        return orch.Stage(name, inputs, outputs, run, needs_gate=needs_gate)
    return [mk("s1", [], [a], "A"), mk("s2", [a], [b], "B"), mk("s3", [b], [c], "C", needs_gate=gate_s3)]


@pytest.fixture
def orch():
    import orchestrate
    return importlib.reload(orchestrate)


def test_run_records_hashes_and_skips_finished(tmp_path, orch):
    led = tmp_path / "LEDGER.md"
    st = _stages(tmp_path, orch)
    orch.run(["s1", "s2"], stages=st, ledger_file=led)
    text = led.read_text()
    assert text.count("stage s1 done") == 1 and "stage s2 done" in text and "stage s1 start" in text
    assert "sha" in text and (tmp_path / "b.txt").read_text() == "B" and not (tmp_path / "c.txt").exists()
    orch.run(["s1", "s2", "s3"], stages=st, ledger_file=led)
    text = led.read_text()
    assert text.count("stage s1 done") == 1 and "stage s1 skipped" in text and "stage s3 done" in text
    (tmp_path / "a.txt").write_text("changed")                  # an output changed on disk: the stage runs again
    orch.run(["s1"], stages=st, ledger_file=led)
    assert led.read_text().count("stage s1 done") == 2 and (tmp_path / "a.txt").read_text() == "A"


def test_failure_stops_the_chain(tmp_path, orch):
    led = tmp_path / "LEDGER.md"
    st = _stages(tmp_path, orch, fail="s2")
    with pytest.raises(orch.StageError):
        orch.run(["s1", "s2", "s3"], stages=st, ledger_file=led)
    text = led.read_text()
    assert "stage s2 FAILED" in text and "boom" in text and "s3" not in text and not (tmp_path / "c.txt").exists()


def test_missing_input_fails_before_running(tmp_path, orch):
    led = tmp_path / "LEDGER.md"
    st = _stages(tmp_path, orch)
    with pytest.raises(orch.StageError):
        orch.run(["s2"], stages=st, ledger_file=led)
    assert "missing input" in led.read_text() and not (tmp_path / "b.txt").exists()
    with pytest.raises(orch.StageError):
        orch.run(["nope"], stages=st, ledger_file=led)


def test_gate_dry_run_and_unknown_stage(tmp_path, orch):
    led = tmp_path / "LEDGER.md"
    assert orch.gate_approved(1, led) is False
    st = _stages(tmp_path, orch, gate_s3=1)
    orch.run(["s1", "s2"], stages=st, ledger_file=led)
    with pytest.raises(orch.StageError):
        orch.run(["s3"], stages=st, ledger_file=led)             # GATE 1 not approved
    assert not (tmp_path / "c.txt").exists()
    orch.approve(1, led, note="approved")
    assert orch.gate_approved(1, led) is True and "GATE 1: 批准" in led.read_text()
    orch.run(["s3"], stages=st, ledger_file=led)
    assert (tmp_path / "c.txt").read_text() == "C"
    n = len(led.read_text().splitlines())
    (tmp_path / "c.txt").unlink()
    orch.run(["s3"], stages=st, ledger_file=led, dry_run=True)
    assert not (tmp_path / "c.txt").exists() and len(led.read_text().splitlines()) == n


def test_real_stage_list_and_cli(orch):
    assert [s.name for s in orch.STAGES] == ["search_wait", "select_pooled", "select_single", "library", "clean",
                                             "llm_simplify", "llm_readability", "merge", "fit_library", "gate1"]
    assert all(callable(s.run) for s in orch.STAGES) and orch.STAGES[-1].name == "gate1"
    assert orch.cli_parse(["run", "library", "clean", "--dry-run"]) == ("run", ["library", "clean"], {"dry_run": True, "approve": False})
    assert orch.cli_parse(["gate", "1", "--approve"]) == ("gate", ["1"], {"dry_run": False, "approve": True})
    assert orch.cli_parse(["run"]) == ("run", [], {"dry_run": False, "approve": False})


def test_run_all_runs_concurrently_and_fails_on_any(orch, tmp_path, monkeypatch):
    import sys
    import time
    monkeypatch.setattr(orch, "SRM", tmp_path)
    monkeypatch.setattr(orch, "PY", sys.executable)
    ok = tmp_path / "ok.py"
    ok.write_text("import time, sys; time.sleep(1); print('done', sys.argv[1:])\n")
    bad = tmp_path / "bad.py"
    bad.write_text("raise SystemExit(3)\n")
    t0 = time.time()
    orch._run_all([("a", ok, ["1"], {}), ("b", ok, ["2"], {}), ("c", ok, ["3"], {})])
    assert time.time() - t0 < 2.5 and "done ['2']" in (tmp_path / "logs/b.log").read_text()
    with pytest.raises(RuntimeError, match="bad"):
        orch._run_all([("a", ok, ["1"], {}), ("bad", bad, [], {})])


def test_dry_run_lists_missing_inputs_without_failing_or_ledgering(tmp_path, orch, capsys):
    """A dry run over the whole chain must show every stage even though later stages' inputs come from earlier
    stages of the same run, and must write nothing to the ledger."""
    led = tmp_path / "LEDGER.md"
    st = _stages(tmp_path, orch)
    orch.run(["s1", "s2", "s3"], stages=st, ledger_file=led, dry_run=True)
    out = capsys.readouterr().out
    assert "stage s2" in out and "stage s3" in out and "missing" in out
    assert not led.exists() and not (tmp_path / "a.txt").exists()


def test_stages_read_the_run_tags_from_the_queue_files(orch):
    """Rounds 2-4 (11-04): the selections pool the pooled fronts of all rounds; the single-city selections pool the
    single runs of rounds 2-4; search_wait watches the round-3 and round-4 logs; the library adds the seed agent's files
    and the diagnostic challengers (those that exist)."""
    by = {s.name: s for s in orch.STAGES}
    pooled = {p.parent.name for p in by["select_pooled"].inputs}
    assert pooled == {f"tract_A_city8_s{k}" for k in (0, 1, 2, 10, 11, 12, 20, 21, 22)} | {"tract_B_city8_s0", "tract_B_city8_s10", "tract_B_city8_s20"}
    singles = [p.parent.name for p in by["select_single"].inputs]
    assert len(singles) == 40 and all("single-" in n and n[-4:] in ("_s10", "_s20", "_s40", "_s41", "_s42") for n in singles)
    assert orch.single_runs("BAY9") == [orch.SRM / f"runs/tract_A_single-BAY9_s{k}" for k in (10, 20, 40, 41, 42)]
    assert orch.pooled_runs("B") == [orch.SRM / f"runs/tract_B_city8_s{k}" for k in (0, 10, 20)]
    assert [p.name for p in by["search_wait"].outputs] == ["queue_r3.log", "snap_hof_r3.log", "queue_r4.log", "snap_hof_r4.log"]
    assert [p.name for p in orch.EXTRA_SELECT] == ["select_seeds_agent.json", "select_seeds_agent_r4.json", "select_challengers_diag.json"]


def test_search_wait_needs_every_round_done(orch, tmp_path, monkeypatch):
    logs = [(tmp_path / "q3", tmp_path / "s3"), (tmp_path / "q4", tmp_path / "s4")]
    monkeypatch.setattr(orch, "WAIT_LOGS", logs)
    for q, s in logs:
        q.write_text("x QUEUE_DONE\n")
        s.write_text("x SNAP_DONE\n")
    assert orch._search_done()
    logs[1][1].write_text("still running\n")
    assert not orch._search_done()
