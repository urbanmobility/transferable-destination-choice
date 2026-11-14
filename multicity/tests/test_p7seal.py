"""Plan 7 Task 2: the sign-off guard. Nothing on the sealed data lines (M19, M_nw) and no new outcome or covariate of a
law-test city is computed before sign-off of results/multicity/laws/registered_p7.json."""
import json

import pytest

import p7seal

POOL_ENTRY = {"cid": "c0001", "equation": "f = log(#6 + 1.0); p1 = []; p2 = []", "template": None, "variant": "A",
              "complexity": 5}


@pytest.fixture
def sealed_p7(tmp_path, monkeypatch):
    monkeypatch.setattr(p7seal, "REG", tmp_path / "registered_p7.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "P7_SIGNED")
    p7seal.REG.write_text(json.dumps([{"id": "P7-01", "outcome": "ed_5_10km", "covariate": "circuity"}]))
    return tmp_path


def test_paths_do_not_follow_the_data_line():
    import sr_data
    assert p7seal.REG == sr_data.ROOT / "results/multicity/laws/registered_p7.json"
    assert p7seal.SIGNED == sr_data.ROOT / "results/multicity/laws/P7_SIGNED"
    assert p7seal.SEALED_LINES == ("M19", "M_nw")


def test_guards_refuse_before_signing(sealed_p7, monkeypatch):
    monkeypatch.setattr(p7seal.config, "cities_with_role", lambda r: ["10900"] if r == "lawtest" else [])
    assert not p7seal.is_signed()
    with pytest.raises(RuntimeError):
        p7seal.require_signed()
    for line in ("M19", "M_nw"):
        with pytest.raises(RuntimeError):
            p7seal.guard_line(line)
    p7seal.guard_line("M")
    with pytest.raises(RuntimeError):
        p7seal.guard_cities(["12060", "10900"])
    p7seal.guard_cities(["12060"])


def test_sign_then_tamper(sealed_p7, monkeypatch):
    monkeypatch.setattr(p7seal.config, "cities_with_role", lambda r: ["10900"] if r == "lawtest" else [])
    lines = []
    sha = p7seal.sign(log=lines.append)
    assert len(sha) == 64 and p7seal.SIGNED.read_text().strip() == sha and p7seal.is_signed()
    assert len(lines) == 1 and sha[:12] in lines[0]
    assert [h["id"] for h in p7seal.require_signed()] == ["P7-01"]
    p7seal.guard_line("M19")
    p7seal.guard_cities(["10900"])
    with pytest.raises(FileExistsError):
        p7seal.sign(log=lambda s: None)
    p7seal.REG.write_text(p7seal.REG.read_text().replace("P7-01", "P7-02"))
    with pytest.raises(RuntimeError):
        p7seal.require_signed()
    with pytest.raises(RuntimeError):
        p7seal.guard_line("M19")


def test_sign_needs_a_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(p7seal, "REG", tmp_path / "registered_p7.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "P7_SIGNED")
    with pytest.raises(FileNotFoundError):
        p7seal.sign(log=lambda s: None)
    assert not p7seal.SIGNED.exists()


def test_fit_library_and_banded_refuse_a_sealed_line(sealed_p7, tmp_path, monkeypatch):
    import banded
    import fit_library
    import sr_data
    monkeypatch.setattr(sr_data, "LINE", "M19")
    with pytest.raises(RuntimeError):
        fit_library.main(cities=["12060"], cands=[POOL_ENTRY], out_dir=tmp_path / "o")
    with pytest.raises(RuntimeError):
        banded.city_banded("12060")
    assert not (tmp_path / "o").exists()


def test_guard_cities_reads_the_M_line_roles_and_refuses_sealed_lines(sealed_p7, monkeypatch):
    import sr_data
    p7seal.guard_cities(["12060"])
    with pytest.raises(RuntimeError):
        p7seal.guard_cities(["10900"])                    # a law-test city in config/cities.csv, whatever the line
    monkeypatch.setattr(sr_data, "LINE", "M19")
    monkeypatch.setattr(p7seal.config, "cities_with_role", lambda r: [])     # the M19 table has no lawtest role
    with pytest.raises(RuntimeError):
        p7seal.guard_cities(["12060"])                    # nothing guarded runs on a sealed line before signing
    with pytest.raises(RuntimeError):
        p7seal.guard_cities(["10900"])
