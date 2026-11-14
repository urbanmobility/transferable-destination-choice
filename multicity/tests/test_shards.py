"""Task 3: parquet shard integrity (magic bytes) and per-day shard presence."""
import pyarrow as pa
import pyarrow.parquet as pq


def test_check_flags_bad_and_missing(tmp_path):
    from shards import check
    v = tmp_path / "visits"
    v.mkdir()
    pq.write_table(pa.table({"a": [1]}), v / "2024-03-01--part-00000.snappy.parquet")
    (v / "2024-03-02--part-00000.snappy.parquet").write_bytes(b"Upstream error 503\n")
    r = check(tmp_path, ("visits",), ("2024-03",))
    assert r["n_files"] == 2 and len(r["bad"]) == 1 and r["bad"][0].endswith("03-02--part-00000.snappy.parquet")
    assert r["per_day"]["visits"] == {"2024-03-01": 1, "2024-03-02": 1}
    assert "2024-03-03" in r["days_missing"]["visits"] and len(r["days_missing"]["visits"]) == 29


def test_monthly_table_keyed_by_month(tmp_path):
    from shards import check
    h = tmp_path / "home-location"
    h.mkdir()
    pq.write_table(pa.table({"a": [1]}), h / "2024-03--data_x_0.snappy.parquet")
    r = check(tmp_path, ("home-location",), ("2024-03", "2024-04"))
    assert r["per_day"]["home-location"] == {"2024-03": 1}
    assert r["days_missing"]["home-location"] == ["2024-04"] and r["bad"] == []
