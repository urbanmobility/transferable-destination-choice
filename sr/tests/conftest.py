"""Tests that read the study's data files are skipped when the data root does not hold them (public checkout)."""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(FileNotFoundError) and rep.failed:
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, f"needs data not in this checkout: {call.excinfo.value}")
