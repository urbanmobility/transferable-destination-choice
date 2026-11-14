"""Paper figures, Task 2: shared style — sizes in mm, one palette, keyword-only text, editable exports."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import style


@pytest.fixture(autouse=True)
def _apply():
    style.apply()
    yield
    plt.close("all")


def test_figure_size_in_mm():
    fig = style.figure(183, 100)
    assert fig.get_size_inches() == pytest.approx((183 / 25.4, 100 / 25.4))
    assert style.W2 == 183 and style.W15 == 120 and style.MAX_H == 170
    with pytest.raises(ValueError):
        style.figure(183, 171)


def test_rc_is_publication_style():
    import matplotlib as mpl
    assert mpl.rcParams["pdf.fonttype"] == 42 and mpl.rcParams["svg.fonttype"] == "none"
    assert mpl.rcParams["font.sans-serif"][0] == "Arial" and mpl.rcParams["font.size"] == 7
    assert not mpl.rcParams["axes.spines.top"] and not mpl.rcParams["axes.spines.right"] and not mpl.rcParams["axes.grid"]


def test_orange_means_confirmation_or_threshold_only():
    orange = style.PALETTE["confirmation"]
    assert sorted(k for k, v in style.PALETTE.items() if v == orange) == ["confirmation", "threshold"]
    assert len(set(style.PALETTE.values())) == len(style.PALETTE) - 1
    assert set(style.MARKERS) == {"discovery", "heldout", "confirmation"}
    assert style.MARKERS["discovery"]["facecolor"] == "none" and style.MARKERS["confirmation"]["marker"] == "D"


def test_check_text_allows_keywords_numbers_and_math():
    fig = style.figure(89, 60)
    ax = fig.add_subplot()
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Distance elasticity")
    ax.text(0.5, 0.5, "26/32")
    ax.text(0.2, 0.2, r"$\eta = a_j^{\theta_4} - \theta_0 d^{\theta_1}/(N_j+\theta_2)^{\theta_3}$")
    ax.text(0.1, 0.1, "Two-regime power law")
    ax.text(0.3, 0.3, "slope −0.17 [−0.24, −0.11]")
    assert style.check_text(fig) == []


def test_check_text_catches_sentences_and_long_labels():
    fig = style.figure(89, 60)
    ax = fig.add_subplot()
    ax.set_title("Larger cities have steeper decay")
    ax.text(0.5, 0.5, "Share of the gravity gap")
    ax.text(0.1, 0.1, "Not supported.")
    bad = style.check_text(fig)
    assert bad == ["Larger cities have steeper decay", "Not supported.", "Share of the gravity gap"]


def test_panel_label():
    fig = style.figure(89, 60)
    ax = fig.add_subplot()
    t = style.panel_label(ax, "a")
    assert t.get_text() == "a" and t.get_fontweight() == "bold" and t.get_fontsize() == 8
    with pytest.raises(ValueError):
        style.panel_label(ax, "A")
    assert style.check_text(fig) == []


def test_save_writes_editable_vector_files_and_source_data(tmp_path):
    fig = style.figure(89, 60)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Distance (km)")
    out = style.save(fig, "figX", {"a": pd.DataFrame({"x": [0, 1], "y": [0, 1]})}, fig_dir=tmp_path / "f", src_dir=tmp_path / "s")
    assert {p.suffix for p in out} == {".pdf", ".svg", ".tiff", ".png"}
    pdf, svg = (tmp_path / "f" / "figX.pdf").read_bytes(), (tmp_path / "f" / "figX.svg").read_text()
    assert b"/Type3" not in pdf and b"Arial" in pdf
    assert "<text" in svg and "Arial" in svg and "Distance (km)" in svg
    assert pd.read_csv(tmp_path / "s" / "figX_a.csv").shape == (2, 2)
    from PIL import Image
    im = Image.open(tmp_path / "f" / "figX.tiff")
    assert im.size[0] == pytest.approx(89 / 25.4 * 600, abs=2)                # exact width: nothing is cropped


def test_save_refuses_sentences(tmp_path):
    fig = style.figure(89, 60)
    fig.add_subplot().set_title("This panel shows the main result")
    with pytest.raises(ValueError, match="text"):
        style.save(fig, "figY", {}, fig_dir=tmp_path, src_dir=tmp_path)


def test_size_colormap_runs_light_to_dark():
    cm = style.size_cmap()
    lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]  # noqa: E731
    v = [lum(cm(x)) for x in np.linspace(0, 1, 9)]
    assert all(a > b for a, b in zip(v, v[1:]))
