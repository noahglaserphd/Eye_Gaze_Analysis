from __future__ import annotations

import pandas as pd
import pytest

import core.pipeline_runner as pr


def _minimal_gaze_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [0.0, 0.05, 0.10, 0.15],
            "gaze_x_norm": [0.4, 0.401, 0.399, 0.4],
            "gaze_y_norm": [0.5, 0.501, 0.499, 0.5],
        }
    )


@pytest.fixture
def patched_dirs(monkeypatch, tmp_path):
    root = tmp_path
    vdir = root / "videos"
    gdir = root / "gaze_samples"
    fdir = root / "fixations"
    sdir = root / "saccades"
    mdir = root / "metrics"

    def ensure_tmp() -> None:
        for d in (
            vdir,
            gdir,
            fdir,
            sdir,
            mdir,
            root / "time_windows",
            root / "figures",
            root / "aoi",
            root / "cache_clips",
            root / "exports",
        ):
            d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pr, "ensure_project_dirs", ensure_tmp)
    monkeypatch.setattr(pr, "ROOT", root)
    monkeypatch.setattr(pr, "VIDEO_DIR", vdir)
    monkeypatch.setattr(pr, "GAZE_DIR", gdir)
    monkeypatch.setattr(pr, "FIX_DIR", fdir)
    monkeypatch.setattr(pr, "SAC_DIR", sdir)
    monkeypatch.setattr(pr, "METRICS_DIR", mdir)
    ensure_tmp()
    return root, vdir, gdir, fdir, sdir


def test_run_analyze_gaze_paths_processes_multiple_csvs(patched_dirs):
    _, _, gdir, fdir, _sdir = patched_dirs
    df = _minimal_gaze_df()
    df.to_csv(gdir / "a.csv", index=False)
    df.to_csv(gdir / "b.csv", index=False)

    lines = pr.run_analyze_gaze_paths([gdir / "a.csv", gdir / "b.csv"], log=None)
    assert (fdir / "a_fixations.csv").is_file()
    assert (fdir / "b_fixations.csv").is_file()
    assert any("a:" in ln for ln in lines)
    assert any("b:" in ln for ln in lines)


def test_extract_from_videos_keeps_standalone_gaze_csv(monkeypatch, patched_dirs):
    _, vdir, gdir, fdir, _sdir = patched_dirs
    _minimal_gaze_df().to_csv(gdir / "standalone.csv", index=False)
    (fdir / "standalone_fixations.csv").write_text("file,placeholder\n", encoding="utf-8")

    (vdir / "vid.mp4").write_bytes(b"")

    def fake_extract(_vp, out_csv):
        d = _minimal_gaze_df()
        d.to_csv(out_csv, index=False)
        return d

    monkeypatch.setattr(pr, "extract_gaze_from_video", fake_extract)

    pr.run_extract_from_videos(log=None)

    assert (gdir / "standalone.csv").is_file(), "CSV-only gaze must not be deleted when videos/ is processed"
    assert (fdir / "standalone_fixations.csv").is_file(), "Fixations for CSV-only sessions must be preserved"
    assert (gdir / "vid.csv").is_file()


def test_full_pipeline_runs_csv_only_after_videos(monkeypatch, patched_dirs):
    _, vdir, gdir, fdir, sdir = patched_dirs
    (vdir / "vid.mp4").write_bytes(b"")

    def fake_extract(_vp, out_csv):
        d = _minimal_gaze_df()
        d.to_csv(out_csv, index=False)
        return d

    monkeypatch.setattr(pr, "extract_gaze_from_video", fake_extract)

    _minimal_gaze_df().to_csv(gdir / "extra_session.csv", index=False)

    monkeypatch.setattr(pr, "run_time_windows", lambda log=None: [])
    monkeypatch.setattr(pr, "run_session_figures", lambda log=None: [])
    monkeypatch.setattr(pr, "run_aggregate_figure", lambda log=None: [])

    lines = pr.run_full_pipeline(log=None)

    assert (fdir / "extra_session_fixations.csv").is_file()
    assert (sdir / "extra_session_saccades.csv").is_file()
    assert any("Also processing" in ln for ln in lines)
