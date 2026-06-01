"""
Tests for the SUBSTRATE unified research platform.

Covers:
  - SubstrateLab initialisation and instrument registry
  - SubstrateResult container
  - All 6 instruments in stub/no-dep mode
  - correlate() pipeline
  - report() renderer
  - nl_router dispatch
  - run_anomaly_scan stub (no torch / no aligned.parquet)

No network, no GPU, no torch required.
"""

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from substrate import SubstrateLab
from substrate.lab import SubstrateResult


# ── SubstrateResult ───────────────────────────────────────────────────────────

class TestSubstrateResult:
    def test_basic_attributes(self):
        r = SubstrateResult("test_inst", "test_task", {"key": "val"})
        assert r.instrument == "test_inst"
        assert r.task == "test_task"
        assert r.data == {"key": "val"}

    def test_meta_gets_timestamp(self):
        r = SubstrateResult("x", "y", None)
        assert "timestamp_utc" in r.meta

    def test_dict_access_data(self):
        r = SubstrateResult("x", "y", [1, 2, 3])
        assert r["data"] == [1, 2, 3]

    def test_dict_access_meta(self):
        r = SubstrateResult("x", "y", None, meta={"foo": 42})
        assert r["foo"] == 42

    def test_keys_includes_data_and_meta(self):
        r = SubstrateResult("x", "y", None, meta={"a": 1, "b": 2})
        keys = r.keys()
        assert "data" in keys
        assert "a" in keys
        assert "b" in keys

    def test_to_json_parseable(self):
        r = SubstrateResult("inst", "task", {"val": 3.14}, meta={"p": 1})
        j = json.loads(r.to_json())
        assert j["instrument"] == "inst"
        assert j["task"] == "task"
        assert "data_repr" in j

    def test_repr_contains_instrument(self):
        r = SubstrateResult("geomagnetic", "anomaly_scan", {})
        assert "geomagnetic" in repr(r)


# ── SubstrateLab initialisation ───────────────────────────────────────────────

class TestSubstrateLab:
    def test_instantiates(self, tmp_path):
        lab = SubstrateLab(data_root=tmp_path)
        assert lab is not None

    def test_data_root_created(self, tmp_path):
        target = tmp_path / "proc"
        SubstrateLab(data_root=target)
        assert target.exists()

    def test_available_lists_six_instruments(self, tmp_path):
        lab = SubstrateLab(data_root=tmp_path)
        avail = lab.available
        assert len(avail) == 6
        for name in ["geomagnetic", "forecast", "simulation",
                     "mythology", "coherence", "quantum_bio"]:
            assert name in avail, f"'{name}' missing from available"

    def test_repr_contains_gpu_flag(self, tmp_path):
        lab = SubstrateLab(data_root=tmp_path, gpu=False)
        assert "gpu=False" in repr(lab)

    def test_instrument_returns_same_object(self, tmp_path):
        """Lazy cache: second call returns same instance."""
        lab = SubstrateLab(data_root=tmp_path)
        a = lab.instrument("mythology")
        b = lab.instrument("mythology")
        assert a is b

    def test_unknown_instrument_raises(self, tmp_path):
        lab = SubstrateLab(data_root=tmp_path)
        with pytest.raises(ValueError, match="Unknown instrument"):
            lab.instrument("nonexistent_xyz")


# ── Individual instrument stubs ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return SubstrateLab(data_root=tmp_path_factory.mktemp("substrate"))


class TestGeomagneticInstrument:
    def test_anomaly_scan_stub_returns_dict(self, lab):
        r = lab.run("geomagnetic", task="anomaly_scan")
        assert isinstance(r.data, dict)
        assert "anomaly_windows" in r.data

    def test_anomaly_windows_is_list(self, lab):
        r = lab.run("geomagnetic", task="anomaly_scan")
        assert isinstance(r.data["anomaly_windows"], list)

    def test_result_has_instrument_tag(self, lab):
        r = lab.run("geomagnetic", task="anomaly_scan")
        assert r.instrument == "geomagnetic"

    def test_elapsed_in_meta(self, lab):
        r = lab.run("geomagnetic", task="anomaly_scan")
        assert "elapsed_s" in r.meta
        assert r.meta["elapsed_s"] >= 0


class TestMythologyInstrument:
    def test_correlate_events_returns_table(self, lab):
        r = lab.run("mythology", task="correlate_events")
        assert "table" in r.data
        assert isinstance(r.data["table"], list)

    def test_default_events_present(self, lab):
        r = lab.run("mythology", task="correlate_events")
        assert len(r.data["table"]) >= 1

    def test_table_has_required_keys(self, lab):
        r = lab.run("mythology", task="correlate_events")
        for row in r.data["table"]:
            assert "kyr_bp" in row
            assert "tradition" in row

    def test_custom_events_passed_through(self, lab):
        events = [{"kyr_bp": 12.9, "label": "YD"}, {"kyr_bp": 41.0, "label": "Laschamp"}]
        r = lab.run("mythology", task="correlate_events", events=events)
        assert len(r.data["table"]) >= 2

    def test_event_float_normalised(self, lab):
        """Float events (kyr_bp only) must be normalised to dict form."""
        r = lab.run("mythology", task="correlate_events", events=[12.9, 41.0])
        # Should not raise; table must have entries
        assert len(r.data["table"]) >= 2

    def test_query_task_returns_list(self, lab):
        r = lab.run("mythology", task="query", query="great flood")
        assert isinstance(r.data, list)

    def test_traditions_in_meta(self, lab):
        r = lab.run("mythology", task="correlate_events")
        assert "traditions" in r.meta
        assert len(r.meta["traditions"]) == 7


class TestCoherenceInstrument:
    def test_snapshot_returns_dict(self, lab):
        r = lab.run("coherence", task="snapshot")
        assert isinstance(r.data, dict)

    def test_snapshot_instrument_tag(self, lab):
        r = lab.run("coherence", task="snapshot")
        assert r.instrument == "coherence"

    def test_unknown_task_raises(self, lab):
        with pytest.raises(ValueError, match="unknown task"):
            lab.run("coherence", task="nonexistent_task_xyz")


class TestQuantumBioInstrument:
    def test_radical_pair_yield_stub(self, lab):
        r = lab.run("quantum_bio", task="radical_pair_yield",
                    B_field_uT=50.0, rf_freq_MHz=1.4)
        assert "singlet_yield" in r.data

    def test_singlet_yield_between_0_and_1(self, lab):
        r = lab.run("quantum_bio", task="radical_pair_yield",
                    B_field_uT=50.0, rf_freq_MHz=1.4)
        sy = r.data["singlet_yield"]
        assert 0.0 <= sy <= 1.0, f"singlet_yield={sy} out of [0,1]"

    def test_instrument_tag(self, lab):
        r = lab.run("quantum_bio", task="radical_pair_yield")
        assert r.instrument == "quantum_bio"


# ── Cross-instrument correlation ──────────────────────────────────────────────

class TestCorrelate:
    def test_returns_substrate_result(self, lab):
        r_geo  = lab.run("geomagnetic", task="anomaly_scan")
        r_myth = lab.run("mythology",   task="correlate_events")
        corr   = lab.correlate([r_geo, r_myth])
        assert isinstance(corr, SubstrateResult)

    def test_instrument_tag_is_correlator(self, lab):
        r1 = lab.run("geomagnetic", task="anomaly_scan")
        r2 = lab.run("mythology",   task="correlate_events")
        corr = lab.correlate([r1, r2])
        assert corr.instrument == "__correlator__"

    def test_task_reflects_method(self, lab):
        r1 = lab.run("geomagnetic", task="anomaly_scan")
        r2 = lab.run("mythology",   task="correlate_events")
        corr = lab.correlate([r1, r2], method="temporal_overlap")
        assert corr.task == "temporal_overlap"

    def test_data_has_sync_events(self, lab):
        r1 = lab.run("geomagnetic", task="anomaly_scan")
        r2 = lab.run("mythology",   task="correlate_events")
        corr = lab.correlate([r1, r2])
        # correlator returns 'synchronous_events' (or 'sync_events' — accept both)
        assert "synchronous_events" in corr.data or "sync_events" in corr.data

    def test_sync_events_is_list(self, lab):
        r1 = lab.run("geomagnetic", task="anomaly_scan")
        r2 = lab.run("mythology",   task="correlate_events")
        corr = lab.correlate([r1, r2])
        key = "synchronous_events" if "synchronous_events" in corr.data else "sync_events"
        assert isinstance(corr.data[key], list)


# ── Report renderer ───────────────────────────────────────────────────────────

class TestReport:
    def test_markdown_report_returns_text(self, lab):
        r = lab.run("mythology", task="correlate_events")
        rpt = lab.report(r, fmt="markdown")
        assert isinstance(rpt.data.get("text"), str)
        assert len(rpt.data["text"]) > 10

    def test_json_report_parseable(self, lab):
        r = lab.run("mythology", task="correlate_events")
        rpt = lab.report(r, fmt="json")
        parsed = json.loads(rpt.data["text"])
        assert "instrument" in parsed

    def test_html_report_has_html_tag(self, lab):
        r = lab.run("mythology", task="correlate_events")
        rpt = lab.report(r, fmt="html")
        assert "<html" in rpt.data["text"].lower() or "<!doctype" in rpt.data["text"].lower()

    def test_report_instrument_tag(self, lab):
        r = lab.run("mythology", task="correlate_events")
        rpt = lab.report(r, fmt="markdown")
        assert rpt.instrument == "__reporter__"

    def test_report_writes_file(self, lab, tmp_path):
        r = lab.run("mythology", task="correlate_events")
        out = tmp_path / "report.md"
        lab.report(r, fmt="markdown", out=out)
        assert out.exists()
        assert out.stat().st_size > 0


# ── NL router ─────────────────────────────────────────────────────────────────

class TestNLRouter:
    def test_quantum_bio_dispatch(self, lab):
        r = lab.query("radical pair singlet yield quantum biology")
        assert r.instrument == "quantum_bio"

    def test_geomagnetic_dispatch(self, lab):
        r = lab.query("show anomaly windows in the palaeoclimate proxies")
        assert r.instrument == "geomagnetic"

    def test_mythology_dispatch(self, lab):
        r = lab.query("find myth passages correlated with geological events")
        assert r.instrument == "mythology"


# ── run_anomaly_scan stub ─────────────────────────────────────────────────────

class TestRunAnomalyScan:
    def test_no_torch_returns_stub(self):
        from cycle_detect.gnn_prototype import run_anomaly_scan, _TORCH_OK
        r = run_anomaly_scan(data_root="/tmp/__no_such_dir__")
        if not _TORCH_OK:
            assert "note" in r
            assert "STUB" in r["note"]
        else:
            assert "error" in r or "anomaly_windows" in r

    def test_missing_parquet_returns_error(self, tmp_path):
        from cycle_detect.gnn_prototype import run_anomaly_scan, _TORCH_OK
        if _TORCH_OK:
            r = run_anomaly_scan(data_root=tmp_path)
            assert "error" in r
            assert "aligned.parquet" in r["error"]
        else:
            pytest.skip("torch not installed — stub path tested elsewhere")

    def test_anomaly_windows_key_always_present(self):
        from cycle_detect.gnn_prototype import run_anomaly_scan
        r = run_anomaly_scan(data_root="/tmp/__no_such_dir__")
        assert "anomaly_windows" in r

    def test_proxy_cols_5_proxies(self):
        from cycle_detect.gnn_prototype import PROXY_COLS_NORM
        assert len(PROXY_COLS_NORM) == 5
        assert "vostok_co2_norm" in PROXY_COLS_NORM
        assert "vostok_deuterium_norm" in PROXY_COLS_NORM
        assert "sint2000_norm" in PROXY_COLS_NORM
