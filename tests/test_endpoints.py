"""Integration tests for every FastAPI endpoint in main.py."""
import gzip
import json
from datetime import datetime
from pathlib import Path

import pytest

import main
from tests.conftest import (
    _make_compact,
    _make_compact_with_fever,
    _make_compact_with_person,
)


# ---------------------------------------------------------------------------
# Health check / dashboard
# ---------------------------------------------------------------------------

class TestHealthAndDashboard:
    def test_health_check_returns_200(self, client):
        r = client.get("/api/test")
        assert r.status_code == 200

    def test_health_check_body(self, client):
        body = client.get("/api/test").json()
        assert "server is running" in body["status"]
        assert "time" in body

    def test_dashboard_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Thermal Camera Dashboard" in r.text

    def test_dashboard_contains_poll_url(self, client):
        assert "/api/thermal/current/poll" in client.get("/").text


# ---------------------------------------------------------------------------
# POST /api/thermal
# ---------------------------------------------------------------------------

class TestPostThermal:
    def test_valid_compact_returns_success(self, client):
        r = client.post("/api/thermal", json=_make_compact(sensor_id="s1"))
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_valid_compact_received_count(self, client):
        payload = _make_compact(w=4, h=2, sensor_id="s1")
        body = client.post("/api/thermal", json=payload).json()
        assert body["received"] == 8  # 4×2 pixels

    def test_valid_compact_returns_occupancy(self, client):
        body = client.post("/api/thermal", json=_make_compact(sensor_id="s1")).json()
        assert "occupancy" in body
        assert isinstance(body["occupancy"], int)

    def test_compact_with_person_detects_occupancy(self, client):
        body = client.post("/api/thermal", json=_make_compact_with_person()).json()
        assert body["occupancy"] >= 1

    def test_compact_with_fever_sets_fever_flag(self, client):
        # Two consecutive fever frames needed to confirm fever
        payload = _make_compact_with_fever()
        client.post("/api/thermal", json=payload)
        body = client.post("/api/thermal", json=payload).json()
        assert body["any_fever"] is True

    def test_valid_expanded_format_accepted(self, client):
        pixels = [
            {"row": r, "col": c, "temp": 21.0, "r": 0, "g": 128, "b": 255}
            for r in range(2) for c in range(4)
        ]
        payload = {
            "width": 4, "height": 2,
            "min_temp": 21.0, "max_temp": 21.0,
            "pixels": pixels,
            "sensor_id": "s-expanded",
        }
        r = client.post("/api/thermal", json=payload)
        assert r.status_code == 200

    def test_missing_w_returns_400(self, client):
        d = _make_compact()
        del d["w"]
        assert client.post("/api/thermal", json=d).status_code == 400

    def test_length_mismatch_returns_400(self, client):
        d = _make_compact()
        d["t"] = d["t"][:-1]
        assert client.post("/api/thermal", json=d).status_code == 400

    def test_empty_body_returns_error(self, client):
        # FastAPI returns 422 for empty/missing request body
        r = client.post("/api/thermal", json={})
        assert r.status_code in (400, 422)

    def test_returns_frame_valid_flag(self, client):
        body = client.post("/api/thermal", json=_make_compact(sensor_id="s1")).json()
        assert "frame_valid" in body

    def test_fever_fields_present(self, client):
        body = client.post("/api/thermal", json=_make_compact(sensor_id="s1")).json()
        for field in ("fever_count", "elevated_count", "any_fever", "any_elevated"):
            assert field in body


# ---------------------------------------------------------------------------
# GET /api/thermal/current/poll
# ---------------------------------------------------------------------------

class TestPollThermal:
    def test_no_data_returns_404(self, client):
        assert client.get("/api/thermal/current/poll").status_code == 404

    def test_after_post_returns_200(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="s1"))
        assert client.get("/api/thermal/current/poll").status_code == 200

    def test_response_contains_pixels(self, client):
        client.post("/api/thermal", json=_make_compact(w=4, h=2, sensor_id="s1"))
        body = client.get("/api/thermal/current/poll").json()
        assert len(body["pixels"]) == 8

    def test_response_contains_people_clusters(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="s1"))
        body = client.get("/api/thermal/current/poll").json()
        assert "people_clusters" in body
        assert isinstance(body["people_clusters"], list)

    def test_response_contains_last_update(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="s1"))
        body = client.get("/api/thermal/current/poll").json()
        assert body["last_update"] is not None

    def test_filter_by_sensor_id(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-A"))
        body = client.get("/api/thermal/current/poll?sensor_id=cam-A").json()
        assert body["sensor_id"] == "cam-A"

    def test_unknown_sensor_id_returns_404(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-A"))
        assert client.get("/api/thermal/current/poll?sensor_id=cam-B").status_code == 404

    def test_people_clusters_populated_when_person_present(self, client):
        client.post("/api/thermal", json=_make_compact_with_person())
        body = client.get("/api/thermal/current/poll").json()
        assert len(body["people_clusters"]) >= 1
        cluster = body["people_clusters"][0]
        assert "center" in cluster
        assert "size" in cluster
        assert "representative_temp_c" in cluster


# ---------------------------------------------------------------------------
# GET /api/thermal/current/all and aliases
# ---------------------------------------------------------------------------

class TestThermalCurrentAll:
    def test_empty_returns_empty_dict(self, client):
        body = client.get("/api/thermal/current/all").json()
        assert body == {}

    def test_after_post_contains_sensor(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-1"))
        body = client.get("/api/thermal/current/all").json()
        assert "cam-1" in body

    def test_multiple_sensors(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-1"))
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-2"))
        body = client.get("/api/thermal/current/all").json()
        assert "cam-1" in body
        assert "cam-2" in body

    def test_predicted_all_alias(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-1"))
        all_body = client.get("/api/thermal/current/all").json()
        pred_body = client.get("/api/thermal/predicted/all").json()
        assert all_body == pred_body

    def test_predicted_poll_alias(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-1"))
        poll_body = client.get("/api/thermal/current/poll").json()
        pred_body = client.get("/api/thermal/predicted/poll").json()
        # Both should return the same pixels for the same sensor
        assert poll_body["pixels"] == pred_body["pixels"]


# ---------------------------------------------------------------------------
# GET /api/sensors
# ---------------------------------------------------------------------------

class TestSensors:
    def test_empty_returns_empty_list(self, client):
        body = client.get("/api/sensors").json()
        assert body["sensors"] == []
        assert body["count"] == 0

    def test_after_post_contains_sensor(self, client):
        client.post("/api/thermal", json=_make_compact(sensor_id="cam-X"))
        body = client.get("/api/sensors").json()
        assert "cam-X" in body["sensors"]
        assert body["count"] >= 1

    def test_deduplicates_same_sensor(self, client):
        for _ in range(3):
            client.post("/api/thermal", json=_make_compact(sensor_id="cam-X"))
        body = client.get("/api/sensors").json()
        assert body["sensors"].count("cam-X") == 1


# ---------------------------------------------------------------------------
# GET /api/thermal/history
# ---------------------------------------------------------------------------

class TestThermalHistory:
    def test_empty_data_dir_returns_empty(self, client):
        body = client.get("/api/thermal/history").json()
        assert body["count"] == 0

    def test_limit_capped_at_500(self, client):
        body = client.get("/api/thermal/history?limit=9999").json()
        # _safe_int clamps at 500
        assert body["limit"] == 500

    def test_pagination_fields_present(self, client):
        body = client.get("/api/thermal/history").json()
        assert "has_more" in body
        assert "next_offset" in body
        assert body["has_more"] is False
        assert body["next_offset"] is None

    def _write_compact_gz(self, data_dir: Path, fname: str, ts_iso: str, sensor_id: str, compact: dict) -> None:
        payload = {
            "timestamp": ts_iso,
            "format": "compact",
            "sensor_id": sensor_id,
            "data": compact,
        }
        raw = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        (data_dir / fname).write_bytes(raw)

    def test_has_more_false_when_page_fills_exact_total(self, client, tmp_path):
        """Two frames total, limit 2: no spurious next page."""
        c1 = _make_compact(w=2, h=2, sensor_id="s1")
        self._write_compact_gz(
            tmp_path,
            "thermal_s1_20260403_120000_000_compact.json.gz",
            "2026-04-03T12:00:00",
            "s1",
            c1,
        )
        c2 = _make_compact(w=2, h=2, base_temp=22.0, sensor_id="s1")
        self._write_compact_gz(
            tmp_path,
            "thermal_s1_20260403_110000_000_compact.json.gz",
            "2026-04-03T11:00:00",
            "s1",
            c2,
        )
        body = client.get("/api/thermal/history?limit=2").json()
        assert body["count"] == 2
        assert body["has_more"] is False
        assert body["next_offset"] is None

    def test_has_more_true_and_next_offset(self, client, tmp_path):
        """Three frames, limit 2: second page exists."""
        for suf, ts, base in [
            ("130000", "2026-04-03T13:00:00", 21.0),
            ("120000", "2026-04-03T12:00:00", 22.0),
            ("110000", "2026-04-03T11:00:00", 23.0),
        ]:
            c = _make_compact(w=2, h=2, base_temp=base, sensor_id="s1")
            self._write_compact_gz(
                tmp_path,
                f"thermal_s1_20260403_{suf}_000_compact.json.gz",
                ts,
                "s1",
                c,
            )
        p1 = client.get("/api/thermal/history?limit=2&offset=0").json()
        assert p1["count"] == 2
        assert p1["has_more"] is True
        assert p1["next_offset"] == 2
        p2 = client.get("/api/thermal/history?limit=2&offset=2").json()
        assert p2["count"] == 1
        assert p2["has_more"] is False
        assert p2["next_offset"] is None

    def test_include_data_expands_compact_and_sets_formats(self, client, tmp_path):
        c = _make_compact(w=2, h=2, sensor_id="s1")
        self._write_compact_gz(
            tmp_path,
            "thermal_s1_20260403_120000_000_compact.json.gz",
            "2026-04-03T12:00:00",
            "s1",
            c,
        )
        body = client.get("/api/thermal/history?include_data=true&limit=1").json()
        row = body["data"][0]
        assert row["source_format"] == "compact"
        assert row["returned_format"] == "expanded"
        assert row["format"] == "expanded"
        assert row["data"]["width"] == 2
        assert row["data"]["height"] == 2
        assert len(row["data"]["pixels"]) == 4


# ---------------------------------------------------------------------------
# GET /api/occupancy/history
# ---------------------------------------------------------------------------

class TestOccupancyHistory:
    def _write_occ_file(self, date_str, entries):
        path = main.DATA_DIR / f"occupancy_{date_str}.jsonl"
        path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

    def test_no_file_returns_404(self, client):
        assert client.get("/api/occupancy/history?date=20000101").status_code == 404

    def test_returns_entries(self, client):
        date_str = "20260403"
        self._write_occ_file(date_str, [
            {"timestamp": "2026-04-03T10:00:00", "sensor_id": "s1",
             "occupancy": 2, "room_temperature": 21.0},
        ])
        body = client.get(f"/api/occupancy/history?date={date_str}").json()
        assert body["count"] == 1
        assert body["data"][0]["occupancy"] == 2

    def test_filters_by_sensor_id(self, client):
        date_str = "20260403"
        self._write_occ_file(date_str, [
            {"timestamp": "2026-04-03T10:00:00", "sensor_id": "s1", "occupancy": 1},
            {"timestamp": "2026-04-03T10:00:01", "sensor_id": "s2", "occupancy": 3},
        ])
        body = client.get(f"/api/occupancy/history?date={date_str}&sensor_id=s1").json()
        assert body["count"] == 1
        assert body["data"][0]["sensor_id"] == "s1"

    def test_malformed_line_is_skipped(self, client):
        date_str = "20260403"
        path = main.DATA_DIR / f"occupancy_{date_str}.jsonl"
        path.write_text(
            '{"timestamp": "2026-04-03T10:00:00", "sensor_id": "s1", "occupancy": 1}\n'
            "THIS IS NOT JSON\n"
            '{"timestamp": "2026-04-03T10:00:01", "sensor_id": "s1", "occupancy": 2}\n'
        )
        body = client.get(f"/api/occupancy/history?date={date_str}").json()
        assert body["count"] == 2  # bad line skipped, 2 good entries returned


# ---------------------------------------------------------------------------
# GET /api/occupancy/stats
# ---------------------------------------------------------------------------

class TestOccupancyStats:
    def _write_occ_file(self, date_str, occupancies, sensor_id="s1"):
        entries = [
            {"timestamp": f"2026-04-03T10:{i:02d}:00", "sensor_id": sensor_id, "occupancy": v}
            for i, v in enumerate(occupancies)
        ]
        path = main.DATA_DIR / f"occupancy_{date_str}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    def test_no_file_returns_404(self, client):
        assert client.get("/api/occupancy/stats?date=20000101").status_code == 404

    def test_correct_stats(self, client):
        self._write_occ_file("20260403", [0, 1, 2, 3, 4])
        body = client.get("/api/occupancy/stats?date=20260403").json()
        assert body["min_occupancy"] == 0
        assert body["max_occupancy"] == 4
        assert body["avg_occupancy"] == pytest.approx(2.0)
        assert body["total_readings"] == 5
        assert body["current_occupancy"] == 4

    def test_malformed_line_is_skipped(self, client):
        path = main.DATA_DIR / "occupancy_20260403.jsonl"
        path.write_text(
            '{"sensor_id": "s1", "occupancy": 2}\n'
            "GARBAGE\n"
            '{"sensor_id": "s1", "occupancy": 4}\n'
        )
        body = client.get("/api/occupancy/stats?date=20260403").json()
        assert body["total_readings"] == 2

    def test_filter_by_sensor_id(self, client):
        entries = [
            {"timestamp": "2026-04-03T10:00:00", "sensor_id": "s1", "occupancy": 5},
            {"timestamp": "2026-04-03T10:00:01", "sensor_id": "s2", "occupancy": 1},
        ]
        path = main.DATA_DIR / "occupancy_20260403.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        body = client.get("/api/occupancy/stats?date=20260403&sensor_id=s1").json()
        assert body["total_readings"] == 1
        assert body["current_occupancy"] == 5


# ---------------------------------------------------------------------------
# GET /api/occupancy/trends
# ---------------------------------------------------------------------------

class TestOccupancyTrends:
    def _write_occ_file(self, date_str):
        entries = [
            {"timestamp": f"2026-04-03T{h:02d}:00:00", "sensor_id": "s1",
             "occupancy": h, "room_temperature": 21.0}
            for h in range(4)
        ]
        path = main.DATA_DIR / f"occupancy_{date_str}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    def test_returns_hour_buckets(self, client):
        self._write_occ_file("20260403")
        body = client.get("/api/occupancy/trends?date=20260403&bucket=hour").json()
        assert body["bucket"] == "hour"
        assert body["count"] == 4

    def test_returns_day_bucket(self, client):
        self._write_occ_file("20260403")
        body = client.get("/api/occupancy/trends?date=20260403&bucket=day").json()
        assert body["bucket"] == "day"
        assert body["count"] == 1

    def test_invalid_bucket_returns_400(self, client):
        assert client.get("/api/occupancy/trends?bucket=week").status_code == 400

    def test_no_file_returns_empty_buckets(self, client):
        body = client.get("/api/occupancy/trends?date=20000101&bucket=hour").json()
        assert body["count"] == 0

    def test_bucket_contains_expected_fields(self, client):
        self._write_occ_file("20260403")
        body = client.get("/api/occupancy/trends?date=20260403&bucket=hour").json()
        bucket = body["data"][0]
        assert "bucket_start" in bucket
        assert "bucket_end" in bucket
        assert "avg_occupancy" in bucket
        assert "max_occupancy" in bucket
        assert "sample_count" in bucket


# ---------------------------------------------------------------------------
# GET /api/occupancy/predict
# ---------------------------------------------------------------------------

class TestOccupancyPredict:
    def test_returns_200(self, client):
        assert client.get("/api/occupancy/predict").status_code == 200

    def test_default_24_hour_horizon(self, client):
        body = client.get("/api/occupancy/predict").json()
        assert body["count"] == 24

    def test_custom_horizon(self, client):
        body = client.get("/api/occupancy/predict?horizon_hours=6").json()
        assert body["count"] == 6

    def test_horizon_out_of_range_returns_422(self, client):
        assert client.get("/api/occupancy/predict?horizon_hours=0").status_code == 422
        assert client.get("/api/occupancy/predict?horizon_hours=49").status_code == 422

    def test_bucket_fields_present(self, client):
        body = client.get("/api/occupancy/predict?horizon_hours=1").json()
        bucket = body["data"][0]
        assert "bucket_start" in bucket
        assert "bucket_end" in bucket
        assert "expected_occupancy" in bucket


# ---------------------------------------------------------------------------
# _restore_state_from_disk (startup handler)
# ---------------------------------------------------------------------------

class TestRestoreStateFromDisk:
    def test_empty_dir_leaves_state_empty(self, tmp_path):
        main.DATA_DIR = tmp_path
        main._restore_state_from_disk()
        assert main.latest_thermal_data is None
        assert main.latest_thermal_by_sensor == {}

    def test_loads_latest_expanded_file(self, tmp_path):
        main.DATA_DIR = tmp_path
        pixels = [
            {"row": r, "col": c, "temp": 21.0, "r": 0, "g": 128, "b": 255}
            for r in range(2) for c in range(4)
        ]
        payload = {
            "timestamp": "2026-04-03T10:00:00",
            "format": "expanded",
            "sensor_id": "cam-1",
            "data": {
                "width": 4, "height": 2,
                "min_temp": 21.0, "max_temp": 21.0,
                "sensor_id": "cam-1",
                "pixels": pixels,
            },
        }
        (tmp_path / "thermal_cam1_20260403_100000_000_expanded.json").write_text(
            json.dumps(payload)
        )
        main._restore_state_from_disk()
        assert "cam-1" in main.latest_thermal_by_sensor
        assert main.latest_thermal_data is not None

    def test_loads_latest_occupancy_file(self, tmp_path):
        main.DATA_DIR = tmp_path
        entry = {
            "timestamp": "2026-04-03T10:00:00",
            "sensor_id": "cam-1",
            "occupancy": 3,
            "room_temperature": 21.5,
            "people_clusters": [],
            "fever_count": 0,
            "any_fever": False,
        }
        (tmp_path / "occupancy_20260403.jsonl").write_text(json.dumps(entry) + "\n")
        main._restore_state_from_disk()
        assert "cam-1" in main.latest_occupancy_by_sensor
        assert main.latest_occupancy_by_sensor["cam-1"]["occupancy"] == 3

    def test_loads_compact_gzip_file(self, tmp_path):
        main.DATA_DIR = tmp_path
        compact = _make_compact(w=2, h=2, sensor_id="gz-s")
        payload = {
            "timestamp": "2026-04-03T12:00:00",
            "format": "compact",
            "sensor_id": "gz-s",
            "data": compact,
        }
        raw = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        (tmp_path / "thermal_gzs_20260403_120000_000_compact.json.gz").write_bytes(raw)
        main._restore_state_from_disk()
        assert "gz-s" in main.latest_thermal_by_sensor
        assert main.latest_thermal_by_sensor["gz-s"]["width"] == 2
        assert len(main.latest_thermal_by_sensor["gz-s"]["pixels"]) == 4

    def test_picks_most_recent_expanded_file(self, tmp_path):
        main.DATA_DIR = tmp_path
        for ts, val in [("20260403_090000_000", 20.0), ("20260403_110000_000", 25.0)]:
            pixels = [{"row": 0, "col": 0, "temp": val, "r": 128, "g": 0, "b": 0}]
            payload = {
                "timestamp": f"2026-04-03T{'09' if '09' in ts else '11'}:00:00",
                "format": "expanded",
                "sensor_id": "cam-1",
                "data": {
                    "width": 1, "height": 1,
                    "min_temp": val, "max_temp": val,
                    "sensor_id": "cam-1",
                    "pixels": pixels,
                },
            }
            (tmp_path / f"thermal_cam1_{ts}_expanded.json").write_text(json.dumps(payload))
        main._restore_state_from_disk()
        # The newer file (11:00) should win
        loaded = main.latest_thermal_by_sensor["cam-1"]
        assert loaded["pixels"][0]["temp"] == pytest.approx(25.0)
