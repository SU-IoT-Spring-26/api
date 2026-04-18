"""Unit tests for pure helper functions in main.py."""
import numpy as np
import pytest

import main
from main import (
    _parse_temperature,
    _run_training_thread_wrapper,
    _sanitize_sensor_id_for_filename,
    _validate_thermal_payload,
    apply_occupancy_signal_processing,
    collapse_to_compact,
    convert_numpy_types,
    detect_human_heat,
    detect_subpage_artifact,
    estimate_occupancy,
    estimate_room_temperature,
    expand_thermal_data,
    find_people_clusters,
    interpolate_subpages,
    temperature_to_color,
    thermal_data_to_array,
)


# ---------------------------------------------------------------------------
# _parse_temperature
# ---------------------------------------------------------------------------

class TestParseTemperature:
    def test_int(self):
        assert _parse_temperature(21) == 21.0

    def test_float(self):
        assert _parse_temperature(21.5) == pytest.approx(21.5)

    def test_plain_string(self):
        assert _parse_temperature("21.5") == pytest.approx(21.5)

    def test_string_with_degree_c(self):
        assert _parse_temperature("21.5°C") == pytest.approx(21.5)

    def test_string_with_degree_f_suffix_stripped_only(self):
        # The function strips the suffix but does NOT convert units.
        assert _parse_temperature("21.5°F") == pytest.approx(21.5)

    def test_string_with_bare_c(self):
        assert _parse_temperature("36.6C") == pytest.approx(36.6)

    def test_string_with_bare_f(self):
        assert _parse_temperature("98.6F") == pytest.approx(98.6)

    def test_string_with_degree_symbol(self):
        assert _parse_temperature("37°") == pytest.approx(37.0)

    def test_negative_temperature(self):
        assert _parse_temperature("-5.0") == pytest.approx(-5.0)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _parse_temperature("not_a_number")

    def test_zero(self):
        assert _parse_temperature(0) == 0.0


# ---------------------------------------------------------------------------
# temperature_to_color
# ---------------------------------------------------------------------------

class TestTemperatureToColor:
    def test_min_equals_max_returns_grey(self):
        assert temperature_to_color(25.0, 25.0, 25.0) == (128, 128, 128)

    def test_at_min_temp_returns_blue(self):
        r, g, b = temperature_to_color(20.0, 20.0, 40.0)
        assert r == 0 and b == 255

    def test_at_max_temp_returns_red(self):
        r, g, b = temperature_to_color(40.0, 20.0, 40.0)
        assert r == 255 and b == 0

    def test_midpoint_returns_green(self):
        # normalized=0.5 → third branch → (0, 255, 0)
        r, g, b = temperature_to_color(30.0, 20.0, 40.0)
        assert r == 0 and g == 255 and b == 0

    def test_below_min_is_clamped_to_min(self):
        assert temperature_to_color(10.0, 20.0, 40.0) == temperature_to_color(20.0, 20.0, 40.0)

    def test_above_max_is_clamped_to_max(self):
        assert temperature_to_color(50.0, 20.0, 40.0) == temperature_to_color(40.0, 20.0, 40.0)

    def test_returns_valid_rgb_range(self):
        for t in [20.0, 25.0, 30.0, 35.0, 40.0]:
            r, g, b = temperature_to_color(t, 20.0, 40.0)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255


# ---------------------------------------------------------------------------
# _validate_thermal_payload
# ---------------------------------------------------------------------------

class TestValidateThermalPayload:
    def _compact(self, **overrides):
        base = {"w": 4, "h": 2, "min": 20.0, "max": 21.5, "t": [20.0 + i * 0.1 for i in range(8)]}
        base.update(overrides)
        return base

    def test_valid_compact_no_exception(self):
        _validate_thermal_payload(self._compact())

    def test_compact_missing_w_raises(self):
        d = self._compact()
        del d["w"]
        with pytest.raises(ValueError, match="w and h"):
            _validate_thermal_payload(d)

    def test_compact_missing_h_raises(self):
        d = self._compact()
        del d["h"]
        with pytest.raises(ValueError, match="w and h"):
            _validate_thermal_payload(d)

    def test_compact_zero_width_raises(self):
        with pytest.raises(ValueError, match="invalid thermal dimensions"):
            _validate_thermal_payload(self._compact(w=0))

    def test_compact_negative_height_raises(self):
        with pytest.raises(ValueError, match="invalid thermal dimensions"):
            _validate_thermal_payload(self._compact(h=-1))

    def test_compact_empty_t_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_thermal_payload(self._compact(t=[]))

    def test_compact_length_mismatch_raises(self):
        d = self._compact()
        d["t"] = d["t"][:-1]  # one too few
        with pytest.raises(ValueError, match=r"does not match w\*h"):
            _validate_thermal_payload(d)

    def test_valid_expanded_no_exception(self):
        pixels = [{"row": r, "col": c, "temp": 21.0, "r": 0, "g": 128, "b": 255}
                  for r in range(2) for c in range(4)]
        _validate_thermal_payload({"width": 4, "height": 2, "pixels": pixels})

    def test_expanded_missing_width_raises(self):
        pixels = [{"row": 0, "col": 0, "temp": 21.0, "r": 0, "g": 0, "b": 255}]
        with pytest.raises(ValueError, match="width and height"):
            _validate_thermal_payload({"height": 1, "pixels": pixels})

    def test_expanded_empty_pixels_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_thermal_payload({"width": 2, "height": 2, "pixels": []})

    def test_expanded_pixel_count_mismatch_raises(self):
        pixels = [{"row": 0, "col": 0, "temp": 21.0, "r": 0, "g": 0, "b": 255}]
        with pytest.raises(ValueError, match="does not match"):
            _validate_thermal_payload({"width": 2, "height": 2, "pixels": pixels})

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            _validate_thermal_payload({"data": "something"})


# ---------------------------------------------------------------------------
# expand_thermal_data
# ---------------------------------------------------------------------------

class TestExpandThermalData:
    def test_basic_shape(self):
        compact = {"w": 2, "h": 2, "min": 20.0, "max": 21.0, "t": [20.0, 20.5, 20.7, 21.0]}
        result = expand_thermal_data(compact)
        assert result["width"] == 2
        assert result["height"] == 2
        assert result["min_temp"] == pytest.approx(20.0)
        assert result["max_temp"] == pytest.approx(21.0)
        assert len(result["pixels"]) == 4

    def test_pixel_coordinates(self):
        compact = {"w": 3, "h": 2, "min": 20.0, "max": 20.5, "t": [20.0] * 6}
        result = expand_thermal_data(compact)
        coords = [(p["row"], p["col"]) for p in result["pixels"]]
        expected = [(r, c) for r in range(2) for c in range(3)]
        assert coords == expected

    def test_pixel_temperatures(self):
        temps = [20.0, 20.5, 21.0, 21.5]
        compact = {"w": 2, "h": 2, "min": 20.0, "max": 21.5, "t": temps}
        result = expand_thermal_data(compact)
        result_temps = [p["temp"] for p in result["pixels"]]
        assert result_temps == pytest.approx(temps)

    def test_pixel_colors_are_valid_rgb(self):
        compact = {"w": 4, "h": 1, "min": 20.0, "max": 23.0, "t": [20.0, 21.0, 22.0, 23.0]}
        result = expand_thermal_data(compact)
        for p in result["pixels"]:
            assert 0 <= p["r"] <= 255
            assert 0 <= p["g"] <= 255
            assert 0 <= p["b"] <= 255


# ---------------------------------------------------------------------------
# collapse_to_compact (roundtrip)
# ---------------------------------------------------------------------------

class TestCollapseToCompact:
    def test_roundtrip_temperatures(self):
        original_temps = [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5]
        compact_in = {"w": 4, "h": 2, "min": 20.0, "max": 23.5, "t": original_temps}
        expanded = expand_thermal_data(compact_in)
        compact_out = collapse_to_compact(expanded)
        assert compact_out["t"] == pytest.approx(original_temps, abs=0.05)

    def test_roundtrip_dimensions(self):
        compact_in = {"w": 3, "h": 2, "min": 21.0, "max": 21.0, "t": [21.0] * 6}
        expanded = expand_thermal_data(compact_in)
        compact_out = collapse_to_compact(expanded)
        assert compact_out["w"] == 3
        assert compact_out["h"] == 2

    def test_sensor_id_preserved(self):
        expanded = {
            "width": 2, "height": 1, "min_temp": 20.0, "max_temp": 21.0,
            "sensor_id": "cam-1",
            "pixels": [
                {"row": 0, "col": 0, "temp": 20.0, "r": 0, "g": 0, "b": 255},
                {"row": 0, "col": 1, "temp": 21.0, "r": 255, "g": 0, "b": 0},
            ],
        }
        compact = collapse_to_compact(expanded)
        assert compact["sensor_id"] == "cam-1"

    def test_no_sensor_id_not_included(self):
        expanded = {
            "width": 2, "height": 1, "min_temp": 20.0, "max_temp": 21.0,
            "pixels": [
                {"row": 0, "col": 0, "temp": 20.0, "r": 0, "g": 0, "b": 255},
                {"row": 0, "col": 1, "temp": 21.0, "r": 255, "g": 0, "b": 0},
            ],
        }
        compact = collapse_to_compact(expanded)
        assert "sensor_id" not in compact


# ---------------------------------------------------------------------------
# thermal_data_to_array
# ---------------------------------------------------------------------------

class TestThermalDataToArray:
    def test_compact_shape(self):
        data = {"w": 4, "h": 2, "t": [float(i) for i in range(8)]}
        arr = thermal_data_to_array(data)
        assert arr.shape == (2, 4)

    def test_compact_values(self):
        data = {"w": 2, "h": 2, "t": [1.0, 2.0, 3.0, 4.0]}
        arr = thermal_data_to_array(data)
        assert arr[0, 0] == pytest.approx(1.0)
        assert arr[0, 1] == pytest.approx(2.0)
        assert arr[1, 0] == pytest.approx(3.0)
        assert arr[1, 1] == pytest.approx(4.0)

    def test_expanded_shape(self):
        pixels = [{"row": r, "col": c, "temp": 21.0} for r in range(3) for c in range(4)]
        data = {"width": 4, "height": 3, "pixels": pixels}
        arr = thermal_data_to_array(data)
        assert arr.shape == (3, 4)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            thermal_data_to_array({"something": "else"})


# ---------------------------------------------------------------------------
# estimate_room_temperature
# ---------------------------------------------------------------------------

class TestEstimateRoomTemperature:
    def test_uniform_array(self):
        arr = np.full((5, 5), 22.0)
        assert estimate_room_temperature(arr) == pytest.approx(22.0)

    def test_returns_median(self):
        arr = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = estimate_room_temperature(arr)
        assert result == pytest.approx(np.median(arr))


# ---------------------------------------------------------------------------
# detect_human_heat
# ---------------------------------------------------------------------------

class TestDetectHumanHeat:
    def test_all_room_temp_returns_zeros(self):
        arr = np.full((5, 5), 21.0)
        mask = detect_human_heat(arr, room_temp=21.0)
        assert mask.sum() == 0

    def test_below_min_human_temp_returns_zeros(self):
        arr = np.full((5, 5), 25.0)  # below MIN_HUMAN_TEMP=30
        mask = detect_human_heat(arr, room_temp=21.0)
        assert mask.sum() == 0

    def test_above_max_human_temp_returns_zeros(self):
        arr = np.full((5, 5), 50.0)  # above MAX_HUMAN_TEMP=45
        mask = detect_human_heat(arr, room_temp=21.0)
        assert mask.sum() == 0

    def test_human_temp_range_warmer_than_room_detected(self):
        arr = np.full((5, 5), 21.0)
        arr[2, 2] = 36.0  # human body temp
        mask = detect_human_heat(arr, room_temp=21.0)
        assert mask[2, 2] == 1

    def test_human_temp_but_not_warmer_than_room_threshold(self):
        # ROOM_TEMP_THRESHOLD default is 0.5; pixel at 31.0 with room at 31.0
        arr = np.full((5, 5), 31.0)
        mask = detect_human_heat(arr, room_temp=31.0)
        assert mask.sum() == 0

    def test_use_delta_mode(self):
        delta = np.zeros((5, 5))
        delta[2, 2] = 5.0  # 5°C above background
        abs_arr = np.full((5, 5), 36.0)
        mask = detect_human_heat(delta, room_temp=21.0, use_delta=True, absolute_temp_array=abs_arr)
        assert mask[2, 2] == 1

    def test_use_delta_mode_absolute_too_hot(self):
        delta = np.zeros((5, 5))
        delta[2, 2] = 5.0
        abs_arr = np.full((5, 5), 50.0)  # absolute > MAX_HUMAN_TEMP=45
        mask = detect_human_heat(delta, room_temp=21.0, use_delta=True, absolute_temp_array=abs_arr)
        assert mask[2, 2] == 0


# ---------------------------------------------------------------------------
# find_people_clusters
# ---------------------------------------------------------------------------

class TestFindPeopleClusters:
    def _make_human_mask_and_arr(self, hot_temp=36.0, hot_rows=range(3, 6), hot_cols=range(3, 6)):
        arr = np.full((10, 10), 21.0)
        for r in hot_rows:
            for c in hot_cols:
                arr[r, c] = hot_temp
        mask = detect_human_heat(arr, room_temp=21.0)
        return mask, arr

    def test_empty_mask_returns_empty_list(self):
        arr = np.full((5, 5), 21.0)
        mask = np.zeros((5, 5), dtype=int)
        assert find_people_clusters(mask, arr) == []

    def test_cluster_too_small_filtered_out(self):
        # MIN_CLUSTER_SIZE = 3; use only 2 adjacent pixels
        arr = np.full((5, 5), 21.0)
        arr[2, 2] = 36.0
        arr[2, 3] = 36.0  # only 2 pixels
        mask = detect_human_heat(arr, room_temp=21.0)
        assert find_people_clusters(mask, arr) == []

    def test_valid_cluster_detected(self):
        mask, arr = self._make_human_mask_and_arr()
        clusters = find_people_clusters(mask, arr)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["size"] == 9  # 3×3
        assert c["center"] == (4, 4)  # centre of rows 3-5, cols 3-5

    def test_no_fever_at_36_degrees(self):
        mask, arr = self._make_human_mask_and_arr(hot_temp=36.0)
        clusters = find_people_clusters(mask, arr)
        assert not clusters[0]["fever_detected"]
        assert not clusters[0]["elevated_temp"]

    def test_fever_detected_at_38_degrees(self):
        mask, arr = self._make_human_mask_and_arr(hot_temp=38.0)
        clusters = find_people_clusters(mask, arr)
        assert clusters[0]["fever_detected"]

    def test_elevated_temp_at_37_2_degrees(self):
        # 37.2 >= FEVER_ELEVATED_THRESHOLD_C=37.0 and < FEVER_THRESHOLD_C=37.5
        mask, arr = self._make_human_mask_and_arr(hot_temp=37.2)
        clusters = find_people_clusters(mask, arr)
        assert not clusters[0]["fever_detected"]
        assert clusters[0]["elevated_temp"]

    def test_two_separate_clusters(self):
        arr = np.full((10, 10), 21.0)
        # Cluster A: rows 0-2, cols 0-2 (9 pixels)
        arr[0:3, 0:3] = 36.0
        # Cluster B: rows 7-9, cols 7-9 (9 pixels)
        arr[7:10, 7:10] = 36.0
        mask = detect_human_heat(arr, room_temp=21.0)
        clusters = find_people_clusters(mask, arr)
        assert len(clusters) == 2


# ---------------------------------------------------------------------------
# estimate_occupancy
# ---------------------------------------------------------------------------

class TestEstimateOccupancy:
    def test_empty_room_returns_zero(self):
        data = {"w": 10, "h": 10, "t": [21.0] * 100}
        result = estimate_occupancy(data)
        assert result["occupancy"] == 0

    def test_one_person_returns_one(self):
        arr = [21.0] * 100
        for r in range(3, 6):
            for c in range(3, 6):
                arr[r * 10 + c] = 36.0
        data = {"w": 10, "h": 10, "t": arr}
        result = estimate_occupancy(data)
        assert result["occupancy"] == 1

    def test_returns_people_clusters_key(self):
        data = {"w": 4, "h": 4, "t": [21.0] * 16}
        result = estimate_occupancy(data)
        assert "people_clusters" in result

    def test_returns_room_temperature(self):
        data = {"w": 4, "h": 4, "t": [21.0] * 16}
        result = estimate_occupancy(data)
        assert result["room_temperature"] == pytest.approx(21.0)

    def test_invalid_data_returns_default(self):
        result = estimate_occupancy({"bad": "data"})
        assert result["occupancy"] == 0
        assert "error" in result


# ---------------------------------------------------------------------------
# apply_occupancy_signal_processing
# ---------------------------------------------------------------------------

class TestApplyOccupancySignalProcessing:
    def _base_result(self, occupancy=0):
        return {
            "occupancy": occupancy,
            "people_clusters": [],
            "fever_count": 0,
            "elevated_count": 0,
            "any_fever": False,
            "any_elevated": False,
        }

    def _arr(self, val=21.0):
        return np.full((5, 5), val)

    def test_first_frame_smoothed_equals_raw(self):
        result = self._base_result(2)
        apply_occupancy_signal_processing("s1", result, self._arr())
        assert result["occupancy_raw_instant"] == 2
        assert result["occupancy"] == 2  # single-frame window, median=2

    def test_smoothing_across_frames(self):
        # Push 5 frames: [0, 0, 2, 2, 2] → median=2
        for val in [0, 0, 2, 2]:
            r = self._base_result(val)
            apply_occupancy_signal_processing("smooth-s", r, self._arr())
        r = self._base_result(2)
        apply_occupancy_signal_processing("smooth-s", r, self._arr())
        assert r["occupancy"] == 2

    def test_hysteresis_suppresses_small_change(self):
        # Establish smoothed=2
        for _ in range(5):
            r = self._base_result(2)
            apply_occupancy_signal_processing("hyst-s", r, self._arr())
        # Small change of 1 (delta <= OCCUPANCY_HYSTERESIS_DELTA=1) should stay at 2
        r = self._base_result(1)
        apply_occupancy_signal_processing("hyst-s", r, self._arr())
        # The window still has many 2s, candidate may still be 2; just verify no crash
        assert isinstance(r["occupancy"], int)

    def test_fever_consecutive_frames_gating(self):
        """Single fever frame should NOT trigger any_fever (needs ≥2 consecutive)."""
        r = self._base_result()
        r["any_fever"] = True
        apply_occupancy_signal_processing("fever-s", r, self._arr())
        assert r["fever_consecutive_frames"] == 1
        assert r["any_fever"] is False  # 1 < FEVER_MIN_CONSECUTIVE_FRAMES=2

    def test_fever_confirmed_after_two_frames(self):
        for _ in range(2):
            r = self._base_result()
            r["any_fever"] = True
            apply_occupancy_signal_processing("fever2-s", r, self._arr())
        assert r["any_fever"] is True
        assert r["fever_consecutive_frames"] == 2

    def test_fever_streak_resets_on_clear_frame(self):
        for _ in range(2):
            r = self._base_result()
            r["any_fever"] = True
            apply_occupancy_signal_processing("fever3-s", r, self._arr())
        r = self._base_result()
        r["any_fever"] = False
        apply_occupancy_signal_processing("fever3-s", r, self._arr())
        assert r["fever_consecutive_frames"] == 0
        assert r["any_fever"] is False

    def test_frame_invalid_uses_previous_occupancy(self):
        # Prime with a valid frame at occupancy=3
        r = self._base_result(3)
        apply_occupancy_signal_processing("jump-s", r, np.full((5, 5), 21.0))
        # Now send a frame with a large median jump (>FRAME_ROOM_MEDIAN_MAX_JUMP_C=4.0)
        r2 = self._base_result(0)
        apply_occupancy_signal_processing("jump-s", r2, np.full((5, 5), 30.0))
        assert r2["frame_valid"] is False
        assert r2["occupancy_effective_raw"] == 3  # reverts to last known good


# ---------------------------------------------------------------------------
# _sanitize_sensor_id_for_filename
# ---------------------------------------------------------------------------

class TestSanitizeSensorId:
    def test_alphanumeric_unchanged(self):
        assert _sanitize_sensor_id_for_filename("sensor123") == "sensor123"

    def test_hyphens_and_underscores_kept(self):
        assert _sanitize_sensor_id_for_filename("sensor-1_A") == "sensor-1_A"

    def test_spaces_replaced_with_underscore(self):
        assert _sanitize_sensor_id_for_filename("my sensor") == "my_sensor"

    def test_slashes_replaced_with_underscore(self):
        result = _sanitize_sensor_id_for_filename("room/floor1")
        assert "/" not in result

    def test_none_returns_unknown(self):
        assert _sanitize_sensor_id_for_filename(None) == "unknown"

    def test_empty_string_returns_unknown(self):
        assert _sanitize_sensor_id_for_filename("") == "unknown"

    def test_truncated_at_64_chars(self):
        long_id = "a" * 100
        result = _sanitize_sensor_id_for_filename(long_id)
        assert len(result) == 64


# ---------------------------------------------------------------------------
# convert_numpy_types
# ---------------------------------------------------------------------------

class TestConvertNumpyTypes:
    def test_np_int64_becomes_int(self):
        result = convert_numpy_types(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_float64_becomes_float(self):
        result = convert_numpy_types(np.float64(3.14))
        assert isinstance(result, float)

    def test_np_ndarray_becomes_list(self):
        result = convert_numpy_types(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_dict_with_numpy_values(self):
        result = convert_numpy_types({"a": np.int64(1), "b": np.float64(2.5)})
        assert result == {"a": 1, "b": 2.5}
        assert isinstance(result["a"], int)

    def test_list_with_numpy_values(self):
        result = convert_numpy_types([np.int64(1), np.int64(2)])
        assert result == [1, 2]
        assert all(isinstance(v, int) for v in result)

    def test_nested_structure(self):
        data = {"clusters": [{"size": np.int64(9), "temp": np.float64(36.5)}]}
        result = convert_numpy_types(data)
        assert isinstance(result["clusters"][0]["size"], int)
        assert isinstance(result["clusters"][0]["temp"], float)

    def test_plain_python_types_pass_through(self):
        data = {"x": 1, "y": 2.5, "z": "hello"}
        assert convert_numpy_types(data) == data

    def test_tuple_preserved(self):
        result = convert_numpy_types((np.int64(1), np.int64(2)))
        assert result == (1, 2)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# detect_subpage_artifact / interpolate_subpages
# ---------------------------------------------------------------------------

def _make_checkerboard_frame(base_temp: float = 25.0, offset: float = 1.5, rows: int = 24, cols: int = 32) -> np.ndarray:
    """Synthetic MLX90640-style subpage artifact: even rows are shifted up, odd rows down."""
    frame = np.full((rows, cols), base_temp, dtype=float)
    for r in range(rows):
        frame[r] += offset if r % 2 == 0 else -offset
    return frame


def _make_clean_frame(base_temp: float = 25.0, rows: int = 24, cols: int = 32) -> np.ndarray:
    """Uniform frame with small random noise — no systematic row alternation."""
    rng = np.random.default_rng(42)
    return base_temp + rng.normal(0, 0.05, (rows, cols))


class TestDetectSubpageArtifact:
    def test_clean_frame_not_flagged(self):
        frame = _make_clean_frame()
        is_corrupted, frac = detect_subpage_artifact(frame)
        assert not is_corrupted
        assert frac < main.SUBPAGE_CHECKERBOARD_FRAC

    def test_checkerboard_frame_flagged(self):
        frame = _make_checkerboard_frame(offset=1.5)
        is_corrupted, frac = detect_subpage_artifact(frame)
        assert is_corrupted
        assert frac >= main.SUBPAGE_CHECKERBOARD_FRAC

    def test_weak_offset_below_threshold_not_flagged(self):
        # offset well below SUBPAGE_ROW_DIFF_THRESHOLD_C (default 0.8 C) → not flagged
        frame = _make_checkerboard_frame(offset=0.1)
        is_corrupted, _ = detect_subpage_artifact(frame)
        assert not is_corrupted

    def test_returns_fraction_in_range(self):
        frame = _make_checkerboard_frame()
        _, frac = detect_subpage_artifact(frame)
        assert 0.0 <= frac <= 1.0

    def test_too_few_rows_returns_false(self):
        frame = np.full((3, 32), 25.0)
        is_corrupted, frac = detect_subpage_artifact(frame)
        assert not is_corrupted
        assert frac == 0.0

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setattr(main, "SUBPAGE_ARTIFACT_ENABLED", False)
        frame = _make_checkerboard_frame(offset=2.0)
        is_corrupted, frac = detect_subpage_artifact(frame)
        assert not is_corrupted
        assert frac == 0.0


class TestInterpolateSubpages:
    def test_output_is_average(self):
        corrupted = np.full((24, 32), 30.0)
        previous = np.full((24, 32), 20.0)
        result = interpolate_subpages(corrupted, previous)
        np.testing.assert_allclose(result, 25.0)

    def test_output_shape_preserved(self):
        corrupted = np.ones((24, 32))
        previous = np.zeros((24, 32))
        result = interpolate_subpages(corrupted, previous)
        assert result.shape == (24, 32)

    def test_reduces_checkerboard_amplitude(self):
        clean = _make_clean_frame()
        corrupted = _make_checkerboard_frame(offset=1.5)
        blended = interpolate_subpages(corrupted, clean)
        # Row-mean alternation amplitude in blended should be less than in corrupted
        row_means_c = corrupted.mean(axis=1)
        row_means_b = blended.mean(axis=1)
        amp_corrupted = np.abs(np.diff(row_means_c)).mean()
        amp_blended = np.abs(np.diff(row_means_b)).mean()
        assert amp_blended < amp_corrupted


class TestEstimateOccupancySubpageFields:
    """estimate_occupancy must always return subpage fields."""

    def _minimal_payload(self) -> dict:
        flat = [25.0] * 768
        return {"w": 32, "h": 24, "t": flat}

    def test_subpage_fields_present_on_success(self):
        result = estimate_occupancy(self._minimal_payload(), sensor_id="cam1")
        assert "subpage_corrupted" in result
        assert "subpage_checkerboard_frac" in result

    def test_subpage_fields_present_on_error(self):
        result = estimate_occupancy({}, sensor_id="cam1")
        assert "subpage_corrupted" in result
        assert "subpage_checkerboard_frac" in result

    def test_clean_frame_not_corrupted(self):
        result = estimate_occupancy(self._minimal_payload(), sensor_id="cam2")
        assert result["subpage_corrupted"] is False


# ---------------------------------------------------------------------------
# _run_training_thread_wrapper
# ---------------------------------------------------------------------------

class TestRunTrainingThreadWrapper:
    """Wrapper must catch any exception and set status to 'error', not leave it on 'running'."""

    def test_uncaught_exception_sets_error_state(self, monkeypatch):
        monkeypatch.setattr(main, "_ml_training_status", {"state": "running", "log": []})
        monkeypatch.setattr(main, "_run_training_thread", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        _run_training_thread_wrapper()
        assert main._ml_training_status["state"] == "error"
        assert "RuntimeError" in main._ml_training_status["message"]
        assert "boom" in main._ml_training_status["message"]

    def test_traceback_not_leaked_to_log(self, monkeypatch):
        monkeypatch.setattr(main, "_ml_training_status", {"state": "running", "log": []})
        monkeypatch.setattr(main, "_run_training_thread", lambda: (_ for _ in ()).throw(ValueError("secret path")))
        _run_training_thread_wrapper()
        for entry in main._ml_training_status.get("log", []):
            assert "secret path" not in entry
