"""CPU checks for observer lifetime controls, independent of CUDA measurements."""
import math

import pytest

from experiments.acquisition.probe import capture, completion_deadline


def test_completion_control_keeps_idle_tail_from_control_read_time():
    completion = dict(clock="perf_counter", completion_monotonic_ms=1250.0)
    deadline = completion_deadline(8.0, completion, now_s=1.3,
                                   idle_gap_ms=20.0, sample_interval_ms=1.0)
    assert deadline == pytest.approx(1.322)


def test_completion_control_never_extends_capture_upper_bound():
    completion = dict(clock="perf_counter", completion_monotonic_ms=7995.0)
    assert completion_deadline(8.0, completion, now_s=7.999,
                               idle_gap_ms=20.0, sample_interval_ms=1.0) == 8.0


def test_tail_accounts_for_configured_sampling_interval():
    completion = dict(clock="perf_counter", completion_monotonic_ms=2000.0)
    assert completion_deadline(8.0, completion, now_s=1.99,
                               idle_gap_ms=20.0, sample_interval_ms=2.0) == pytest.approx(2.024)


@pytest.mark.parametrize("completion", [None, {},
    dict(clock="time", completion_monotonic_ms=1000),
    dict(clock="perf_counter", completion_monotonic_ms=math.nan),
    dict(clock="perf_counter", completion_monotonic_ms=True),
    dict(clock="perf_counter", completion_monotonic_ms="1000")])
def test_invalid_completion_signal_is_rejected(completion):
    with pytest.raises(ValueError):
        completion_deadline(8.0, completion, now_s=1.0,
                            idle_gap_ms=20.0, sample_interval_ms=1.0)


@pytest.mark.parametrize("interval", [0, -1, math.nan, math.inf, True])
def test_invalid_sample_cadence_is_rejected(interval):
    with pytest.raises(ValueError):
        completion_deadline(8.0, dict(clock="perf_counter", completion_monotonic_ms=1000),
                            now_s=1.0, idle_gap_ms=20.0, sample_interval_ms=interval)


def test_stale_control_file_is_rejected_before_cuda_initialization(tmp_path):
    control = tmp_path / "stop.json"
    control.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        capture(8.0, 1, stop_file=control)
