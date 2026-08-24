import math

import pytest

from openscenesense.exceptions import ConfigurationError
from openscenesense.frame_selectors import (
    DynamicFrameSelector,
    UniformFrameSelector,
    _compute_frame_indices,
    _target_frame_count,
)
from openscenesense.models import VideoMetadata


def test_dynamic_selector_detects_synthetic_cuts_within_budget(synthetic_video):
    selector = DynamicFrameSelector(
        scene_change_threshold=0.10,
        scene_scan_fps=2.0,
        min_scene_gap=0.5,
    )
    frames = selector.select_frames(str(synthetic_video), 10, 10, 60)

    assert len(frames) == 10
    assert [frame.timestamp for frame in frames] == sorted(frame.timestamp for frame in frames)
    assert len({frame.timestamp for frame in frames}) == len(frames)
    cuts = [frame.timestamp for frame in frames if frame.selection_reason == "scene_change"]
    assert all(any(abs(found - expected) <= 0.55 for found in cuts) for expected in (1, 2, 3))
    assert selector.last_scan_frame_count <= math.ceil(4 * 2.0) + 2


def test_uniform_selector_caps_short_video_and_marks_boundaries(synthetic_video):
    frames = UniformFrameSelector().select_frames(str(synthetic_video), 8, 12, 4)

    assert len(frames) == 8
    assert frames[0].selection_reason == "opening"
    assert frames[-1].selection_reason == "closing"
    assert len({frame.timestamp for frame in frames}) == len(frames)


@pytest.mark.parametrize("selector_type", [DynamicFrameSelector, UniformFrameSelector])
def test_selectors_return_one_unique_frame_for_one_frame_video(one_frame_video, selector_type):
    frames = selector_type().select_frames(str(one_frame_video), 8, 12, 4)

    assert len(frames) == 1
    assert frames[0].selection_reason == "opening"


def test_slow_gradient_does_not_create_false_scene_changes(gradient_video):
    selector = DynamicFrameSelector(scene_change_threshold=0.18, scene_scan_fps=4)
    frames = selector.select_frames(str(gradient_video), 8, 8, 120)

    assert all(frame.selection_reason != "scene_change" for frame in frames)


def test_frame_budget_helpers_cover_empty_zero_fps_and_invalid_inputs():
    assert _compute_frame_indices(0, 5) == []
    assert _compute_frame_indices(3, 10) == [0, 1, 2]
    assert _target_frame_count(120, 100, 2, 10, 3) == 6
    with pytest.raises(ConfigurationError):
        _target_frame_count(1, 1, 0, 10, 1)

    selector = DynamicFrameSelector()
    assert selector._scan_indices(VideoMetadata(frame_count=10, duration=2, fps=0)) == [
        0,
        2,
        4,
        6,
        9,
    ]
    assert len(selector._scan_indices(VideoMetadata(frame_count=121, duration=0, fps=0))) == 120


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scene_change_threshold": -0.1},
        {"scene_scan_fps": 0},
        {"scan_width": 31},
        {"scene_budget_ratio": 1.1},
    ],
)
def test_dynamic_selector_rejects_invalid_configuration(kwargs):
    with pytest.raises(ConfigurationError):
        DynamicFrameSelector(**kwargs)
