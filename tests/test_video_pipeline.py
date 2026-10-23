from pathlib import Path

import cv2
import numpy as np
import pytest

from video_pipeline import (
    ArtifactBundle,
    ArtifactContractError,
    RepetitionTracker,
    SequenceState,
    VideoProcessingError,
    analyze_video,
    build_warnings,
    counting_landmarks_are_valid,
    extract_invariant_features,
    feature_landmarks_are_valid,
    prepare_model_input,
    validate_artifact_contract,
    validate_video_path,
)


class RecordingScaler:
    n_features_in_ = 660

    def __init__(self):
        self.seen = None

    def transform(self, values):
        self.seen = values.copy()
        return values + 1.0


class FakeModel:
    input_shape = (None, 30, 22)
    output_shape = (None, 4)


class FakeEncoder:
    classes_ = np.array(
        ["barbell biceps curl", "push-up", "shoulder press", "squat"]
    )


def sample_landmarks():
    coords = {
        11: (0.1, 0.1, 0.1),
        12: (0.3, 0.1, 0.1),
        13: (0.1, 0.2, 0.1),
        14: (0.3, 0.2, 0.1),
        15: (0.2, 0.2, 0.1),
        16: (0.2, 0.2, 0.1),
        23: (0.1, 0.4, 0.1),
        24: (0.3, 0.4, 0.1),
        25: (0.1, 0.6, 0.1),
        26: (0.3, 0.6, 0.1),
        27: (0.2, 0.6, 0.1),
        28: (0.2, 0.6, 0.1),
    }
    return {idx: (*xyz, 1.0) for idx, xyz in coords.items()}


def test_extract_invariant_features_preserves_repository_order():
    features = extract_invariant_features(sample_landmarks())

    expected = np.array(
        [
            90,
            90,
            90,
            90,
            180,
            180,
            0,
            0,
            2 / 3,
            2 / 3,
            2 / 3,
            2 / 3,
            1,
            1,
            4 / 3,
            4 / 3,
            np.sqrt(0.02) / 0.3,
            np.sqrt(0.02) / 0.3,
            np.sqrt(0.05) / 0.3,
            np.sqrt(0.05) / 0.3,
            1 / 3,
            1 / 3,
        ]
    )
    np.testing.assert_allclose(features, expected, atol=1e-7)


def test_prepare_model_input_flattens_frame_major_then_restores_shape():
    frames = np.arange(30 * 22, dtype=float).reshape(30, 22)
    scaler = RecordingScaler()

    prepared = prepare_model_input(frames, scaler)

    assert scaler.seen.shape == (1, 660)
    np.testing.assert_array_equal(scaler.seen[0], np.arange(660))
    assert prepared.shape == (1, 30, 22)
    np.testing.assert_array_equal(prepared[0], frames + 1)


def test_artifact_contract_checks_all_four_artifacts():
    validate_artifact_contract(FakeModel(), RecordingScaler(), FakeEncoder())

    bad_scaler = RecordingScaler()
    bad_scaler.n_features_in_ = 22
    with pytest.raises(ArtifactContractError, match="660"):
        validate_artifact_contract(FakeModel(), bad_scaler, FakeEncoder())


@pytest.mark.parametrize(
    ("label", "down", "up", "counter_key"),
    [
        ("push-up", {"left_arm": 200}, {"left_arm": 250}, "push-up"),
        (
            "squat",
            {"right_leg": 170, "left_leg": 200},
            {"right_leg": 130, "left_leg": 220},
            "squat",
        ),
        (
            "barbell biceps curl",
            {"right_arm": 180, "left_arm": 180},
            {"right_arm": 40, "left_arm": 40},
            "barbell biceps curl",
        ),
        (
            "shoulder press",
            {"right_arm": 300, "left_arm": 70},
            {"right_arm": 230, "left_arm": 130},
            "shoulder press",
        ),
    ],
)
def test_original_repetition_transitions(label, down, up, counter_key):
    tracker = RepetitionTracker()
    tracker.update_from_angles(label, down)
    tracker.update_from_angles(label, up)
    assert tracker.counters[counter_key] == 1


def test_label_change_resets_stage_without_losing_completed_counts():
    tracker = RepetitionTracker()
    tracker.update_from_angles("push-up", {"left_arm": 200})
    tracker.counters["squat"] = 2

    tracker.update_from_angles("squat", {"right_leg": 130, "left_leg": 220})
    tracker.update_from_angles("push-up", {"left_arm": 250})

    assert tracker.counters["squat"] == 2
    assert tracker.counters["push-up"] == 0


def test_six_missing_pose_frames_clear_buffer_label_and_stage_only():
    tracker = RepetitionTracker()
    tracker.current_label = "push-up"
    tracker.stages["push-up"] = "down"
    tracker.counters["push-up"] = 3
    state = SequenceState(tracker=tracker, features=[np.zeros(22) for _ in range(10)])

    for _ in range(5):
        assert state.note_missing_pose() is False
    assert len(state.features) == 10
    assert tracker.current_label == "push-up"

    assert state.note_missing_pose() is True
    assert state.features == []
    assert tracker.current_label is None
    assert all(stage is None for stage in tracker.stages.values())
    assert tracker.counters["push-up"] == 3


def test_warning_classification_for_no_pose_and_short_video():
    assert build_warnings(total_frames=20, pose_frames=0, prediction_count=0) == [
        "No pose was detected in the video.",
        "Video did not provide 30 valid pose frames for classification.",
    ]


def test_missing_video_is_rejected(tmp_path: Path):
    with pytest.raises(VideoProcessingError, match="does not exist"):
        validate_video_path(tmp_path / "missing.mp4")


def test_cropped_lower_body_can_classify_but_not_count_a_squat():
    landmarks = sample_landmarks()
    for idx in (25, 26, 27, 28):
        x, y, z, _ = landmarks[idx]
        landmarks[idx] = (x, y, z, 0.01)

    assert feature_landmarks_are_valid(landmarks) is True
    assert counting_landmarks_are_valid("shoulder press", landmarks) is True
    assert counting_landmarks_are_valid("squat", landmarks) is False


def test_short_video_without_pose_still_writes_valid_artifacts(tmp_path: Path):
    input_path = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
    )
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    model_path = tmp_path / "fake-model.h5"
    model_path.write_bytes(b"fake")
    bundle = ArtifactBundle(
        model=FakeModel(),
        scaler=RecordingScaler(),
        label_encoder=FakeEncoder(),
        model_path=model_path,
        scaler_path=tmp_path / "fake-scaler.pkl",
        encoder_path=tmp_path / "fake-encoder.pkl",
    )

    output_dir = tmp_path / "output"
    summary = analyze_video(input_path, output_dir, artifacts=bundle)

    assert summary["predicted_label"] is None
    assert summary["total_frames"] == 10
    assert summary["frames_with_pose"] == 0
    assert summary["warnings"] == [
        "No pose was detected in the video.",
        "Video did not provide 30 valid pose frames for classification.",
    ]
    assert (output_dir / "annotated.mp4").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "landmarks.csv").read_text(encoding="utf-8").splitlines() == [
        "frame_index,timestamp,landmark_id,x,y,z,visibility"
    ]
