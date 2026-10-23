"""Offline human-movement analysis engine and command-line interface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


WINDOW_SIZE = 30
FEATURE_COUNT = 22
SCALER_FEATURE_COUNT = WINDOW_SIZE * FEATURE_COUNT
VISIBILITY_THRESHOLD = 0.5
MAX_MISSING_POSE_FRAMES = 6

MODEL_FILENAME = "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
SCALER_FILENAME = "thesis_bidirectionallstm_scaler.pkl"
ENCODER_FILENAME = "thesis_bidirectionallstm_label_encoder.pkl"

EXERCISE_CLASSES = (
    "barbell biceps curl",
    "push-up",
    "shoulder press",
    "squat",
)

# The order is inherited from ExerciseAiTrainer.py and must not be changed
# without retraining the scaler and model.
RELEVANT_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

Landmark = Tuple[float, float, float, float]
LandmarkMap = Mapping[int, Landmark]


class VideoProcessingError(RuntimeError):
    """Raised when an input video cannot be processed safely."""


class ArtifactContractError(RuntimeError):
    """Raised when the model, scaler, and encoder do not agree."""


def _angle_180(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = abs(math.degrees(radians))
    return 360.0 - angle if angle > 180.0 else angle


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:3], dtype=float) - np.asarray(b[:3], dtype=float)))


def extract_invariant_features(landmarks: LandmarkMap) -> np.ndarray:
    """Create the repository's 8-angle + 12-distance + 2-Y-distance vector."""
    missing = [idx for idx in RELEVANT_LANDMARKS if idx not in landmarks]
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")

    points = {idx: landmarks[idx][:3] for idx in RELEVANT_LANDMARKS}
    angle_triplets = (
        (11, 13, 15),
        (12, 14, 16),
        (23, 25, 27),
        (24, 26, 28),
        (11, 23, 25),
        (12, 24, 26),
        (23, 11, 13),
        (24, 12, 14),
    )
    distance_pairs = (
        (11, 12),
        (23, 24),
        (23, 25),
        (24, 26),
        (11, 23),
        (12, 24),
        (13, 25),
        (14, 26),
        (15, 11),
        (16, 12),
        (15, 23),
        (16, 24),
    )
    normalization_pairs = ((11, 23), (12, 24), (23, 25), (24, 26))

    angles = [_angle_180(points[a], points[b], points[c]) for a, b, c in angle_triplets]
    distances = [_distance(points[a], points[b]) for a, b in distance_pairs]
    normalization_factor = next(
        (
            value
            for value in (_distance(points[a], points[b]) for a, b in normalization_pairs)
            if value > 0
        ),
        0.5,
    )
    normalized_distances = [value / normalization_factor for value in distances]
    normalized_y_distances = [
        abs(points[13][1] - points[11][1]) / normalization_factor,
        abs(points[14][1] - points[12][1]) / normalization_factor,
    ]
    features = np.asarray(angles + normalized_distances + normalized_y_distances, dtype=np.float32)
    if features.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(features)):
        raise ValueError("Feature extraction did not produce 22 finite values")
    return features


def prepare_model_input(features: Sequence[Sequence[float]], scaler: Any) -> np.ndarray:
    """Scale a temporal block using the repository's frame-major layout."""
    array = np.asarray(features, dtype=float)
    if array.shape != (WINDOW_SIZE, FEATURE_COUNT):
        raise ValueError(f"Expected feature shape (30, 22), got {array.shape}")
    flattened = array.reshape(1, SCALER_FEATURE_COUNT, order="C")
    scaled = np.asarray(scaler.transform(flattened))
    if scaled.shape != (1, SCALER_FEATURE_COUNT):
        raise ArtifactContractError(
            f"Scaler returned {scaled.shape}; expected (1, {SCALER_FEATURE_COUNT})"
        )
    return scaled.reshape(1, WINDOW_SIZE, FEATURE_COUNT, order="C")


def validate_artifact_contract(model: Any, scaler: Any, label_encoder: Any) -> None:
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    scaler_features = int(getattr(scaler, "n_features_in_", -1))
    classes = tuple(str(value) for value in getattr(label_encoder, "classes_", ()))
    if input_shape[-2:] != (WINDOW_SIZE, FEATURE_COUNT):
        raise ArtifactContractError(
            f"Model input is {input_shape}; expected (None, 30, 22)"
        )
    if output_shape[-1] != len(EXERCISE_CLASSES):
        raise ArtifactContractError(
            f"Model output is {output_shape}; expected four exercise classes"
        )
    if scaler_features != SCALER_FEATURE_COUNT:
        raise ArtifactContractError(
            f"Scaler expects {scaler_features} values; expected {SCALER_FEATURE_COUNT}"
        )
    if classes != EXERCISE_CLASSES:
        raise ArtifactContractError(
            f"Encoder classes are {classes}; expected {EXERCISE_CLASSES}"
        )


@dataclass
class ArtifactBundle:
    model: Any
    scaler: Any
    label_encoder: Any
    model_path: Path
    scaler_path: Path
    encoder_path: Path

    @property
    def classes(self) -> Tuple[str, ...]:
        return tuple(str(value) for value in self.label_encoder.classes_)


def load_artifacts(artifact_dir: Path) -> ArtifactBundle:
    import joblib
    from tensorflow.keras.models import load_model

    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / MODEL_FILENAME
    scaler_path = artifact_dir / SCALER_FILENAME
    encoder_path = artifact_dir / ENCODER_FILENAME
    missing = [str(path) for path in (model_path, scaler_path, encoder_path) if not path.is_file()]
    if missing:
        raise ArtifactContractError(f"Missing model artifacts: {', '.join(missing)}")

    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
    except Exception as exc:
        raise ArtifactContractError(f"Could not load model artifacts: {exc}") from exc
    validate_artifact_contract(model, scaler, label_encoder)
    return ArtifactBundle(model, scaler, label_encoder, model_path, scaler_path, encoder_path)


@dataclass
class RepetitionTracker:
    counters: dict = field(default_factory=lambda: {label: 0 for label in EXERCISE_CLASSES})
    stages: dict = field(
        default_factory=lambda: {
            "push-up": None,
            "squat": None,
            "right_biceps": None,
            "left_biceps": None,
            "shoulder press": None,
        }
    )
    current_label: Optional[str] = None

    def reset_motion(self) -> None:
        for key in self.stages:
            self.stages[key] = None

    def clear_label_and_motion(self) -> None:
        self.current_label = None
        self.reset_motion()

    def set_label(self, label: str) -> None:
        if label not in EXERCISE_CLASSES:
            raise ValueError(f"Unsupported exercise label: {label}")
        if label != self.current_label:
            self.reset_motion()
            self.current_label = label

    def update_from_angles(self, label: str, angles: Mapping[str, float]) -> None:
        self.set_label(label)
        if label == "push-up":
            if angles["left_arm"] < 220:
                self.stages["push-up"] = "down"
            if angles["left_arm"] > 240 and self.stages["push-up"] == "down":
                self.stages["push-up"] = "up"
                self.counters[label] += 1
        elif label == "squat":
            if angles["right_leg"] > 160 and angles["left_leg"] < 220:
                self.stages["squat"] = "down"
            if (
                angles["right_leg"] < 140
                and angles["left_leg"] > 210
                and self.stages["squat"] == "down"
            ):
                self.stages["squat"] = "up"
                self.counters[label] += 1
        elif label == "barbell biceps curl":
            if 160 < angles["right_arm"] < 200:
                self.stages["right_biceps"] = "down"
            if 140 < angles["left_arm"] < 200:
                self.stages["left_biceps"] = "down"
            curled = lambda value: value > 310 or value < 60
            if (
                self.stages["right_biceps"] == "down"
                and self.stages["left_biceps"] == "down"
                and curled(angles["right_arm"])
                and curled(angles["left_arm"])
            ):
                self.stages["right_biceps"] = "up"
                self.stages["left_biceps"] = "up"
                self.counters[label] += 1
        elif label == "shoulder press":
            if angles["right_arm"] > 280 and angles["left_arm"] < 80:
                self.stages["shoulder press"] = "down"
            if (
                angles["right_arm"] < 240
                and angles["left_arm"] > 120
                and self.stages["shoulder press"] == "down"
            ):
                self.stages["shoulder press"] = "up"
                self.counters[label] += 1

    def update_from_landmarks(self, label: str, points: Mapping[int, Tuple[float, float]]) -> None:
        def angle(a: int, b: int, c: int) -> float:
            radians = math.atan2(points[c][1] - points[b][1], points[c][0] - points[b][0]) - math.atan2(
                points[a][1] - points[b][1], points[a][0] - points[b][0]
            )
            value = math.degrees(radians)
            return value + 360.0 if value < 0 else value

        if label == "push-up":
            values = {"left_arm": angle(11, 13, 15)}
        elif label == "squat":
            values = {"right_leg": angle(24, 26, 28), "left_leg": angle(23, 25, 27)}
        else:
            values = {"right_arm": angle(12, 14, 16), "left_arm": angle(11, 13, 15)}
        self.update_from_angles(label, values)


@dataclass
class SequenceState:
    tracker: RepetitionTracker = field(default_factory=RepetitionTracker)
    features: list = field(default_factory=list)
    missing_pose_frames: int = 0

    def note_valid_pose(self) -> None:
        self.missing_pose_frames = 0

    def note_missing_pose(self) -> bool:
        self.missing_pose_frames += 1
        if self.missing_pose_frames == MAX_MISSING_POSE_FRAMES:
            self.features.clear()
            self.tracker.clear_label_and_motion()
            return True
        return False


def build_warnings(total_frames: int, pose_frames: int, prediction_count: int) -> list:
    warnings = []
    if pose_frames == 0:
        warnings.append("No pose was detected in the video.")
    if prediction_count == 0:
        warnings.append("Video did not provide 30 valid pose frames for classification.")
    return warnings


def validate_video_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise VideoProcessingError(f"Input video does not exist: {input_path}")
    return input_path


def feature_landmarks_are_valid(landmarks: LandmarkMap) -> bool:
    """Match the training path: all 12 inferred coordinates must be finite.

    Visibility is deliberately not a classifier gate. MediaPipe still infers
    off-screen lower-body landmarks, and the repository model was trained from
    those coordinates without applying a visibility threshold.
    """
    for idx in RELEVANT_LANDMARKS:
        value = landmarks.get(idx)
        if value is None or not all(math.isfinite(v) for v in value[:3]):
            return False
    return True


def counting_landmarks_are_valid(label: str, landmarks: LandmarkMap) -> bool:
    required = {
        "push-up": (11, 13, 15),
        "squat": (23, 24, 25, 26, 27, 28),
        "barbell biceps curl": (11, 12, 13, 14, 15, 16),
        "shoulder press": (11, 12, 13, 14, 15, 16),
    }[label]
    return all(
        idx in landmarks
        and landmarks[idx][3] >= VISIBILITY_THRESHOLD
        and all(math.isfinite(value) for value in landmarks[idx])
        for idx in required
    )


def _dominant_label(predictions: Sequence[str]) -> Optional[str]:
    if not predictions:
        return None
    counts = Counter(predictions)
    maximum = max(counts.values())
    return next(label for label in reversed(predictions) if counts[label] == maximum)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _draw_overlay(frame: np.ndarray, tracker: RepetitionTracker, confidence: Optional[float]) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 112), (0, 0, 0), -1)
    if tracker.current_label is None:
        label_text = "Exercise: waiting for 30 valid pose frames"
    else:
        suffix = "" if confidence is None else f" ({confidence:.2f})"
        label_text = f"Exercise: {tracker.current_label}{suffix}"
    cv2.putText(frame, label_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    short = {
        "barbell biceps curl": "Curl",
        "push-up": "Push-up",
        "shoulder press": "Press",
        "squat": "Squat",
    }
    counter_text = " | ".join(f"{short[label]}: {tracker.counters[label]}" for label in EXERCISE_CLASSES)
    cv2.putText(frame, counter_text, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "Pose gap resets after 6 invalid frames", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)


def analyze_video(
    input_path: Path,
    output_dir: Path,
    artifact_dir: Optional[Path] = None,
    artifacts: Optional[ArtifactBundle] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> dict:
    """Analyze one video and atomically create annotated MP4, CSV, and JSON outputs."""
    input_path = validate_video_path(input_path)
    artifact_dir = Path(artifact_dir or Path(__file__).resolve().parent)
    bundle = artifacts or load_artifacts(artifact_dir)
    validate_artifact_contract(bundle.model, bundle.scaler, bundle.label_encoder)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError(f"OpenCV could not open input video: {input_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise VideoProcessingError("Input video has invalid frame dimensions")
    fps_warning = None
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
        fps_warning = "Source FPS was invalid; output and timestamps use 30 FPS."

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video = output_dir / "annotated.mp4"
    final_csv = output_dir / "landmarks.csv"
    final_json = output_dir / "summary.json"
    temp_video = output_dir / "annotated.tmp.mp4"
    temp_csv = output_dir / "landmarks.tmp.csv"
    temp_json = output_dir / "summary.tmp.json"
    temp_paths = (temp_video, temp_csv, temp_json)
    for path in temp_paths:
        path.unlink(missing_ok=True)

    writer = cv2.VideoWriter(
        str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise VideoProcessingError("OpenCV could not create the MP4 output writer")

    pose = None
    state = SequenceState()
    predictions = []
    prediction_records = []
    total_frames = 0
    pose_frames = 0
    current_confidence = None
    started = None
    processing_error = None

    try:
        import mediapipe as mp

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        started = time.perf_counter()
        with temp_csv.open("w", newline="", encoding="utf-8") as csv_handle:
            csv_writer = csv.writer(csv_handle)
            csv_writer.writerow(
                ["frame_index", "timestamp", "landmark_id", "x", "y", "z", "visibility"]
            )
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = total_frames
                total_frames += 1
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    pose_frames += 1
                    timestamp = frame_index / source_fps
                    landmark_map = {}
                    pixel_points = {}
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        value = (
                            float(landmark.x),
                            float(landmark.y),
                            float(landmark.z),
                            float(landmark.visibility),
                        )
                        landmark_map[idx] = value
                        pixel_points[idx] = (landmark.x * width, landmark.y * height)
                        csv_writer.writerow([frame_index, timestamp, idx, *value])

                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                    )
                    if feature_landmarks_are_valid(landmark_map):
                        state.note_valid_pose()
                        state.features.append(extract_invariant_features(landmark_map))
                        if len(state.features) == WINDOW_SIZE:
                            model_input = prepare_model_input(state.features, bundle.scaler)
                            probabilities = np.asarray(
                                bundle.model.predict(model_input, verbose=0)
                            )
                            if probabilities.shape != (1, len(bundle.classes)):
                                raise ArtifactContractError(
                                    f"Model prediction shape is {probabilities.shape}; expected (1, 4)"
                                )
                            class_index = int(np.argmax(probabilities[0]))
                            label = bundle.classes[class_index]
                            current_confidence = float(probabilities[0, class_index])
                            predictions.append(label)
                            prediction_records.append(
                                {
                                    "frame_index": frame_index,
                                    "label": label,
                                    "confidence": current_confidence,
                                }
                            )
                            state.features.clear()
                            state.tracker.set_label(label)
                        if state.tracker.current_label is not None and counting_landmarks_are_valid(
                            state.tracker.current_label, landmark_map
                        ):
                            state.tracker.update_from_landmarks(
                                state.tracker.current_label, pixel_points
                            )
                    else:
                        state.note_missing_pose()
                else:
                    state.note_missing_pose()

                _draw_overlay(frame, state.tracker, current_confidence)
                writer.write(frame)
                if progress_callback is not None:
                    progress_callback(total_frames)
    except Exception as exc:
        processing_error = exc
    finally:
        cap.release()
        writer.release()
        if pose is not None:
            pose.close()

    if processing_error is not None:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise processing_error

    processing_seconds = time.perf_counter() - started
    if total_frames == 0:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise VideoProcessingError("Input video opened but contained no readable frames")

    warnings = build_warnings(total_frames, pose_frames, len(predictions))
    if fps_warning:
        warnings.append(fps_warning)
    summary = {
        "predicted_label": _dominant_label(predictions),
        "repetition_counts": state.tracker.counters,
        "total_frames": total_frames,
        "frames_with_pose": pose_frames,
        "processing_seconds": processing_seconds,
        "processing_fps": total_frames / processing_seconds if processing_seconds > 0 else None,
        "source_video": {
            "filename": input_path.name,
            "width": width,
            "height": height,
            "fps": source_fps,
        },
        "model": {
            "filename": bundle.model_path.name,
            "sha256": _sha256(bundle.model_path) if bundle.model_path.is_file() else None,
            "input_shape": [WINDOW_SIZE, FEATURE_COUNT],
            "scaler_features": SCALER_FEATURE_COUNT,
            "classes": list(bundle.classes),
            "feature_layout": "30 chronological frames x 22 features, flattened frame-major before scaling",
        },
        "predictions": prediction_records,
        "warnings": warnings,
    }

    try:
        with temp_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_video, final_video)
        os.replace(temp_csv, final_csv)
        os.replace(temp_json, final_json)
    except Exception:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze exercise movement in an MP4 video")
    parser.add_argument("input_video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the .h5 model, scaler, and label encoder",
    )
    args = parser.parse_args(argv)
    try:
        summary = analyze_video(args.input_video, args.output_dir, args.artifact_dir)
    except (VideoProcessingError, ArtifactContractError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
