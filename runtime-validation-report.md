# Runtime Validation Report

**Generated**: 2026-10-23T14:47:00+07:00  
**Target**: Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting

## Summary

| Step | Status | Exit Code | Details |
|------|--------|-----------|---------|
| Environment | PASS | 0 | Clean Python 3.11 virtualenv; TensorFlow 2.16.1, MediaPipe 0.10.14, OpenCV 4.9.0 |
| Regression tests | PASS | 0 | `python -m pytest -q -p no:cacheprovider --basetemp C:\tmp\fitness-pytest-no-cache`; 13 passed, 0 failed |
| Video pipeline | PASS | 0 | Two sample videos processed; both produced `annotated.mp4`, `landmarks.csv`, and `summary.json` |
| Browser E2E | N/A | n/a | This is a local CLI/video-processing application with no browser UI |

**Overall**: PASS

## Environment and reproducibility

- Host OS: Windows 11 10.0.22631
- Python used: 3.11.15 (`C:\tmp\fitness-ai-venv`)
- TensorFlow: 2.16.1
- MediaPipe: 0.10.14
- OpenCV: 4.9.0
- NumPy: 1.23.5
- Model: `final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5`
- Model SHA-256: `859d01a10894f88196dc2512e3c01261178c450a7942dc5447f34956bee32052`
- Model input: `(30, 22)`; scaler input: `660`; classes: 4

The repository `.venv` could not import TensorFlow because its installation was missing
`calibration_statistics_pb2`. The experiment was therefore repeated successfully in the
clean virtualenv above. No application source code was changed to work around this
environment issue.

## Video experiment results

| Video | Resolution | Frames | Pose frames | Predicted label | Repetitions | Processing time | Processing FPS | Avg confidence | Min–max confidence | Warnings |
|-------|------------|-------:|------------:|-----------------|------------:|----------------:|---------------:|---------------:|--------------------:|----------|
| `demo_2.mp4` | 1280×720 | 300 | 300 (100%) | barbell biceps curl | 3 | 25.928 s | 11.570 | 0.995093 | 0.992266–0.996447 | None |
| `shoulder_press_form.mp4` | 1280×720 | 315 | 315 (100%) | shoulder press | 4 | 21.938 s | 14.359 | 0.994136 | 0.991425–0.996260 | None |

Repetition counts for classes not present in each video were zero. The model emitted
10 predictions per video, beginning at frame 29 because the temporal input window is
30 frames.

## Regression test evidence

Command:

```powershell
C:\tmp\fitness-ai-venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp C:\tmp\fitness-pytest-no-cache tests\test_video_pipeline.py
```

Result: **13 passed, 0 failed, exit code 0**.

The two warnings are TensorFlow/JAX deprecation warnings and do not affect the result.

## Generated artifacts

- `outputs/rerun-20261023/barbell_biceps_curl/summary.json`
- `outputs/rerun-20261023/barbell_biceps_curl/annotated.mp4`
- `outputs/rerun-20261023/barbell_biceps_curl/landmarks.csv`
- `outputs/rerun-20261023/shoulder_press/summary.json`
- `outputs/rerun-20261023/shoulder_press/annotated.mp4`
- `outputs/rerun-20261023/shoulder_press/landmarks.csv`
- `outputs/rerun-20261023/rerun-results.json`

## Verdict

The tested pipeline correctly classified both supplied sample videos, detected a pose
on every frame, and completed repetition counting without warnings. On this small
two-video sample, the observed label accuracy was 2/2 (100%) and pose detection
coverage was 615/615 frames (100%). This is a smoke-test result, not a dataset-level
accuracy, precision, recall, or F1 estimate.
