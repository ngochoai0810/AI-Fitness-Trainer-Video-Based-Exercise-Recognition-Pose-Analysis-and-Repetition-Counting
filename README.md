# AI Fitness Trainer: Video-Based Exercise Recognition, Pose Analysis, and Repetition Counting

CLI phân tích video tập luyện bằng OpenCV, MediaPipe và model BiLSTM có sẵn.
AI Fitness Trainer nhận diện bài tập, phân tích tư thế và đếm số lần lặp
(`repetition counting`) cho `barbell biceps curl`, `push-up`, `shoulder press`
và `squat`; đồng thời xuất video đã chú thích cùng dữ liệu máy đọc được.

## Cài đặt

Yêu cầu Python 3.9. Trên Windows nên đặt virtualenv ở đường dẫn ngắn vì TensorFlow có các module nằm sâu và có thể vướng giới hạn `MAX_PATH` nếu repo nằm trong thư mục dài.

```bash
uv venv --python 3.9 C:/tmp/hm-venv
source C:/tmp/hm-venv/Scripts/activate
uv pip install -r requirements.txt
```

Linux/macOS có thể dùng `.venv` thông thường:

```bash
uv venv --python 3.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Chạy

```bash
python video_pipeline.py input.mp4 --output-dir outputs/run-name
```

Model, scaler và label encoder mặc định được tìm cạnh `video_pipeline.py`; có thể truyền thư mục khác bằng `--artifact-dir`.

Mỗi lần chạy tạo:

- `annotated.mp4`: skeleton, nhãn, confidence và bộ đếm bốn động tác.
- `landmarks.csv`: `frame_index,timestamp,landmark_id,x,y,z,visibility`.
- `summary.json`: nhãn tổng kết, số rep, frame, thời gian, FPS, video nguồn, cảnh báo và metadata model.

`frame_index` bắt đầu từ 0; `timestamp = frame_index / source_fps`. CSV giữ nguyên giá trị MediaPipe: `x` và `y` chuẩn hóa theo chiều rộng/chiều cao ảnh, `z` là độ sâu tương đối với giá trị nhỏ hơn gần camera hơn, `visibility` là độ tin cậy landmark hiện diện/không bị che.

## Hợp đồng model

- 12 landmark theo thứ tự vai, khuỷu tay, cổ tay, hông, gối, cổ chân trái/phải.
- Mỗi frame có 22 feature theo đúng code gốc: 8 góc, 12 khoảng cách 3D chuẩn hóa, 2 khoảng cách Y chuẩn hóa.
- 30 frame được xếp theo thời gian thành `(30, 22)`, flatten frame-major thành `(1, 660)`, scale rồi reshape lại `(1, 30, 22)`.
- Scaler xác nhận kích thước 660 nhưng không tự chứng minh ý nghĩa/thứ tự; hợp đồng trên được đối chiếu đồng thời với code tạo feature, thao tác flatten/reshape, scaler và input model.
- Sau 6 frame liên tiếp không có pose hợp lệ, buffer, nhãn và trạng thái chuyển động bị reset; counter đã hoàn thành được giữ lại. Đổi nhãn cũng reset trạng thái chuyển động.

## Kiểm chứng thực tế

Máy đo: Windows 11 10.0.22631, Intel Core i5-11400H (12 logical processors), RAM 7.7 GB, CPU inference, Python 3.9.25. FPS là tổng frame chia thời gian đọc + pose + inference + đếm + vẽ + ghi video; không gồm thời gian tải model.

### Kết quả chính chạy lại — 2026-10-23

Môi trường chạy lại: Windows 11, Python 3.11.15, TensorFlow 2.16.1,
MediaPipe 0.10.14, OpenCV 4.9.0, CPU inference.

| Video | Độ phân giải | Nhãn dự đoán | Rep thực tế | Rep đếm | Sai số | Frame pose | Thời gian | FPS | Confidence TB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `demo_2.mp4` | 1280×720 | barbell biceps curl | 3 | 3 | 0 | 300/300 (100%) | 25.928 s | 11.570 | 0.995093 |
| `shoulder_press_form.mp4` | 1280×720 | shoulder press | 5 | 4 | 1 | 315/315 (100%) | 21.938 s | 14.359 | 0.994136 |

Tóm tắt trên hai video mẫu:

- Nhận diện đúng: **2/2 video (100%)**.
- Pose hợp lệ: **615/615 frame (100%)**.
- Đếm rep: `barbell biceps curl` đúng 3/3; `shoulder press` thiếu 1 rep.
- Test tự động: **13/13 pass**, exit code 0.

Đây là smoke test trên tập mẫu nhỏ, không đại diện cho Accuracy, Precision,
Recall hoặc F1 trên toàn dataset. Push-up và squat chưa được kiểm chứng bằng
video thực nghiệm trong lần chạy này.

Chạy test (Python 3.11.15, môi trường sạch, 13/13 pass):

```bash
python -m pytest -q
```

Artifact kiểm chứng nằm trong `verification/barbell_biceps_curl/` và `verification/shoulder_press/`.

Kết quả chạy lại ngày 2026-10-23, gồm log test và các chỉ số confidence/FPS,
được ghi trong [`runtime-validation-report.md`](runtime-validation-report.md);
artifact video nằm trong `outputs/rerun-20261023/`.

## Giới hạn và nguồn

- Cần một người trong khung hình; góc quay, che khuất và landmark kém làm giảm độ chính xác.
- Model cần đủ 30 frame pose để phát nhãn đầu tiên nên có thể bỏ lỡ rep đầu video.
- Video không có pose hoặc ngắn hơn cửa sổ vẫn xuất artifact với warning và nhãn `null`; video không đọc được trả lỗi và dọn file tạm.
- Source/model kế thừa từ [Riccardo Riccio](https://github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting) và bài báo [Real-Time Fitness Exercise Classification and Counting from Video Frames](https://arxiv.org/abs/2411.11548).
- Repo nguồn không có file giấy phép phần mềm; dự án này không tự gán giấy phép. Dataset tham chiếu công bố CC BY-NC-SA 4.0, nhưng không có video dataset nào được thêm vào bàn giao này.
