# AI Service System

Microservice nhận diện khuôn mặt cho hệ thống chấm công nội bộ. Repo này
**chỉ chịu trách nhiệm inference**: nhận ảnh → trả vector embedding 512-d.
Toàn bộ phần lưu trữ vector (Qdrant), database nghiệp vụ (PostgreSQL), logic
sure / review / unknown, và xác nhận kết quả đều do team Backend
(`attendance-system-backend`) quản lý.

---

## 🎯 Trách nhiệm của service

```text
                      ┌────────────────────────┐
                      │   Backend (BE)         │
                      │  attendance-system     │
   ┌─────────┐ image  │   /api/recognize       │
   │ Camera  │──────▶ │                        │
   └─────────┘        └──────────┬─────────────┘
                                 │ image bytes
                                 ▼
                      ┌────────────────────────┐
                      │   AI Service (THIS)    │
                      │  POST /api/v1/embed    │
                      │                        │
                      │  detect → align → embed│
                      │                        │
                      └──────────┬─────────────┘
                                 │ 512-d vector
                                 ▼
                      ┌────────────────────────┐
                      │   Backend (BE)         │
                      │   Qdrant search +      │
                      │   PostgreSQL logic     │
                      └────────────────────────┘
```

---

## 🧱 Tech Stack

| Lớp | Lựa chọn |
|---|---|
| Framework | FastAPI (Python 3.11+) |
| Inference | ONNX Runtime GPU 1.19+ (auto-fallback CPU) |
| Detector | SCRFD-10GF  |
| Recognizer | ArcFace R50 |
| Container | NVIDIA CUDA 12.4 + cuDNN 9 (Ubuntu 22.04) |
| Test | pytest + httpx + FastAPI TestClient |

---

## 📁 Cấu trúc dự án

```text
ai-service-system/
├── app/
│   ├── api/                  # FastAPI routes (health, embed)
│   ├── core/                 # config, logging, runtime, constants
│   ├── ml/
│   │   ├── base.py           # ONNX session wrapper
│   │   ├── detector/         # 👤 OWNED BY DETECT MODULE TEAMMATE
│   │   │   ├── interface.py
│   │   │   ├── scrfd.py
│   │   │   └── types.py
│   │   └── recognizer/       # 👤 OWNED BY YOU (recognize module)
│   │       ├── interface.py
│   │       ├── arcface.py    #    ← FILL IN _preprocess / _postprocess HERE
│   │       └── types.py
│   ├── schemas/              # Pydantic DTOs (API contract)
│   ├── services/             # FacePipeline orchestration
│   ├── utils/                # image decode, face alignment
│   ├── dependencies.py       # Singleton pipeline lifecycle
│   └── main.py               # FastAPI entry + lifespan
├── docker/
│   ├── Dockerfile            # CUDA 12.4 + cuDNN 9 base
│   └── entrypoint.sh
├── tests/                    # pytest (uses fake detector/embedder)
├── models/                # ONNX weights (gitignored, mount as volume)
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── requirements-windows-gpu.txt
└── pyproject.toml
```

---

## 🚀 Quickstart — Local development

### 1. Chuẩn bị model weights

Đặt 2 file ONNX trong `./models/`:

- Detector: (SCRFD detector, ~16 MB)
- Recognizer: (ArcFace R50 recognizer, ~174 MB)

> Cả hai chưa có sẵn trong thư mục `models/` của repo.

### 2. Virtual environment + dependencies

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

**Windows (native GPU):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements-windows-gpu.txt
```

### 3. Cấu hình `.env`

```bash
cp .env.example .env
# chỉnh nếu cần (mặc định đã chạy được local)
```

### 4. Chạy server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Mở docs: <http://localhost:8001/docs>.

> Lần đầu khởi động sẽ mất ~3–5s để ONNX Runtime load `det_10g.onnx` và
> `w600k_r50.onnx`. Sau warm-up, `/api/v1/ready` báo `recognizer_ready: true`.

---

## 🐳 Run với Docker (GPU + CPU fallback)

```bash
# Build và chạy (GPU host)
docker compose up --build

# Trên máy không có GPU: bỏ block `deploy.resources.reservations.devices`
# trong docker-compose.yml. ONNX Runtime sẽ tự dùng CPUExecutionProvider.
```

Image được build từ `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, vì vậy
container chạy được ngay cả khi host chỉ có CPU — ONNX Runtime drop
`CUDAExecutionProvider` khi không có driver NVIDIA và fall back sang CPU
ở session creation.

Verify provider runtime:

```bash
curl http://localhost:8001/api/v1/ready
# → { "status": "ok", "detector_ready": true, "recognizer_ready": true,
#     "providers": ["CUDAExecutionProvider","CPUExecutionProvider"] }
```

---

## 📡 API Contract

### `GET /api/v1/health`

Liveness probe. Không chạm tới ML model.

```json
{ "status": "ok", "service": "ai-service-system", "version": "0.1.0", "environment": "development" }
```

### `GET /api/v1/ready`

Readiness probe — báo cáo trạng thái load của từng model + ONNX providers
đang dùng.

### `POST /api/v1/embed`

**Request:** `multipart/form-data` với 1 trường `image` (JPEG / PNG / WEBP / BMP).

**Response — 3 trường hợp:**

1. Phát hiện được mặt:
   ```json
   {
     "status": "ok",
     "embedding": [0.012, -0.034, ..., 0.087],
     "det_score": 0.97,
     "bbox": { "x1": 102.0, "y1": 88.0, "x2": 220.0, "y2": 240.0 },
     "landmarks": [
       { "x": 130.0, "y": 130.0 }, { "x": 192.0, "y": 130.0 },
       { "x": 161.0, "y": 165.0 },
       { "x": 138.0, "y": 200.0 }, { "x": 184.0, "y": 200.0 }
     ]
   }
   ```
2. Không có mặt nào: `{ "status": "no_face" }`
3. Ảnh không decode được: `{ "status": "invalid_image", "message": "..." }`

---

## 👥 Phân công 2 module ML

### 🔵 Detector (`app/ml/detector/`)

- **Owner**: Me.
- **Trạng thái hiện tại**: scaffold đầy đủ — preprocess (letterbox 640×640),
  forward, postprocess (anchor decode + NMS) đã chạy được với `det_10g.onnx`.
- **Việc cần làm**: review pipeline, tune threshold, optimize batch nếu cần.
- **Mở rộng**: implement `FaceDetector` interface và đăng ký trong
  `app/dependencies.py` để thay backbone (YOLO-Face, RetinaFace, v.v.).

### 🟢 Recognizer (`app/ml/recognizer/`)

- **Owner**: My friend.
- **Trạng thái hiện tại**: full implementation với `w600k_r50.onnx`.
  Preprocess: ArcFace chuẩn (BGR→RGB, `(x − 127.5) / 127.5`, NCHW).
  Postprocess: L2-normalize → 512-d float32.
- **Mở rộng**: muốn thay backbone (mobile, w600k_mbf, custom)? Tạo class
  mới kế thừa `FaceEmbedder` rồi swap trong `app/dependencies.py`. Toàn
  bộ pipeline + API không cần đổi.

---

## 🧪 Testing

```bash
pytest
```

Tests dùng **fake detector / fake embedder** (không cần weights thật), nên
chạy được trên CI CPU-only. Khi bạn implement xong recognizer, có thể thêm
test integration với weights thật trong `tests/integration/` (gated bằng
env var).

---

## 🔧 Tinh chỉnh runtime

Tất cả thông số runtime ngoài config qua `.env`:

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MODELS_DIR` | `./models` | Thư mục chứa file `.onnx` |
| `USE_GPU` | `true` | Thử `CUDAExecutionProvider`; tự fallback CPU |
| `ORT_INTRA_OP_THREADS` | `0` | `0` = ORT tự chọn; set rõ trên node CPU-only |
| `DETECTOR_SCORE_THRESHOLD` | `0.5` | Ngưỡng score của detector |
| `DETECTOR_NMS_THRESHOLD` | `0.4` | Ngưỡng IoU cho NMS |
| `DETECTOR_MAX_FACES` | `10` | Tối đa số face giữ lại sau NMS |
| `MAX_UPLOAD_BYTES` | `10485760` | Giới hạn upload (10 MB) |
| `CORS_ORIGINS` | `["*"]` | Whitelist origin (production nên thu hẹp) |

---

## 🚨 Notes for production

- **Image size**: weights NOT bake-in. Mount qua volume read-only.
- **Non-root user**: Dockerfile chạy với user `app` (uid không-root).
- **Healthcheck**: container HEALTHCHECK gọi `/api/v1/health`.
- **Logging**: stdout, định dạng dòng đơn — dễ scrape bằng Loki / Datadog.
- **Concurrency**: ONNX Runtime session thread-safe; uvicorn nhiều worker
  sẽ load model nhiều lần → đối với GPU, prefer 1 worker + nhiều thread.
