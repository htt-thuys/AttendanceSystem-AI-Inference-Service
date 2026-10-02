# Sử dụng base image Python gọn nhẹ hoặc image CUDA nếu deploy GPU
FROM python:3.11-slim


WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy file định nghĩa thư viện và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Tạo thư mục chứa trọng số mô hình (.onnx)
RUN mkdir -p /app/weights

# Copy toàn bộ mã nguồn vào container
COPY . .

EXPOSE 8000

# Khởi chạy FastAPI Gateway bằng Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
