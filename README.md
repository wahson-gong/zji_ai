安装依赖:

bash
pip install -r requirements.txt


运行服务:

bash
python app.py

生产环境建议使用:

bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app


导入算法:

bash
curl -X POST http://localhost:5000/import_ai_algorithm \
-H "Content-Type: application/json" \
-d '{
  "algorithm_id": "yolo_v8_person",
  "algorithm_name": "YOLOv8 Person Detection",
  "confidence": 0.7,
  "iou": 0.5,
  "algorithm_params": [
    {"key": "img_size", "value": "640"},
    {"key": "batch_size", "value": "16"}
  ],
  "model_file_url": "https://example.com/models/yolov8s.pt",
  "callback_url": "http://go-service/callback",
  "targets": [
    {"target_key": "person", "target_name": "Person"},
    {"target_key": "car", "target_name": "Car"}
  ],
  "model_path": "/models/yolov8"
}'


启动检测任务:

bash
curl -X POST http://localhost:5000/start_ai_stream_push \
-H "Content-Type: application/json" \
-d '{
  "flight_id": "FL12345",
  "stream_url": "rtsp://example.com/live/stream",
  "rtmp_host": "rtmp://192.168.0.20/live/",
  "algorithm_ids": ["yolo_v8_person"],
  "ntp": "ntp.example.com",
  "callback_api": "http://go-service/detection_callback"
}'


停止检测任务:

bash
curl -X POST http://localhost:5000/stop_ai_stream_push \
-H "Content-Type: application/json" \
-d '{"flight_id": "FL12345"}'