import cv2
import time
import threading
import numpy as np
import base64
import requests
from ultralytics import YOLO
from collections import defaultdict
from config import Config
from models.algorithm import AlgorithmManager  # 导入算法管理器


class DetectionTask:
    def __init__(self, task_data):
        self.task_data = task_data
        self.is_running = False
        self.thread = None
        self.model_cache = {}

    def start(self):
        """启动检测任务"""
        if self.is_running:
            return False

        self.is_running = True
        self.thread = threading.Thread(target=self._run_detection)
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop(self):
        """停止检测任务"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _run_detection(self):
        """检测主循环"""
        # 打开视频流
        cap = cv2.VideoCapture(self.task_data['stream_url'])
        if not cap.isOpened():
            print(f"无法打开视频流: {self.task_data['stream_url']}")
            return

        # 加载所有需要的模型
        for algo_id in self.task_data['algorithm_ids']:
            self._load_model(algo_id)

        # 主处理循环
        while self.is_running and cap.isOpened():
            start_time = time.time()

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break

            # 调整帧大小
            frame = self._resize_frame(frame)

            # 处理所有算法
            algorithm_results = []
            for algo_id in self.task_data['algorithm_ids']:
                algo_result = self._process_frame(frame, algo_id)
                if algo_result:
                    algorithm_results.append(algo_result)

            # 上报结果
            if algorithm_results:
                self._report_results(frame, algorithm_results)

            # 控制处理速率
            elapsed = time.time() - start_time
            sleep_time = max(0, Config.FRAME_PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_time)

        # 清理资源
        cap.release()
        print(f"检测任务结束: {self.task_data['flight_id']}")

    def _load_model(self, algorithm_id):
        """加载算法模型 - 使用本地保存的模型文件"""
        if algorithm_id in self.model_cache:
            return self.model_cache[algorithm_id]

        # 获取算法配置
        algo_config = AlgorithmManager.get_algorithm(algorithm_id)
        if not algo_config:
            print(f"算法配置未找到: {algorithm_id}")
            return None

        # 检查模型文件路径
        model_path = algo_config.get('local_model_path')
        if not model_path or not os.path.exists(model_path):
            print(f"模型文件未找到: {algorithm_id}")
            return None

        try:
            # 加载模型
            model = YOLO(model_path)
            self.model_cache[algorithm_id] = model
            print(f"成功加载模型: {algorithm_id}")
            return model
        except Exception as e:
            print(f"加载模型失败: {algorithm_id}, 错误: {str(e)}")
            return None

    def _resize_frame(self, frame):
        """调整帧大小"""
        h, w = frame.shape[:2]
        max_w, max_h = Config.MAX_FRAME_SIZE

        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h))
        return frame

    def _process_frame(self, frame, algorithm_id):
        """处理单帧图像"""
        model = self._load_model(algorithm_id)
        if not model:
            return None

        # 获取算法配置
        algo_config = AlgorithmManager.get_algorithm(algorithm_id)
        confidence = algo_config.get('confidence', 0.5)
        iou = algo_config.get('iou', 0.5)

        # 运行检测
        results = model(frame, conf=confidence, iou=iou, verbose=False)

        # 统计目标数量
        target_counts = defaultdict(int)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                # 获取目标key
                target_key = self._get_target_key(algo_config, cls_id)
                if target_key:
                    target_counts[target_key] += 1

        # 构建结果
        return {
            "algorithm_id": algorithm_id,
            "targets": [{"target_key": k, "number": v} for k, v in target_counts.items()]
        }

    def _get_target_key(self, algo_config, class_id):
        """根据类别ID获取目标key"""
        if 'targets' not in algo_config or class_id >= len(algo_config['targets']):
            return None
        return algo_config['targets'][class_id]['target_key']

    def _report_results(self, frame, algorithm_results):
        """上报检测结果"""
        # 编码图像为base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 构建上报数据
        report_data = {
            "flight_id": self.task_data['flight_id'],
            "image_base64": image_base64,
            "detection_timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "algorithm": algorithm_results
        }

        # 发送上报请求
        try:
            response = requests.post(
                self.task_data['callback_api'],
                json=report_data,
                timeout=3
            )
            if response.status_code != 200:
                print(f"结果上报失败: {response.status_code}")
        except Exception as e:
            print(f"结果上报异常: {str(e)}")