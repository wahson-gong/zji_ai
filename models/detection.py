import cv2
import time
import threading
import numpy as np
import base64
import requests
import subprocess
import os
import sys
import onnxruntime as ort  # 主要使用 ONNX Runtime
from collections import defaultdict
from config import Config
from models.algorithm import AlgorithmManager
import logging

logger = logging.getLogger(__name__)


class DetectionTask:
    def __init__(self, task_data, ai_stream_url):
        self.task_data = task_data
        self.ai_stream_url = ai_stream_url
        self.is_running = False
        self.thread = None
        self.model_cache = {}
        self.ffmpeg_process = None

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

        # 停止FFmpeg进程
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except Exception as e:
                logger.error(f"停止FFmpeg失败: {str(e)}")

    def _init_ffmpeg(self, width, height, fps=25):
        """初始化FFmpeg推流进程"""
        try:
            # 构建FFmpeg命令
            command = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',  # OpenCV使用的像素格式
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',  # 从标准输入读取
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-f', 'flv',
                self.ai_stream_url
            ]

            logger.info(f"启动FFmpeg: {' '.join(command)}")
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            return True
        except Exception as e:
            logger.error(f"初始化FFmpeg失败: {str(e)}")
            return False

    def _push_frame_to_rtmp(self, frame):
        """将帧推送到RTMP流"""
        if not self.ffmpeg_process:
            # 第一次运行时初始化FFmpeg
            height, width = frame.shape[:2]
            if not self._init_ffmpeg(width, height):
                return False

        try:
            # 将帧写入FFmpeg的标准输入
            self.ffmpeg_process.stdin.write(frame.tobytes())
            return True
        except Exception as e:
            logger.error(f"推送帧失败: {str(e)}")
            # 尝试重新初始化FFmpeg
            try:
                self.ffmpeg_process.terminate()
            except:
                pass
            self.ffmpeg_process = None
            return False

    def _run_detection(self):
        """检测主循环"""
        # 打开视频流
        cap = cv2.VideoCapture(self.task_data['stream_url'])
        if not cap.isOpened():
            logger.error(f"无法打开视频流: {self.task_data['stream_url']}")
            return

        # 获取视频流基本信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # 默认帧率

        # 加载所有需要的模型
        for algo_id in self.task_data['algorithm_ids']:
            self._load_model(algo_id)

        # 主处理循环
        frame_count = 0
        last_report_time = time.time()

        while self.is_running and cap.isOpened():
            start_time = time.time()

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                logger.warning("读取视频帧失败")
                time.sleep(0.1)
                continue

            # 调整帧大小
            processed_frame = self._resize_frame(frame)

            # 处理所有算法
            algorithm_results = []
            for algo_id in self.task_data['algorithm_ids']:
                algo_result = self._process_frame(processed_frame, algo_id)
                if algo_result:
                    algorithm_results.append(algo_result)

            # 在原始帧上绘制检测结果（用于推送）
            output_frame = frame.copy()
            if algorithm_results:
                # 绘制检测结果到输出帧
                self._draw_detection_results(output_frame, algorithm_results)

                # 定期上报结果（例如每秒1次）
                current_time = time.time()
                if current_time - last_report_time >= 1.0:
                    self._report_results(output_frame, algorithm_results)
                    last_report_time = current_time

            # 推送到RTMP
            self._push_frame_to_rtmp(output_frame)

            # 控制处理速率
            elapsed = time.time() - start_time
            sleep_time = max(0, Config.FRAME_PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_time)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"处理帧数: {frame_count}, 当前FPS: {1 / max(0.001, elapsed + sleep_time):.1f}")

        # 清理资源
        cap.release()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        logger.info(f"检测任务结束: {self.task_data['flight_id']}")

    def _draw_detection_results(self, frame, algorithm_results):
        """在帧上绘制检测结果"""
        # 简单示例：在左上角显示算法ID和帧数
        cv2.putText(frame, f"Flight: {self.task_data['flight_id']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示每个算法的目标数量
        y_offset = 60
        for algo_result in algorithm_results:
            algo_text = f"{algo_result['algorithm_id']}: "
            for target in algo_result['targets']:
                algo_text += f"{target['target_key']}={target['number']} "

            cv2.putText(frame, algo_text,
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

    def _load_model(self, algorithm_id):
        """加载算法模型 - 专注于ONNX格式"""
        if algorithm_id in self.model_cache:
            return self.model_cache[algorithm_id]

        # 获取算法配置
        algo_config = AlgorithmManager.get_algorithm(algorithm_id)
        if not algo_config:
            logger.error(f"算法配置未找到: {algorithm_id}")
            return None

        # 检查模型文件路径
        model_path = algo_config.get('local_model_path')
        if not model_path or not os.path.exists(model_path):
            logger.error(f"模型文件未找到: {algorithm_id}")
            return None

        try:
            # 检查文件扩展名
            _, ext = os.path.splitext(model_path)
            ext = ext.lower()

            if ext == '.onnx':
                # 加载ONNX模型
                logger.info(f"加载ONNX模型: {algorithm_id}")

                # 配置ONNX Runtime - 优先使用GPU
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider']
                elif 'TensorrtExecutionProvider' in available_providers:
                    providers = ['TensorrtExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']

                # 创建推理会话
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=providers
                )

                # 获取输入输出信息
                input_details = session.get_inputs()[0]
                output_details = session.get_outputs()

                # 保存模型信息
                model_info = {
                    'type': 'onnx',
                    'session': session,
                    'input_name': input_details.name,
                    'input_shape': input_details.shape,
                    'output_names': [out.name for out in output_details]
                }

                self.model_cache[algorithm_id] = model_info
                return model_info
            else:
                logger.error(f"不支持的模型格式: {ext}")
                return None

        except Exception as e:
            logger.error(f"加载模型失败: {algorithm_id}, 错误: {str(e)}")
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
        """处理单帧图像 - ONNX推理"""
        model_info = self._load_model(algorithm_id)
        if not model_info:
            return None

        # 获取算法配置
        algo_config = AlgorithmManager.get_algorithm(algorithm_id)
        confidence = algo_config.get('confidence', 0.5)

        try:
            # ONNX模型处理
            if model_info['type'] == 'onnx':
                # 预处理帧
                input_tensor, original_shape = self._preprocess_onnx_frame(frame, model_info)

                # 运行推理
                outputs = model_info['session'].run(
                    model_info['output_names'],
                    {model_info['input_name']: input_tensor}
                )

                # 后处理结果
                target_counts = self._postprocess_onnx_results(
                    outputs,
                    original_shape,
                    confidence_threshold=confidence,
                    algorithm_id=algorithm_id
                )

                # 构建结果
                return {
                    "algorithm_id": algorithm_id,
                    "targets": [{"target_key": k, "number": v} for k, v in target_counts.items()]
                }
            else:
                logger.error(f"未知模型类型: {model_info['type']}")
                return None

        except Exception as e:
            logger.error(f"处理帧时出错: {algorithm_id}, 错误: {str(e)}")
            return None

    def _preprocess_onnx_frame(self, frame, model_info):
        """预处理帧以输入ONNX模型"""
        # 获取目标输入形状 (通常为 [1, 3, height, width])
        input_shape = model_info['input_shape']
        _, _, target_height, target_width = input_shape

        # 保存原始形状
        original_height, original_width = frame.shape[:2]

        # 调整大小
        resized = cv2.resize(frame, (int(target_width), int(target_height)))

        # 转换颜色空间 BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 归一化 (根据模型要求调整)
        normalized = rgb.astype(np.float32) / 255.0

        # 调整维度顺序 HWC -> CHW
        chw = np.transpose(normalized, (2, 0, 1))

        # 添加批次维度
        input_tensor = np.expand_dims(chw, axis=0)

        return input_tensor, (original_height, original_width)

    def _postprocess_onnx_results(self, outputs, original_shape, confidence_threshold, algorithm_id):
        """后处理ONNX模型输出"""
        # 获取算法配置
        algo_config = AlgorithmManager.get_algorithm(algorithm_id)
        if not algo_config:
            logger.error(f"算法配置未找到: {algorithm_id}")
            return {}

        # 获取原始尺寸
        orig_h, orig_w = original_shape

        # 统计目标数量
        target_counts = defaultdict(int)

        # 获取检测结果 - 假设第一个输出是检测结果
        detections = outputs[0][0]  # [batch, num_detections, 6]

        # 遍历所有检测
        for detection in detections:
            # 获取置信度和类别ID
            *bbox, conf, class_id = detection

            # 检查置信度阈值
            if conf < confidence_threshold:
                continue

            # 转换为整数类别
            class_id = int(class_id)

            # 获取目标key
            target_key = self._get_target_key_by_class_id(algo_config, class_id)
            if target_key:
                # 更新计数
                target_counts[target_key] += 1

        return target_counts

    def _get_target_key_by_class_id(self, algo_config, class_id):
        """根据类别ID获取目标key"""
        # 检查算法配置中是否有目标映射
        if 'targets' not in algo_config:
            return f"class_{class_id}"

        # 查找匹配的目标
        for target in algo_config['targets']:
            # 假设配置中有class_id字段或使用索引
            if 'class_id' in target and target['class_id'] == class_id:
                return target['target_key']

            # 或者使用索引作为类ID
            if 'index' in target and target['index'] == class_id:
                return target['target_key']

        # 如果没有匹配，使用默认格式
        return f"class_{class_id}"

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

            # 检查响应是否符合统一格式
            if response.status_code != 200:
                logger.error(f"结果上报失败: HTTP { self.task_data['callback_api']} --->>>{report_data} {response.status_code}")
            else:
                resp_data = response.json()
                if resp_data.get('code', 0) != 0:
                    logger.error(f"结果上报返回错误: {resp_data.get('message', '')}")
        except Exception as e:
            logger.error(f"结果上报异常: {str(e)}")