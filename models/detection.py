import cv2
import time
import threading
import numpy as np
import base64
import requests
import subprocess
import os
import sys
import onnxruntime as ort
from collections import defaultdict
from config import Config
from models.algorithm import AlgorithmManager
import logging
import re
import json

logger = logging.getLogger(__name__)


class DJIStreamDecoder:
    """大疆无人机流解码器"""

    @staticmethod
    def is_dji_stream(stream_url):
        """检查是否是大疆无人机流"""
        return "rtmp://" in stream_url and "live/" in stream_url and "FJ" in stream_url

    @staticmethod
    def get_dji_stream_info(stream_url):
        """从流URL中提取大疆无人机信息"""
        match = re.search(r'live/([A-Z0-9]+)-(\d+)-(\d+)-(\d+)', stream_url)
        if match:
            return {
                "device_id": match.group(1),
                "video_type": int(match.group(2)),
                "resolution": int(match.group(3)),
                "fps": int(match.group(4))
            }
        return None

    @staticmethod
    def get_dji_resolution(resolution_code):
        """根据大疆分辨率代码获取实际分辨率"""
        resolutions = {
            0: (4096, 2160),  # 4K
            1: (3840, 2160),  # 4K UHD
            2: (2720, 1530),  # 2.7K
            3: (1920, 1080),  # 1080p
            4: (1280, 720),  # 720p
            5: (1920, 1080),  # 1080i
        }
        return resolutions.get(resolution_code, (1920, 1080))

    @staticmethod
    def get_dji_decoder_command(stream_url, width, height):
        """获取大疆专用解码命令"""
        # 大疆专用解码参数
        return [
            Config.FFMPEG_PATH,
            '-hide_banner',
            '-loglevel', 'warning',
            '-fflags', 'nobuffer',
            '-analyzeduration', '1000000',  # 减少分析时间
            '-probesize', '32',  # 减少探测大小
            '-f', 'flv',  # 强制输入格式为FLV
            '-i', stream_url,
            '-c:v', 'h264',  # 指定H.264解码器
            '-flags', 'low_delay',  # 低延迟模式
            '-strict', 'experimental',  # 允许实验性功能
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV使用的格式
            '-'
        ]

class DetectionTask:
    def __init__(self, task_data, ai_stream_url):
        self.task_data = task_data
        self.ai_stream_url = ai_stream_url
        self.is_running = False
        self.thread = None
        self.model_cache = {}
        self.ffmpeg_process = None
        self.ffmpeg_input = None
        self.dji_stream_info = None
        self.frame_size = None
        self.frame_shape = None

        # 检查是否是大疆无人机流
        if DJIStreamDecoder.is_dji_stream(task_data['stream_url']):
            self.dji_stream_info = DJIStreamDecoder.get_dji_stream_info(task_data['stream_url'])
            logger.info(f"检测到大疆无人机流: {self.dji_stream_info}")

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

        # 停止输入流进程
        if self.ffmpeg_input:
            try:
                self.ffmpeg_input.terminate()
                self.ffmpeg_input.wait(timeout=2)
            except Exception as e:
                logger.error(f"停止输入流进程失败: {str(e)}")

    def _init_ffmpeg(self, width, height, fps=25):
        """初始化FFmpeg推流进程，添加错误处理"""
        try:
            # 构建FFmpeg命令
            command = [
                Config.FFMPEG_PATH,
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

            logger.info(f"启动FFmpeg推流: {' '.join(command)}")

            # 创建日志文件
            ffmpeg_log = open('ffmpeg_push.log', 'a') if Config.FFMPEG_DEBUG_LOGGING else subprocess.DEVNULL

            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=ffmpeg_log,
                bufsize=10 ** 7  # 大缓冲区 (10MB)
            )

            # 添加进程检查
            time.sleep(0.5)  # 等待进程启动
            if self.ffmpeg_process.poll() is not None:
                logger.error("FFmpeg进程启动后立即退出，状态码: %s", self.ffmpeg_process.returncode)
                return False

            logger.info("FFmpeg推流进程启动成功")
            return True
        except Exception as e:
            logger.error(f"初始化FFmpeg失败: {str(e)}")
            return False

    def _push_frame_to_rtmp(self, frame):
        """将帧推送到RTMP流，处理Broken pipe错误"""
        try:
            # 检查FFmpeg进程是否仍在运行
            if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                logger.warning("FFmpeg进程已终止，状态码: %s", self.ffmpeg_process.returncode)
                self.ffmpeg_process = None

            # 如果FFmpeg进程不存在，重新初始化
            if not self.ffmpeg_process:
                height, width = frame.shape[:2]
                if not self._init_ffmpeg(width, height):
                    logger.error("无法重新初始化FFmpeg推流进程")
                    return False
                else:
                    logger.info("FFmpeg推流进程已重新启动")

            # 将帧写入FFmpeg的标准输入
            self.ffmpeg_process.stdin.write(frame.tobytes())
            self.ffmpeg_process.stdin.flush()  # 确保数据被刷新
            return True
        except BrokenPipeError:
            logger.error("Broken pipe错误: FFmpeg管道已断开")
            # 尝试重新初始化FFmpeg
            try:
                if self.ffmpeg_process:
                    self.ffmpeg_process.terminate()
            except:
                pass
            self.ffmpeg_process = None
            return False
        except Exception as e:
            logger.error(f"推送帧失败: {str(e)}")
            # 尝试重新初始化FFmpeg
            try:
                if self.ffmpeg_process:
                    self.ffmpeg_process.terminate()
            except:
                pass
            self.ffmpeg_process = None
            return False

    def _init_dji_stream_reader(self):
        """初始化大疆无人机流读取器"""
        try:
            # 获取分辨率
            if self.dji_stream_info:
                width, height = DJIStreamDecoder.get_dji_resolution(self.dji_stream_info['resolution'])
            else:
                width, height = Config.DJI_DEFAULT_RESOLUTION

            self.frame_size = width * height * 3  # 3通道 (BGR)
            self.frame_shape = (height, width, 3)

            # 构建大疆专用解码命令
            command = DJIStreamDecoder.get_dji_decoder_command(
                self.task_data['stream_url'],
                width,
                height
            )

            logger.info(f"启动大疆专用解码器: {' '.join(command)}")

            # 创建日志文件
            log_file = open('dji_stream.log', 'w') if Config.DJI_DEBUG_LOGGING else subprocess.DEVNULL

            self.ffmpeg_input = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=log_file,
                bufsize=10 ** 8  # 大缓冲区
            )

            logger.info(f"大疆流配置: {width}x{height}, 帧大小: {self.frame_size}字节")
            return True
        except Exception as e:
            logger.error(f"初始化大疆流读取器失败: {str(e)}")
            return False

    def _read_dji_frame(self):
        """读取大疆无人机视频帧"""
        if not self.ffmpeg_input:
            if not self._init_dji_stream_reader():
                return None

        try:
            # 从FFmpeg管道读取原始帧数据
            raw_frame = self.ffmpeg_input.stdout.read(self.frame_size)
            if not raw_frame:
                logger.warning("读取到空帧数据")
                return None

            if len(raw_frame) != self.frame_size:
                logger.warning(f"读取帧数据不完整: {len(raw_frame)}/{self.frame_size}字节")
                return None

            # 将字节数据转换为numpy数组
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape(self.frame_shape)
            return frame
        except Exception as e:
            logger.error(f"读取大疆帧失败: {str(e)}")
            return None

    def _run_detection(self):
        """检测主循环"""
        # # 对于大疆无人机流，使用专用解码器
        # if self.dji_stream_info:
        #     return self._run_dji_detection()

        # 对于普通流，使用标准方法
        return self._run_standard_detection()

    def _run_dji_detection(self):
        """大疆无人机流专用检测循环"""
        # 初始化大疆流读取器
        if not self._init_dji_stream_reader():
            logger.error("无法初始化大疆流读取器")
            return

        # 加载所有需要的模型
        for algo_id in self.task_data['algorithm_ids']:
            self._load_model(algo_id)

        # 主处理循环
        frame_count = 0
        last_report_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10
        last_success_time = time.time()
        print("1111111111111")
        # 添加推流重启计数器
        ffmpeg_restart_count = 0

        while self.is_running:
            print("22222222222")
            start_time = time.time()

            # 检查超时
            if time.time() - last_success_time > Config.DJI_STREAM_TIMEOUT:
                logger.error(f"流读取超时 ({Config.DJI_STREAM_TIMEOUT}秒无数据)")
                break

            # 读取帧
            frame = self._read_dji_frame()
            if frame is None:
                consecutive_failures += 1
                logger.warning(f"读取视频帧失败 ({consecutive_failures}/{max_consecutive_failures})")

                if consecutive_failures >= max_consecutive_failures:
                    logger.error("连续读取失败次数过多，尝试重新连接视频流")
                    self.ffmpeg_input.terminate()
                    self.ffmpeg_input = None
                    consecutive_failures = 0
                    if not self._init_dji_stream_reader():
                        logger.error("重新连接视频流失败")
                        break
                else:
                    time.sleep(0.1)
                continue

            # 成功读取帧
            consecutive_failures = 0
            last_success_time = time.time()

            print("33333333")
            # 调整帧大小
            processed_frame = self._resize_frame(frame)

            print("4444444444444")
            logger.warning("55555555555")
            # 处理所有算法
            algorithm_results = []
            for algo_id in self.task_data['algorithm_ids']:
                algo_result = self._process_frame(processed_frame, algo_id)

                logger.exception(f"algo_result:{algo_result}")

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

                # # todo 推送到RTMP
                # push_success = self._push_frame_to_rtmp(output_frame)
                # if not push_success:
                #     logger.warning("推送到RTMP失败")
                #     ffmpeg_restart_count += 1
                #
                #     # 检查是否超过重启限制
                #     if ffmpeg_restart_count > Config.FFMPEG_RESTART_LIMIT:
                #         logger.error("FFmpeg重启次数超过限制，停止推流")
                #         break
                # else:
                #     ffmpeg_restart_count = 0  # 重置计数器

            # 控制处理速率
            elapsed = time.time() - start_time
            sleep_time = max(0, Config.FRAME_PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_time)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"处理帧数: {frame_count}, 当前FPS: {1 / max(0.001, elapsed + sleep_time):.1f}")

        # 清理资源
        if self.ffmpeg_input:
            self.ffmpeg_input.terminate()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        logger.info(f"检测任务结束: {self.task_data['flight_id']}")

    def _run_standard_detection(self):
        """标准视频流检测循环"""
        # 打开视频流
        cap = None
        max_retries = 5
        retry_count = 0
        stream_url = self.task_data['stream_url']

        # 使用FFmpeg后端
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        logger.info(f"尝试使用FFmpeg后端打开视频流: {stream_url}")

        # 重试机制
        while retry_count < max_retries and self.is_running:
            if cap.isOpened():
                break
            else:
                logger.warning(f"视频流连接失败 ({retry_count + 1}/{max_retries})")
                time.sleep(2)
                retry_count += 1
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

        if not cap or not cap.isOpened():
            logger.error(f"无法打开视频流: {stream_url}")
            return

        # 获取视频流基本信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # 默认帧率
            logger.warning(f"无法获取视频流FPS，使用默认值: {fps}")

        # 获取视频流宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"视频流信息: {width}x{height} @ {fps:.2f} FPS")

        # 加载所有需要的模型
        for algo_id in self.task_data['algorithm_ids']:
            self._load_model(algo_id)

        # 主处理循环
        frame_count = 0
        last_report_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10

        while self.is_running and cap.isOpened():
            start_time = time.time()

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning(f"读取视频帧失败 ({consecutive_failures}/{max_consecutive_failures})")

                if consecutive_failures >= max_consecutive_failures:
                    logger.error("连续读取失败次数过多，尝试重新连接视频流")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    consecutive_failures = 0
                    if not cap.isOpened():
                        logger.error("重新连接视频流失败")
                        break
                else:
                    time.sleep(0.1)
                continue

            consecutive_failures = 0  # 重置失败计数

            # 调整帧大小
            processed_frame = self._resize_frame(frame)

            # 处理所有算法
            algorithm_results = []
            for algo_id in self.task_data['algorithm_ids']:
                algo_result = self._process_frame(processed_frame, algo_id)
                if algo_result:
                    algorithm_results.append(algo_result)

            # print("algorithm_results")
            # print(algorithm_results)
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

            # # todo 推送到RTMP
            # if not self._push_frame_to_rtmp(output_frame):
            #     logger.warning("推送到RTMP失败")
            #
            # # 控制处理速率
            # elapsed = time.time() - start_time
            # sleep_time = max(0, Config.FRAME_PROCESS_INTERVAL - elapsed)
            # time.sleep(sleep_time)
            #
            # frame_count += 1
            # if frame_count % 100 == 0:
            #     logger.info(f"处理帧数: {frame_count}, 当前FPS: {1 / max(0.001, elapsed + sleep_time):.1f}")

        # 清理资源
        cap.release()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        logger.info(f"检测任务结束: {self.task_data['flight_id']}")

    def _draw_detection_results(self, frame, algorithm_results):
        """在帧上绘制检测结果：目标框和标签"""
        # 遍历每个算法的结果
        for algo_result in algorithm_results:
            detections = algo_result.get('detections', [])

            # 遍历该算法的所有检测结果
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']

                # 解析边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox

                # 绘制边界框
                color = (0, 255, 0)  # 绿色框
                thickness = 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

                # 绘制标签背景
                label = f"{class_name}: {confidence:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame,
                              (int(x1), int(y1) - text_size[1] - 5),
                              (int(x1) + text_size[0], int(y1) - 5),
                              color, cv2.FILLED)

                # 绘制标签文本
                cv2.putText(frame, label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
        confidence = algo_config.get('config', {}).get('confidence', 0.5)

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


                # logger.exception(f"outputs:{outputs}")

                # 后处理结果
                detections = self._postprocess_onnx_results(
                    outputs,
                    original_shape,
                    confidence_threshold=confidence,
                    algorithm_id=algorithm_id
                )

                # 构建结果
                return {
                    "algorithm_id": algorithm_id,
                    "detections": detections  # 包含所有检测框的详细信息
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
        """后处理ONNX模型输出，返回检测框详细信息"""
        # 获取算法配置
        algo_config = AlgorithmManager.get_algorithm(algorithm_id)
        if not algo_config:
            logger.error(f"算法配置未找到: {algorithm_id}")
            return []

        # 获取原始尺寸
        orig_h, orig_w = original_shape
        detections = []

        # 获取检测结果 - 假设第一个输出是检测结果
        # [batch, num_detections, 6] -> [x1, y1, x2, y2, confidence, class_id]
        detections_output = outputs[0][0]

        for detection in detections_output:
            *bbox, conf, class_id = detection
            # logger.exception(f"bbox:{bbox}")
            # logger.exception(f"conf:{conf}")
            # logger.exception(f"class_id:{class_id}")

            # 检查置信度阈值
            if conf < confidence_threshold:
                continue

            # 转换为整数类别
            class_id = int(class_id)

            # 获取目标名称
            class_name = self._get_class_name_by_id(algo_config, class_id)
            logger.exception(f"class_name:{class_name}")
            # 保存检测结果详细信息
            detections.append({
                'bbox': bbox,  # [x1, y1, x2, y2]
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(conf)
            })

        return detections

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
        """上报检测结果 - 参数类型保持不变"""
        # 编码图像为base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 构建上报数据 - 保持原有结构
        report_data = {
            "flight_id": self.task_data['flight_id'],
            "image_base64": image_base64,
            "detection_timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "algorithm": []  # 保持原有结构
        }

        # 将新的检测结果转换为原有结构
        for algo_result in algorithm_results:
            # 创建目标计数统计
            target_counts = defaultdict(int)
            for detection in algo_result.get('detections', []):
                target_key = detection.get('class_name', f"class_{detection.get('class_id', 'unknown')}")
                target_counts[target_key] += 1

            # 转换为原有的目标列表格式
            targets = [{"target_key": k, "number": v} for k, v in target_counts.items()]

            # 添加到上报数据
            report_data["algorithm"].append({
                "algorithm_id": algo_result['algorithm_id'],
                "targets": targets
            })

        # 发送上报请求
        try:
            response = requests.post(
                self.task_data['callback_api'],
                json=report_data,
                timeout=3
            )

            # 检查响应是否符合统一格式
            if response.status_code != 200:
                logger.error(f"结果上报失败: HTTP {response.status_code}")
            else:
                resp_data = response.json()
                if resp_data.get('code', 0) != 0:
                    logger.error(f"结果上报返回错误: {resp_data.get('message', '')}")
        except Exception as e:
            logger.error(f"结果上报异常: {str(e)}")

    def _get_class_name_by_id(self, algo_config, class_id):
        """根据类别ID获取目标名称"""
        # # 检查算法配置中是否有目标映射
        # if 'targets' not in algo_config.get('config', {}):
        #     return f"class_{class_id}"

        # 查找匹配的目标
        for target in algo_config['config']['targets']:
            # 假设配置中有class_id字段或使用索引
            if  target['target_key'] == class_id:
                return target['target_name']

        # 如果没有匹配，使用默认格式
        return f"class_{class_id}"
