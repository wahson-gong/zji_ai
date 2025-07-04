import json
from datetime import datetime

from flask import request, jsonify
from models.detection import DetectionTask
import threading
import logging
import os
import time

logger = logging.getLogger(__name__)

# 存储活动检测任务
active_tasks = {}
task_lock = threading.Lock()

DATA_DIR = "received_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def start_ai_stream_push():
    # req_data = request.get_json()
    # if not req_data:
    #     return jsonify({"code": 400, "data": None, "message": "请求体为空"}), 400
    #
    # # 生成带时间戳的文件名
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"data_{timestamp}.json"
    # filepath = os.path.join(DATA_DIR, filename)
    #
    # # 保存到文件
    # with open(filepath, 'w') as f:
    #     json.dump(req_data, f, indent=4)

    """启动AI流检测接口并返回AI直播流地址"""
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"code": 400, "data": None, "message": "请求体为空"}), 400

        # 验证必要字段
        required_fields = [
            'flight_id', 'stream_url', 'rtmp_host',
            'algorithm_ids', 'ntp', 'callback_api'
        ]
        for field in required_fields:
            if field not in req_data:
                return jsonify({
                    "code": 400,
                    "data": None,
                    "message": f"缺少必要字段: {field}"
                }), 400

        flight_id = req_data['flight_id']

        # 检查任务是否已存在
        with task_lock:
            if flight_id in active_tasks:
                return jsonify({
                    "code": 409,
                    "data": None,
                    "message": "该航班检测任务已存在"
                }), 409

            # 创建AI流地址
            ai_stream_url = f"{req_data['rtmp_host'].rstrip('/')}/{flight_id}"

            # 创建新任务
            task = DetectionTask(req_data, ai_stream_url)
            if task.start():
                active_tasks[flight_id] = task

                # 构建响应数据
                response_data = {
                    "code": 0,
                    "data": {
                        "ai_stream_url": ai_stream_url,
                        "flight_id": flight_id
                    },
                    "message": "检测任务已启动"
                }
                return jsonify(response_data)
            else:
                return jsonify({
                    "code": 500,
                    "data": None,
                    "message": "无法启动检测任务"
                }), 500

    except Exception as e:
        logger.exception("启动检测任务失败")
        return jsonify({
            "code": 500,
            "data": None,
            "message": f"服务器错误: {str(e)}"
        }), 500



def stop_ai_stream_push():
    """停止AI流检测接口"""
    try:
        req_data = request.get_json()
        if not req_data or 'flight_id' not in req_data:
            return jsonify({
                "code": 400,
                "data": None,
                "message": "缺少flight_id"
            }), 400

        flight_id = req_data['flight_id']

        with task_lock:
            if flight_id in active_tasks:
                task = active_tasks.pop(flight_id)
                task.stop()
                return jsonify({
                    "code": 0,
                    "data": flight_id,
                    "message": "检测任务已停止"
                })
            else:
                return jsonify({
                    "code": 404,
                    "data": None,
                    "message": "未找到检测任务"
                }), 404

    except Exception as e:
        logger.exception("停止检测任务失败")
        return jsonify({
            "code": 500,
            "data": None,
            "message": f"服务器错误: {str(e)}"
        }), 500