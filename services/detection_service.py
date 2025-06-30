from flask import request, jsonify
from models.detection import DetectionTask
import threading

# 存储活动检测任务
active_tasks = {}
task_lock = threading.Lock()


def start_ai_stream_push():
    """启动AI流检测接口"""
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"code": 400, "message": "请求体为空"}), 400

        # 验证必要字段
        required_fields = [
            'flight_id', 'stream_url', 'rtmp_host',
            'algorithm_ids', 'ntp', 'callback_api'
        ]
        for field in required_fields:
            if field not in req_data:
                return jsonify({
                    "code": 400,
                    "message": f"缺少必要字段: {field}"
                }), 400

        flight_id = req_data['flight_id']

        # 检查任务是否已存在
        with task_lock:
            if flight_id in active_tasks:
                return jsonify({
                    "code": 409,
                    "message": "该航班检测任务已存在"
                }), 409

            # 创建新任务
            task = DetectionTask(req_data)
            if task.start():
                active_tasks[flight_id] = task
                return jsonify({
                    "code": 0,
                    "message": "检测任务已启动"
                })
            else:
                return jsonify({
                    "code": 500,
                    "message": "无法启动检测任务"
                }), 500

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": f"服务器错误: {str(e)}"
        }), 500


def stop_ai_stream_push():
    """停止AI流检测接口"""
    try:
        req_data = request.get_json()
        if not req_data or 'flight_id' not in req_data:
            return jsonify({"code": 400, "message": "缺少flight_id"}), 400

        flight_id = req_data['flight_id']

        with task_lock:
            if flight_id in active_tasks:
                task = active_tasks.pop(flight_id)
                task.stop()
                return jsonify({
                    "code": 0,
                    "message": "检测任务已停止"
                })
            else:
                return jsonify({
                    "code": 404,
                    "message": "未找到检测任务"
                }), 404

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": f"服务器错误: {str(e)}"
        }), 500