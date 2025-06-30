from flask import request, jsonify
from models.algorithm import AlgorithmManager
import requests


def import_ai_algorithm():
    """导入AI算法接口"""
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"code": 400, "message": "请求体为空"}), 400

        # 验证必要字段
        required_fields = [
            'algorithm_id', 'algorithm_name', 'model_file_url',
            'model_path', 'targets', 'callback_url'
        ]
        for field in required_fields:
            if field not in req_data:
                return jsonify({
                    "code": 400,
                    "message": f"缺少必要字段: {field}"
                }), 400

        # 保存算法
        result = AlgorithmManager.save_algorithm(req_data)

        # 准备响应
        response_data = {
            "code": 0,
            "data": result['algorithm_id'],
            "message": "算法导入成功"
        }

        # 异步调用回调URL
        try:
            callback_resp = requests.post(
                req_data['callback_url'],
                json={"status": "success", "algorithm_id": req_data['algorithm_id']},
                timeout=3
            )
        except Exception as e:
            print(f"回调失败: {str(e)}")

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": f"服务器错误: {str(e)}"
        }), 500