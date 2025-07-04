from flask import request, jsonify
from models.algorithm import AlgorithmManager
import requests
import logging

logger = logging.getLogger(__name__)


def import_ai_algorithm():
    """导入AI算法接口"""
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"code": 400, "data": None, "message": "请求体为空"}), 400

        # 验证必要字段
        required_fields = [
            'algorithm_id', 'algorithm_name', 'model_file_url',
            'model_path', 'callback_url', 'config'  # 注意：targets 现在在 config 下
        ]
        for field in required_fields:
            if field not in req_data:
                return jsonify({
                    "code": 400,
                    "data": None,
                    "message": f"缺少必要字段: {field}"
                }), 400

        # 检查 config 中的 targets 是否存在
        if 'targets' not in req_data['config']:
            return jsonify({
                "code": 400,
                "data": None,
                "message": "config 中缺少必要字段: targets"
            }), 400

        # 保存算法
        result = AlgorithmManager.save_algorithm(req_data)
        if not result:
            return jsonify({
                "code": 500,
                "data": None,
                "message": "算法导入失败"
            }), 500

        # 准备响应
        response_data = {
            "code": 0,
            "data": result['algorithm_id'],
            "message": "算法导入成功"
        }

        # 异步调用回调URL (GET请求)
        try:
            callback_url = req_data['callback_url']
            logger.info(f"调用回调URL: {callback_url}")

            # 使用GET请求调用回调URL
            callback_resp = requests.get(
                callback_url,
                params={
                    "algorithm_id": req_data['algorithm_id'],
                    "status": "success"
                },
                timeout=3
            )

            # 检查回调响应
            if callback_resp.status_code != 200:
                logger.error(f"回调响应错误: {callback_resp.status_code}")
            else:
                callback_data = callback_resp.json()
                if callback_data.get('code', 0) != 0:
                    logger.error(f"回调返回非零状态码: {callback_data.get('message', '')}")
                else:
                    logger.info(f"回调成功: {callback_data.get('message', '')}")
        except Exception as e:
            logger.error(f"回调失败: {str(e)}")

        return jsonify(response_data)

    except Exception as e:
        logger.exception("算法导入失败")
        return jsonify({
            "code": 500,
            "data": None,
            "message": f"服务器错误: {str(e)}"
        }), 500