import os
import json
import yaml
import time
import requests
from datetime import datetime
from tqdm import tqdm  # 用于显示进度条
from config import Config
import logging
import onnxruntime as ort

logger = logging.getLogger(__name__)


class AlgorithmManager:
    @staticmethod
    def save_algorithm(data):
        """保存算法配置和生成YOLO配置文件"""
        algo_dir = os.path.join(Config.ALGORITHM_STORAGE, data['algorithm_id'])
        os.makedirs(algo_dir, exist_ok=True)

        # 保存算法配置
        config_path = os.path.join(algo_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

        # 生成YOLO配置文件
        yaml_path = AlgorithmManager._generate_yolo_config(data, algo_dir)

        # 下载模型文件 - 使用算法ID作为文件名
        model_path = os.path.join(algo_dir, f"{data['algorithm_id']}.onnx")
        download_success = AlgorithmManager._download_model(
            data['model_file_url'],
            model_path
        )

        if not download_success:
            logger.error(f"模型下载失败: {data['algorithm_id']}")
            return None

        # 更新配置中的模型路径
        data['local_model_path'] = model_path
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

        return {
            "algorithm_id": data['algorithm_id'],
            "config_path": config_path,
            "yaml_path": yaml_path,
            "model_path": model_path,
            "model_dir": algo_dir
        }

    @staticmethod
    def _download_model(url, save_path, max_retries=3, timeout=30):
        """
        下载模型文件 - 简单实现
        使用算法ID作为文件名保存
        """
        # 尝试多次下载
        for attempt in range(max_retries):
            try:
                logger.info(f"开始下载模型: {url} -> {save_path}")

                # 发起HTTP GET请求
                response = requests.get(
                    url,
                    stream=True,  # 流式传输大文件
                    timeout=timeout
                )

                # 检查响应状态
                response.raise_for_status()

                # 获取文件总大小（可能不可用）
                total_size = int(response.headers.get('content-length', 0))

                # 创建进度条
                progress_bar = tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=f"下载模型"
                )

                # 写入文件
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # 过滤掉保持连接的新块
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                progress_bar.close()

                # 检查文件是否下载完整
                if total_size > 0 and progress_bar.n != total_size:
                    logger.error(f"下载不完整: {progress_bar.n}/{total_size} bytes")
                    return False

                logger.info(f"模型下载成功: {save_path}")
                return True

            except (requests.RequestException, IOError) as e:
                logger.error(f"下载尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                time.sleep(2)  # 重试前等待

        return False

    @staticmethod
    def _generate_yolo_config(data, algo_dir):
        """生成YOLO模型配置文件"""
        # 提取目标类别名称
        class_names = [target['target_name'] for target in data['targets']]

        yolo_config = {
            'path': algo_dir,
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names),
            'roboflow': {
                'license': "CC BY 4.0",
                'created': datetime.now().isoformat()
            }
        }

        yaml_path = os.path.join(algo_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yolo_config, f, sort_keys=False)

        return yaml_path

    @staticmethod
    def get_algorithm(algorithm_id):
        """获取算法配置"""
        algo_dir = os.path.join(Config.ALGORITHM_STORAGE, algorithm_id)
        config_path = os.path.join(algo_dir, 'config.json')

        if not os.path.exists(config_path):
            return None

        with open(config_path, 'r') as f:
            config = json.load(f)

        # 添加模型路径信息
        if 'local_model_path' not in config:
            # 尝试查找模型文件
            model_path = os.path.join(algo_dir, f"{algorithm_id}.pt")
            if os.path.exists(model_path):
                config['local_model_path'] = model_path

        return config