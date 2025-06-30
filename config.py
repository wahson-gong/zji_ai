import os


class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    DEBUG = os.environ.get('DEBUG', 'False') == 'True'

    # 存储路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ALGORITHM_STORAGE = os.path.join(BASE_DIR, 'storage', 'algorithms')
    TEMP_STORAGE = os.path.join(BASE_DIR, 'storage', 'temp')

    # 确保目录存在
    os.makedirs(ALGORITHM_STORAGE, exist_ok=True)
    os.makedirs(TEMP_STORAGE, exist_ok=True)

    # 检测参数
    FRAME_PROCESS_INTERVAL = 0.1  # 处理帧的时间间隔(秒)
    MAX_FRAME_SIZE = (640, 640)  # 最大帧尺寸