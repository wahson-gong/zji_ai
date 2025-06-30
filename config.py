import os


class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    DEBUG = os.environ.get('DEBUG', 'False') == 'False'

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

    # ONNX推理配置
    ONNX_EXECUTION_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 按顺序尝试
    ONNX_OPTIMIZATION_LEVEL = 3  # ORT_ENABLE_ALL

    # FFmpeg 配置
    FFMPEG_PATH = 'ffmpeg'  # 默认使用系统路径中的 ffmpeg
    # 如果系统找不到 ffmpeg，可以在这里指定完整路径
    # Windows 示例:
    # FFMPEG_PATH = r'C:\ffmpeg\bin\ffmpeg.exe'

    # Linux/macOS 示例:
    # FFMPEG_PATH = '/usr/bin/ffmpeg'

    # 大疆无人机配置
    DJI_DEFAULT_RESOLUTION = (1920, 1080)  # 默认分辨率
    DJI_DEFAULT_FPS = 30  # 默认帧率

    # FFmpeg 高级配置
    FFMPEG_READ_TIMEOUT = 10  # 读取超时(秒)
    FFMPEG_RECONNECT_DELAY = 2  # 重连延迟(秒)

    # FFmpeg 配置
    FFMPEG_DEBUG_LOGGING = True  # 启用FFmpeg调试日志
    FFMPEG_RESTART_LIMIT = 5  # FFmpeg最大重启次数
    FFMPEG_BUFFER_SIZE = 10 ** 7  # 10MB缓冲区

    # 推流配置
    RTMP_CONNECT_TIMEOUT = 5  # RTMP连接超时(秒)
    RTMP_RECONNECT_DELAY = 1  # 重连延迟(秒)