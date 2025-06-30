from flask import Flask
from config import Config
from services.algorithm_service import import_ai_algorithm
from services.detection_service import start_ai_stream_push, stop_ai_stream_push

app = Flask(__name__)
app.config.from_object(Config)

# 注册路由
app.add_url_rule('/api/v1/algorithm/import', view_func=import_ai_algorithm, methods=['POST'])
app.add_url_rule('/api/v1/ai/stream/start', view_func=start_ai_stream_push, methods=['POST'])
app.add_url_rule('/api/v1/ai/stream/stop', view_func=stop_ai_stream_push, methods=['POST'])


@app.route('/')
def health_check():
    return "AI Detection Server is running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])