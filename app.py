from flask import Flask
from config import Config
from services.algorithm_service import import_ai_algorithm
from services.detection_service import start_ai_stream_push, stop_ai_stream_push
from services.callback_service import handle_algorithm_callback

app = Flask(__name__)
app.config.from_object(Config)

# 注册路由
app.add_url_rule('/import_ai_algorithm', view_func=import_ai_algorithm, methods=['POST'])
app.add_url_rule('/start_ai_stream_push', view_func=start_ai_stream_push, methods=['POST'])
app.add_url_rule('/stop_ai_stream_push', view_func=stop_ai_stream_push, methods=['POST'])
app.add_url_rule('/callback', view_func=handle_algorithm_callback, methods=['POST'])

@app.route('/')
def health_check():
    return "AI Detection Server is running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])