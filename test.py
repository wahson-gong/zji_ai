import cv2

stream_url = "rtmp://192.168.0.20/live/1581F5FJD22BG00BZB2T-67-0-0"
cap = cv2.VideoCapture(stream_url)

if cap.isOpened():
    print("成功连接视频流")
    width = 640
    height = 640
    fps = 25
    print(f"视频信息: {width}x{height} @ {fps:.2f} FPS")

    # 尝试读取一帧
    ret, frame = cap.read()
    if ret:
        print("成功读取帧")
        cv2.imwrite("test_frame.jpg", frame)
    else:
        print("读取帧失败")
else:
    print("无法连接视频流")

cap.release()