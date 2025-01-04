import cv2

# 加载预训练的Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# 打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 读取当前帧
    ret, frame = video_capture.read()

    # 将图像转换为灰度图，因为Haar级联在灰度图上工作更好
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # 表示每次图像尺寸减小的比例
        minNeighbors=5,   # 每个候选矩形应保留的邻居数
        minSize=(30, 30), # 要检测的最小物体大小
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 在检测到的人脸上绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 显示结果帧
    cv2.imshow('Video', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 完成后释放摄像头并关闭窗口
video_capture.release()
cv2.destroyAllWindows()