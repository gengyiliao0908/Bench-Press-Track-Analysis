import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 打開視訊鏡頭 (0 代表預設的電腦鏡頭)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 讀取鏡頭畫面
    success, frame = cap.read() 
    if not success:
        print("無法讀取鏡頭，請確認鏡頭是否有被其他程式佔用。")
        break

    # OpenCV 讀進來的顏色是 BGR，但 MediaPipe 處理需要 RGB，所以要轉換
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 將畫面餵給模型去抓骨架
    results = pose.process(image_rgb)

    # 如果有成功抓到人體骨架
    if results.pose_landmarks:
        # 把骨架連線畫在畫面上
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 抓出右手腕 (MediaPipe 規定右手腕的編號是 16)
        right_wrist = results.pose_landmarks.landmark[16]
        print(f"右手腕 X座標: {right_wrist.x:.3f}, Y座標: {right_wrist.y:.3f}")

    # 顯示視窗畫面
    cv2.imshow('MediaPipe Test', frame)

    # 按下鍵盤的 'q' 鍵就可以關閉視窗
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 釋放資源與關閉視窗
cap.release()
cv2.destroyAllWindows()