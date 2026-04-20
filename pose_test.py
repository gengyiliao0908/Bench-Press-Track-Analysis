import cv2
import mediapipe as mp
#初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
#測試視訊鏡頭(0代表預設的電腦鏡頭)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    #讀取鏡頭畫面
    success, frame = cap.read() 
    if not success:
        print("鏡頭讀取fail")
        break
    #OpenCV 讀進來的顏色是BGR，但MediaPipe處理需要RGB，所以要轉換
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #將畫面餵給模型去抓骨架
    results = pose.process(image_rgb)
    #如果有成功抓到骨架
    if results.pose_landmarks:
        #把骨架連線畫在畫面上
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #抓出右手腕(MediaPipe規定右手腕的編號是16)
        right_wrist = results.pose_landmarks.landmark[16]
        print(f"右手腕 X座標: {right_wrist.x:.3f}, Y座標: {right_wrist.y:.3f}")
    #顯示視窗畫面
    cv2.imshow('MediaPipe Test', frame)
    #按下鍵盤的q鍵就可以關閉視窗
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#釋放資源與關閉視窗
cap.release()
cv2.destroyAllWindows()