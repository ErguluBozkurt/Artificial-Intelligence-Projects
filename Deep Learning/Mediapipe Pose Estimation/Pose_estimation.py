import mediapipe as mp
import cv2
import time

mp_pose = mp.solutions.pose 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

video_path = "pose.mp4"
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():   
        _, frame = cap.read()
        start = time.time() # zamanlayıcıyı başlat
        # opencv resimleri BGR formatında yaparken mediapipe RGB formatında yapar
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(frame, result.pose.landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style)

        end = time.time() # zamanlayıcıyı durdur
        if((end-start)==0): 
            end = 1
        fps = 1/(end-start) # video fps
        cv2.putText(frame, f"FPS : {int(fps)}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        cv2.imshow("Frame", frame)

        if(cv2.waitKey(10) &0xFF == ord("q")):
            break

cap.release()
cv2.destroyAllWindows()
