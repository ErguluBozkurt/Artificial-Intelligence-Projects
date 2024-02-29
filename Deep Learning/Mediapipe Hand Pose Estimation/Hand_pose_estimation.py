import mediapipe as mp
import cv2
import time

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 

video_path = "hand.mp4"
cap = cv2.VideoCapture(video_path)

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:
    while cap.isOpened():
        _, frame = cap.read()
        start = time.time() # zamanlayıcıyı başlat
        # opencv resimleri BGR formatında yaparken mediapipe RGB formatında yapar
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # resmi çizdir
        if(result.multi_hand_landmarks):
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image = frame, landmark_list = hand_landmarks, connections = mp_hands.HAND_CONNECTIONS)

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
