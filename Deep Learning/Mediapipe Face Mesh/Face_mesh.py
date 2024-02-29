import mediapipe as mp
import cv2
import time

mp_face_mesh = mp.solutions.face_mesh # Yüzü ağlara bölmek için
mp_drawing = mp.solutions.drawing_utils # yüzü boyamak için
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1, ) # thickness = kalınlık, resmi çizdirmek için


video_path = "face.mp4"
cap = cv2.VideoCapture(video_path)

with mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    while cap.isOpened():
        _, frame = cap.read()
        start = time.time() # zamanlayıcıyı başlat
        # opencv resimleri BGR formatında yaparken mediapipe RGB formatında yapar
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = face_mesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # resmi çizdir
        if(mesh.multi_face_landmarks):
            for face_landmarks in mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(image = frame, landmark_list = face_landmarks, connections = mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec = drawing_spec, connection_drawing_spec = drawing_spec)

        end = time.time() # zamanlayıcıyı durdur
        if((end-start)==0): # hata vermesi önlendi
            end = 1
        fps = 1/(end-start) # video fps
        cv2.putText(frame, f"FPS : {int(fps)}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        cv2.imshow("Frame", frame)

        if(cv2.waitKey(10) &0xFF == ord("q")):
            break

cap.release()
cv2.destroyAllWindows()
