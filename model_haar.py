import cv2
import time
from pathlib import Path

model_path = Path("assets/haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(model_path)

def detect_face(img):
    face_img = img.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    thickness = max(1, int(min(img.shape[:2]) * 0.005))
    start_time = time.time()
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    end_time = time.time()
    for x, y, w, h in face_rect:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), thickness)

    num_faces = len(face_rect)
    return face_img, num_faces, (end_time - start_time)
