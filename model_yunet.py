import cv2
import time
from pathlib import Path

model_path = Path("assets/face_detection_yunet_2023mar.onnx")

detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),score_threshold=0.6,nms_threshold=0.3,top_k=5000)

def detect_face_yunet(img):
    face_img = img.copy()
    h, w = face_img.shape[:2]
    detector.setInputSize((w, h))
    thickness = max(1, int(min(img.shape[:2]) * 0.005))
    start_time = time.time()
    _, faces = detector.detect(face_img)
    end_time = time.time()

    if faces is not None:
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            cv2.rectangle(face_img, (x, y), (x + fw, y + fh), (0, 255, 0), thickness)

    num_faces = len(faces) if faces is not None else 0
    return face_img, num_faces, (end_time - start_time)