import cv2
from model_haar import detect_face
from model_yunet import detect_face_yunet

expected_faces = 2

img = cv2.imread("./assets/test1.jpeg")
result, n, t = detect_face_yunet(img)
cv2.imwrite("result.png",result)
print("Time taken: ", t)
print("Success ", n == expected_faces)
