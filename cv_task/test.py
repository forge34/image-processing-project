import cv2
from model import detect_face

expected_faces = 2

img = cv2.imread("./assets/images.jpeg")
result, n, t = detect_face(img)
print("Time taken: ", t)
print("Success rate: ", n == expected_faces, "%")
