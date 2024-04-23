import cv2 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()

    cv2.imshow('output', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()