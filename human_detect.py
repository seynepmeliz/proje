import cv2
import numpy as np

cap = cv2.VideoCapture('')
font = cv2.FONT_HERSHEY_SIMPLEX

fullbody_cascade = cv2.CascadeClassifier('Files/haarcascade_fullbody.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    full_body = fullbody_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in full_body:  #x: sol, y: alt, w: widht, h: height
        cv2.rectangle(frame(x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, str('person'), (x, y + h), font, (0, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()