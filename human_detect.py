import cv2
import numpy as np

cap = cv2.VideoCapture('Photos/WhatsApp Video 2020-12-12 at 22.24.21.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX

fullbody_cascade = cv2.CascadeClassifier('Links/haarcascade_fullbody.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    full_body = fullbody_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in full_body:  #x: sol, y: alt, w: widht, h: height
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, str('person'), (x, y + h), font, 1, (0, 0, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()