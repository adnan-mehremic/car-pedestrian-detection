# Pedestrian detection

import cv2

haarcascade_fullbody = cv2.CascadeClassifier('haarcascade/haarcascade_fullbody.xml')
video = cv2.VideoCapture('/pedestrian/path')

if video.isOpened() == False:
    print("Error opening video")
else:
    while video.isOpened():

        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_LINEAR)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_detection = haarcascade_fullbody.detectMultiScale(gray_frame, 1.2, 5)

            for (x, y, width, height) in body_detection:
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 255), 2)
                cv2.imshow('Pedestrian detection', frame)

            # for exit press 'q' on keyboard
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break


video.release()
cv2.destroyAllWindows()
