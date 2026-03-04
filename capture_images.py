import cv2
import os

cam = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_id = input("Enter Face ID: ")

count = 0

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        count += 1

        file_name = f"dataset/User.{face_id}.{count}.jpg"

        cv2.imwrite(file_name, gray[y:y+h,x:x+w])

        cv2.imshow("Capturing Faces", img)

    if cv2.waitKey(1) == 27 or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()

print("Images captured successfully")