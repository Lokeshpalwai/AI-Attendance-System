import cv2
import numpy as np
import csv
import os
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

# Load student database
students = {}

with open("students.csv","r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        students[int(row["FaceID"])] = (
            row["StudentName"],
            row["StudentID"],
            row["Class"]
        )

# create daily attendance file
today = datetime.now().strftime("%Y-%m-%d")

attendance_file = f"attendance_{today}.csv"

if not os.path.exists(attendance_file):

    with open(attendance_file,"w",newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Student Name","Student ID","Class","Time"])

marked_students = set()

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 50 and id in students:

            name, sid, cls = students[id]

            if name not in marked_students:

                now = datetime.now()

                time = now.strftime("%H:%M:%S")

                with open(attendance_file,"a",newline="") as f:

                    writer = csv.writer(f)

                    writer.writerow([name,sid,cls,time])

                marked_students.add(name)

        else:

            name = "Unknown"

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(img,name,(x,y-10),font,1,(255,255,255),2)

    cv2.imshow("AI Attendance System",img)

    if cv2.waitKey(1)==27:
        break

cam.release()
cv2.destroyAllWindows()