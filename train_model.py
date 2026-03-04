import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    faceSamples=[]
    ids=[]

    for imagePath in imagePaths:

        try:
            PIL_img = Image.open(imagePath).convert('L')
        except:
            continue

        img_numpy = np.array(PIL_img,'uint8')

        file_name = os.path.split(imagePath)[-1]
        parts = file_name.split(".")

        if len(parts) < 3:
            continue

        try:
            id = int(parts[1])
        except:
            continue

        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:

            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples, ids


print("Training faces...")

faces, ids = getImagesAndLabels(path)

if len(faces) == 0:
    print("No images found in dataset")
    exit()

recognizer.train(faces, np.array(ids))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.write('trainer/trainer.yml')

print("Training completed successfully")