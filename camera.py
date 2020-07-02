import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        directory = "/media/a/E09AADC89AAD9C12/Downloads/Shared Folder/Emotion Recognition Out Of Facial Expression/Project/videos/"
        
        self.video = cv2.VideoCapture(directory+"sent_anlys_me.mp4")

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            #cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.putText(fr, pred, (x+72, y-5), cv2.FONT_HERSHEY_COMPLEX, 1.6, (255, 200, 150), 3)
            cv2.putText(fr, "Sentiment Analysis Result", (x-48, y-56), cv2.FONT_HERSHEY_COMPLEX, 1, (40, 40, 40), 2)
            cv2.putText(fr, "Sentiment Analysis Result", (x-50, y-59), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 255), 2)
            cv2.putText(fr, "Facial Expression", (x+27, y-91), cv2.FONT_HERSHEY_COMPLEX, 1.001, (40, 40, 40), 2)
            cv2.putText(fr, "Facial Expression", (x+28, y-93), cv2.FONT_HERSHEY_COMPLEX, 1, (70, 200, 20), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h+19),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
