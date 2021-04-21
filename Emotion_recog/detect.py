from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('./haarcascade.xml')
classifier =load_model('./checkpoint.h5')
class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
cap = cv2.imread('./images/train/angry/0.jpg')
# print(cap)
# while True:
# Grab a single frame of video
# ret, frame = cap.read()
# labels = []
frame = cap

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
# rect,face,image = face_detector(frame)


    if np.sum([roi_gray])!=0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

    # make a prediction on the ROI, then lookup the class

        preds = classifier.predict(roi)[0]
        label=class_labels[preds.argmax()]
        label_position = (x,y)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
        cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
cv2.imshow('Emotion Detector',frame)


cap.release()
cv2.destroyAllWindows()