# open collecting data face1 and run , then the images will be stored in face1 folder
# then run facelookdoor it shows whether the face matched if matched it shows as unlocked,if not matched locked

import cv2 #used for live video capture
import pyttsx3 #used for voic engine it speaks the strings we have given
import numpy as np
import os #used for giving folder data path  


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1000)

xcount = 0


def load_count():
    try:
        with open('count.txt', 'r') as file:
            count = int(file.read().strip())
    except FileNotFoundError:
        count = 0
    return count


def save_count(count):
    with open('count.txt', 'w') as file:
        file.write(str(count))


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x: x + w]
    return cropped_face


cap = cv2.VideoCapture(0)
count = load_count()
speak("Please look into the camera...")

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        xcount += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        
        data_path = r"F://ibmproject//face1" #the folder we have used to store our images
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        file_name_path = os.path.join(data_path, str(count) + ".jpg")
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Cropper", face)
    else:
        print('Face not found')
        speak("Face not found...")
        pass

    if cv2.waitKey(1) == 13 or xcount == 10:
        break

save_count(count)
cap.release()
cv2.destroyAllWindows()
print("Collecting samples complete")
