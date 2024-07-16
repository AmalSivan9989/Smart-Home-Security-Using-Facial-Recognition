import cv2 #used for live video capture
import pyttsx3 #used for voic engine it speaks the strings we have given
import numpy as np
from os import listdir #used for giving folder data path
from os.path import isfile, join


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1000)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()


DATA_PATH = r"F:\ibmproject\face1"
CONFIDENCE_THRESHOLD = 83  # we have given a threshold value 83 is best threshold value


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #it is used to recognize the face

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x: x + w]
    return cropped_face


onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
Training_data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = join(DATA_PATH, file)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is not None:
        images = cv2.resize(images, (200, 200))
        Training_data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    else:
        print(f"Warning: Could not read image file {image_path}")

Labels = np.asarray(Labels, dtype=np.int32)

if len(Training_data) == 0:
    print("No training data found. Exiting...")
    exit(1)


model = cv2.face.LBPHFaceRecognizer_create() # the algorithm used LBPH
model.train(np.asarray(Training_data), np.asarray(Labels))
print("Training complete")

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x: x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


cap = cv2.VideoCapture(0)

x = 0
c = 0
d = 0
m = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    image, face = face_detector(frame)

    if face is not None:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        if result[1] < 500:
            confidence = int((1 - (result[1]) / 300) * 100)
            display_string = str(confidence)
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Face Recognition', image)
            x += 1
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)
            c += 1
    else:
        cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Face Recognition', image)
        
        d += 1

    if cv2.waitKey(1) == 13 or x == 10 or c == 30 or d == 20:
        break

cap.release()
cv2.destroyAllWindows()
 # output
if x >= 5:
    m = 1
    speak("Welcome home Boss") 
    speak("Good to have you back") 
elif c == 30:
    print("Face not recognized. Please try again.")
elif d == 20:
    print("Face not found. Please try again.")

if m == 1:
    speak("Door is closing.")
