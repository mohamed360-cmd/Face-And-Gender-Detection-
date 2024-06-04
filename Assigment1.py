'''
Assigment 1 Detect Faces from static Images Predict their Gender and the Age of the Person/s in the Photo 
'''
import cv2
face_Classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # getting the face detection Model 
img = cv2.imread('./Images/img1.jpg',1) 
grayScaleImage = cv2.cvtColor(img,0) 
GENDER_MODEL = 'Weights/deploy_gender.prototxt'
GENDER_PROTO = 'Weights/gender_net.caffemodel'
AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
faces = face_Classifier.detectMultiScale(grayScaleImage,1.3,2)
face_NET = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
age_NET = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
for (x,y,w,h) in faces :
    #drawing rectange over the faces 
    cv2.rectangle(img,(x,y),(x+w,y+h),(100,25,70),5,cv2.LINE_AA)
    #....?
    face_blob = cv2.dnn.blobFromImage(img[y:y+h, x:x+w], 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    face_NET.setInput(face_blob)
    gender_preds = face_NET.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    cv2.putText(img, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2, cv2.LINE_AA)

     # ...?Age prediction
    age_blob = cv2.dnn.blobFromImage(img[y:y+h, x:x+w], 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_NET.setInput(age_blob)
    age_preds = age_NET.forward()
    age = AGE_INTERVALS[age_preds[0].argmax()]
    cv2.putText(img, age, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2, cv2.LINE_AA)
cv2.imshow("Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
