'''
Assigment 1 Detect Faces from static Images Predict their Gender and the Age of the Person/s in the Photo 
'''
import cv2
face_Classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # getting the face detection Model 
img = cv2.imread('./Images/img1.jpg',1) 
grayScaleImage = cv2.cvtColor(img,0) #load same  image in Gray Scale
faces = face_Classifier.detectMultiScale(grayScaleImage,1.2,2)
for (x,y,w,h) in faces :
    cv2.rectangle(img,(x,y),(x+w,y+h),(100,25,70),5,cv2.LINE_AA)

cv2.imshow("Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()