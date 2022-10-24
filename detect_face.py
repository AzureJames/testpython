#  py -m pip     py -m pip install opencv-contrib-python
#ONLY ACCEPTS JPG
import time
import cv2
from PIL import Image

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);   
    numberOfFaces = 0
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        numberOfFaces += 1
    print(numberOfFaces)
    if (numberOfFaces > 0):
        return img_copy
    else:
        return 0



#------------HAAR-----------
#note time before detection
test1 = cv2.imread('dsts.jpg')
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
t1 = time.time()
#call our function to detect faces
haar_detected_img = detect_faces(haar_face_cascade, test1)
#note time after detection
t2 = time.time()
#calculate time difference
dt1 = t2 - t1
#print the time difference

print(dt1)
print(type(haar_detected_img))
# img = Image.open(haar_detected_img)
# img.show()

if(type(haar_detected_img) == int):
    print("No faces detected")
else:
    print("face detected")
    cv2.imshow('Profile Image', haar_detected_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()