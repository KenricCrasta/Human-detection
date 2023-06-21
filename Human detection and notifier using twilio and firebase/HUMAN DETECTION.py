import cv2
import time
from twilio.rest import Client
import pyrebase
import serial
#ser = serial.Serial('/dev/ttyUSB0', 9600)
Config = {
      
      }

firebase=pyrebase.initialize_app(Config)
storage=firebase.storage()


#thres = 0.45 # Threshold to detect object
client= Client('')

from_whatsapp_number=''
to_whatsapp_number=''
flag=0
inittime=0.0
classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img_, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img_,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img_,box,color=(0,255,0),thickness=2)
                    cv2.putText(img_,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img_,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img_,objectInfo

def sendout():
    time.sleep(2.5)
    cv2.imwrite(filename='SAVED_IMAGES/Capture.jpg', img=img_)
    my_image="SAVED_IMAGES/Capture.jpg"
    storage.child("Capture.jpg").put(my_image)
    auth=firebase.auth()
    email=""
    passw=""
    user= auth.sign_in_with_email_and_password(email,passw)
    #geturl
    time.sleep(5)
    url=storage.child('Capture.jpg').get_url(user['idToken'])
    message=client.messages.create(body='You have a visiter',media_url=url,from_=from_whatsapp_number,to=to_whatsapp_number)
    
    

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        success, img_ = cap.read()
        result, objectInfo = getObjects(img_,0.45,0.2,objects=['person'])
        data ="1"       
        cv2.imshow("Output",img_)
        cv2.waitKey(1)
        if objectInfo:
            if data=="1":
                
                if flag==0:
                    '''imgdata='SAVED_IMAGES/'
                    timestamp=time.ctime()
                    timestamp=timestamp.replace(' ','_')
                    imgdata=imgdata+timestamp
                    imgdata=imgdata+('.jpg')
                    print(imgdata)'''
                    sendout()
                if inittime==0.0:
                
                    inittime=time.time() # for person reset after 10 min
                flag=1
                ct=time.time()
                if (ct-inittime)>=30:
                   flag=0
                   inittime=0.0
                   print(ct-inittime)
               
               
            
               
            
            
