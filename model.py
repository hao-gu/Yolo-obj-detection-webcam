import cv2
import math
from ultralytics import YOLO
#model
model = YOLO("yolo-Weights/yolov8n.pt")
classNames = ["person"]
#webcam init
camera = cv2.VideoCapture(0)
camera.set(3, 100)
camera.set(4, 100)

while True:
    ret, frame= camera.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        cnt = 0
        for box in boxes:
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)
            if confidence < 0.8:
                continue
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # class name
            cls = int(box.cls[0])
           #print("Class name -->", classNames[cls])
            if classNames[cls] == 'person':
                cnt+=1

            print(cls)
            print(classNames[cls])
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls] + str(confidence), org, font, fontScale, color, thickness)
        cv2.putText(frame,"people in room: "+str(cnt) , [5,20], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()