# from ultralytics import YOLO
# import cv2
# import cvzone

# cap  = cv2.VideoCapture(1)
# cap.set(3 , 640)
# cap.set(4 , 480)

# while True:
#     success , img = cap.read()
#     cv2.imshow("Image" , img)
#     cv2.waitKey(1)




import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO
import cvzone
import math

# Create a VideoCapture object for the webcam
cap = cv2.VideoCapture(0)

# Set the width and height of the frame
cap.set(3, 1280)
cap.set(4, 720)

# Create a VideoCapture object for the mp4 videos
#cap = cv2.VideoCapture("../Videos/bikes.mp4")

model = YOLO("yolov5s.pt")

# classNames = ['NonViolence', 'Sword', 'Violence', 'gun', 'knife', 'sword']
classNames = ['person','gun', 'knife']

# Create a figure and axis for displaying the live video
fig, ax = plt.subplots()
img_display = ax.imshow(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
plt.axis('off')



def update(frame):
    ret, frame = cap.read()

    results = model(frame , stream = True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding Box
            x1 , y1 ,x2 , y2 = box.xyxy[0]
            x1, y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
            # cv2.rectangle(frame , (x1 ,y1) , (x2 , y2) , (255 , 0 ,255) , 3)

            w , h = x2-x1 , y2-y1
            bbox = int(x1) , int(y1) , int(w) , int(h)

            #print(x1 , y1 , x2 , y2) 

            cvzone.cornerRect(frame , bbox)

            #Confidence
            conf = math.ceil((box.conf[0] * 100))/100
            print(conf)

            #Class Name
            cls = int(box.cls[0])
            print()
            cvzone.putTextRect(frame , f'{classNames[cls]} {conf}' , (max(0 , x1) , max(35 ,y1)) , scale = 1 , thickness = 2)

    img_display.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img_display,

# Use FuncAnimation to continuously update the display
ani = FuncAnimation(fig, update, blit=True)

plt.show()

# Release the webcam
cap.release()
