import torch
import cv2
import yaml
from PIL import ExifTags, Image, ImageOps
#load model


model = torch.hub.load('ultralytics/yolov5', 'custom', path='sign.pt')
model.conf = 0.25
model.iou = 0.45
model.multi_label = False
model.max_det = 1000

    # detection and show part
 # Read webcame
video = cv2.VideoCapture(0) 
while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            print('Error cant find Webcam')
            break
         #Because yolov5 give score in RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        cv2.imshow('Object detector', cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR))#Change back to BGR
         # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'): 
            break
video.release()
cv2.destroyAllWindows()

