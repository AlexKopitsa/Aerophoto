import os.path

import inline
import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#print(model)

def image_capture():
    img = str(input("Enter image URL:"))
    #img = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzzApXNGeoJVWR3GhO-3bjzMXVq5YwReYhiGQvE9pL9A&s'
    #img = 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Droga_ekspresowa_S5S10_Stryszek-Bia%C5%82e_B%C5%82ota_a.jpg'
    results = model(img)
    results.print()
    matplotlib.use('TkAgg')
    plt.imshow(np.squeeze(results.render()))
    plt.show()

def video_capture():
    video_file = str(input("Enter video file name:"))
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        results = model(frame)
        cv2.imshow("video_capture", np.squeeze(results.render()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/last.pt', force_reload=True)
img = os.path.join('yolov5/train/images/00000045_jpg.rf.aba8b608499fc826627f1f2c3b36116a.jpg')
img1 = os.path.join('yolov5/train/images/00000117_jpg.rf.fb8add4be4c96a810babf365d22a3f2b.jpg')
img2 = os.path.join('yolov5/train/images/00000249_jpg.rf.53d5d15f3219eadfb7d21f68761a3fb7.jpg')
img3 = os.path.join('yolov5/train/images/00000265_jpg.rf.22fe8be5fde5da08bdbc97905c8da845.jpg')
img4 = os.path.join('yolov5/train/images/00000396_jpg.rf.ad8eeb4e991b2f5a1bdd17ecfbb26104.jpg')
img5 = os.path.join('yolov5/train/images/00000538_jpg.rf.8e70e2f294047e2de19a9fb1122fdd0a.jpg')
img6 = os.path.join('yolov5/train/images/00000634_jpg.rf.a7ff2951e2c65b40814d9b7a8439ac8b.jpg')
img7 = os.path.join('yolov5/train/images/00000769_jpg.rf.d6ce992f1802fbefa0b0af3a6c3a8197.jpg')

results = model(img)
results1 = model(img1)
results2 = model(img2)
results3 = model(img3)
results4 = model(img4)
results5 = model(img5)
results6 = model(img6)
results7 = model(img7)
results.print()
results1.print()
results2.print()
results3.print()
results4.print()
results5.print()
results6.print()
results7.print()
matplotlib.use('TkAgg')
plt.imshow(np.squeeze(results.render()))
plt.show()






