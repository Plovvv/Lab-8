import cv2 as cv
import numpy as np
import sys


#1
# img = cv.imread("images\\variant-4.jpeg")
 
# if img is None:
#     sys.exit("Could not read the image.")

# blue = cv.split(img)[0]
# zeros = np.zeros(blue.shape, np.uint8)

# blue_colored = cv.merge([blue, zeros, zeros])
# cv.imshow('Blue Color Only', blue_colored)
# k = cv.waitKey(0)


#2, 3
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

img_fly = cv.imread("fly64.png")
if img_fly is None:
    sys.exit("Could not read the image.")

while True:
    h_f, w_f, _ = img_fly.shape

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.rectangle(
        frame,
        (frame.shape[1]//2, 0),
        (frame.shape[1], frame.shape[0]),
        (255,0,0),
        1
        )

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9,9), 2)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=35,
        minRadius=50,
        maxRadius=300)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for x, y, r in circles[0,:]:
            if y-r < 0 or y+r > frame.shape[0] or x-r < 0 or x+r > frame.shape[1]:
                continue

            roi = gray[y-r:y+r, x-r:x+r]
            if roi.size == 0:
                continue
            roi = cv.resize(roi, (200,200))

            _, roi_bin = cv.threshold(roi, 127, 255, cv.THRESH_BINARY)

            white_ratio = np.sum(roi_bin == 255) / (200*200)

            if 0.35 < white_ratio < 0.65: 
                for i in circles[0,:]:
                    cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                    # cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                    frame[i[1]-h_f//2:i[1]+h_f//2, i[0]-w_f//2:i[0]+w_f//2] = img_fly

                    break
            
            if x >= frame.shape[1]/2:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame,'Right',(50,frame.shape[0]-10), font, 1,(255,255,255),2,cv.LINE_AA)
    
    cv.imshow('detected circles',frame)
    if cv.waitKey(1) & 0xFF == 27:
        break


cv.destroyAllWindows()