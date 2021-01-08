import cv2
import numpy as np
import imutils
from imutils import perspective

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def imgscal(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

cap = cv2.VideoCapture('http://192.168.1.6:8080/video')
# cap = cv2.VideoCapture(0)
while True :
    _,frame = cap.read()
    frame = imgscal(frame, 50)
    # ------------- edges --------
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    cany  = cv2.Canny(gray,50,100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    Cany = cv2.dilate(cany, kernel, iterations=1)
    Cany = cv2.erode(cany, kernel, iterations=1)

    # ---------- draw contours ------
    cnts ,_ = cv2.findContours(cany.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]


    for c in cnts :
        if cv2.contourArea(c) > 1000:
            pri = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,pri , True)
            x,y,w,h = cv2.boundingRect(approx)
            # rect = cv2.minAreaRect(c)
            # box = cv2.boxPoints(rect)
            # box = np.array(box)
            # box = perspective.order_points(box.astype(int))
            # (tl, tr, br, bl) = box
            # tm = midpoint(tl, tr)
            # tb = midpoint(bl, br)
            cv2.drawContours(frame, c, -1, (0, 255,0))
            shape = ""
            if len(approx) > 3 :
                shape = "line"
            if len(approx) == 3 :
                shape = "triangle"
            if len(approx) == 4 :
                shape = "square"
            if len(approx) == 5:
                shape = "pentagon"
            if len(approx) == 6:
                shape = "hexagon"
            if len(approx) > 10:
                shape = "circle"
            # "{:.1f}".format(len(approx))
            # (int(tm[0] - 15), int(tb[1] - 10))
            cv2.putText(frame,shape,(x,y), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
            print(len(approx))

    cv2.imshow("frame", frame)
    # cv2.imshow("frame", cany)
    key = cv2.waitKey(1)
    if key == 27 :
        break
cap.release()
cv2.destroyAllWindows()













