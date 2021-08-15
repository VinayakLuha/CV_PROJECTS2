import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import  cmath


import math
def rang(x):
    pass



video = cv2.VideoCapture("oct.mp4")



def angle(pt1,pt2,pt3):
    x1,y1 =pt1
    x2,y2 = pt2
    x3, y3 = pt3
    base =math.sqrt((x2-x1)**2 +(y2-y1)**2)
    hypotenuse = math.sqrt((x3 - x1) **2 + (y3- y1) ** 2)
    per =math.sqrt((x3 - x2) **2 + (y3- y2) ** 2)
    ang =math.atan(per/base)
    ang =math.degrees(ang)
    if(x3<x2):
        ang=-ang
    return ang




def drawBox(frame):
    p1 = (260, 380)
    p2 = (360, 380)
    p3 = (490, 460)
    p4 = (125, 460)

    frame = cv2.line(frame, p1, p2, (0, 0, 255), 1)
    frame = cv2.line(frame, p2, p3, (0, 0, 255), 1)
    frame = cv2.line(frame, p3, p4, (0, 0, 255), 1)
    frame = cv2.line(frame, p4, p1, (0, 0, 255), 1)
    frame = cv2.line(frame, (308,375),(308,385), (0, 0, 255), 2)

    return frame



def wrapping(frame,pts1,pts2):
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (480, 640))
    return result



def histcalculations(frame,pic):
    count =0
    LeftL=[]
    RList=[]

    histogram = np.sum(frame[int(frame.shape[0] / 2):, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # print(leftx_base, rightx_base)


    L = leftx_base
    R = rightx_base

    nonzero = frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []
    minpix = 250
    windows =10

    winh = int(640 / windows)
    margin = 40

    for win in range(windows):
        winL = frame.shape[0] - ((win + 1) * winh)
        winH = winL + winh
        x1 = L - margin
        x2 = L + margin
        x3 = R - margin
        x4 = R + margin

        # pic = cv2.rectangle(pic, (x1, winL), (x2, winH), (0, 0, 0), 2)
        if (x3 - x2 <50 ):
            x3=480-(margin)
            x4=480
            # print(R)
            # pic = cv2.rectangle(pic, (x3, winL), (x4, winH), (0, 0, 0), 2)
            # pic= cv2.line(pic, (L, winH), (L, winH+winh), (0,0,250), 2)
            # pic = cv2.line(pic, (R, winH), (R, winH+winh), (0,0,250), 2)
        # else:
        #     pic = cv2.rectangle(pic, (x3, winL), (x4, winH), (0, 0, 0), 2)

        if(count%2==0):
            # print("X")
            LeftL.append((x1+margin, int(winL+winh/2)))
        count = count + 1
        # print(count)
        if(count%2!=0):
            RList.append((x3+margin,int(winL+winh/2)))
        count = count + 1

        good_left_inds = ((nonzeroy >= winL) & (nonzeroy < winH) & (nonzerox >= x1) & (
                nonzerox < x2)).nonzero()[0]
        good_right_inds = ((nonzeroy >= winL) & (nonzeroy < winH) & (nonzerox >= x3) & (
                nonzerox < x4)).nonzero()[0]
        left_lane_inds.append(good_left_inds)

        print("TO")
        right_lane_inds.append(good_right_inds)
        print((good_left_inds))

        if len(good_left_inds) > minpix:
            L = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            R = np.int(np.mean(nonzerox[good_right_inds]))
        # print("L LIST {0} {1}".format(LeftL,count))
        # print("R LIST {0} {1}".format(RList,count))



    # LeftL.append((L,winL))
    # RList.append((R,winL))


    RList1=RList[::-1]
    FL=LeftL+RList1
    arr =np.array(FL)
    # print(arr)
    pic=cv2.fillPoly(pic,[arr],color=(98, 152, 247))


    # overlay = np.zeros_like(pic)
    # overlay[:] = (98, 152, 247)#polyfill translucent
    # pic = cv2.addWeighted(overlay, 0.3, pic, 0.7, 0)

    # pic = cv2.rectangle(pic, (x3, winL), (x4, winH), (0, 0, 0), 2)
    wi =int(winh/2)

    for i in range(windows):
        # pic = cv2.line(pic, (LeftL[i][0],LeftL[i][1]),(LeftL[i+1][0],LeftL[i+1][1]), (250, 0, 0), 2)
        if(i!=windows-1):
             pic = cv2.line(pic, RList[i], RList[i + 1], (250, 0, 0), 3)
             pic = cv2.line(pic, LeftL[i], LeftL[i + 1], (250, 0, 0), 3)
        pic = cv2.circle(pic, RList[i], 10, (0, 240, 0), -1)
        pic = cv2.circle(pic, LeftL[i], 10, (0, 240, 0), -1)
        pic = cv2.rectangle(pic, (LeftL[i][0] + margin, LeftL[i][1] + wi),(LeftL[i][0] - margin, LeftL[i][1] -wi), (0, 0, 0), 2)
        pic = cv2.rectangle(pic, (RList[i][0]+margin, RList[i][1]+wi), (RList[i][0]-margin, RList[i][1]-wi), (0, 0, 0), 2)




    a,b=LeftL[windows-1]
    c,d=RList[windows-1]
    e=int((a+c)/2)
    pt3=(e,d)
    pic = cv2.circle(pic,pt3,15,(0,0,0),-1)
    pt1 =(int(480/2),640)
    pt2 =(int(480/2),d)
    pic =cv2.line(pic,pt1,pt2,(0,0,255),4)
    pic = cv2.line(pic, pt1, pt3, (0, 0, 0), 4)
    theta =angle(pt1,pt2,pt3)
    return pic,theta





def slidingWindow(frame):
    frames =frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 0, 172])
    upper_blue = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((2, 2), np.uint8)
    result = cv2.erode(mask, kernel, iterations=1)


    pts1 = np.float32([[260, 380], [360, 380], [490, 460], [125, 460]])
    pts2 = np.float32([[0, 0], [480, 0], [480, 640], [0, 640]])

    frame = wrapping(result, pts1, pts2)
    frames = wrapping(frames, pts1, pts2)

    overlay=np.zeros_like(frames)
    overlay[:]=(98,152,247)

    frames=cv2.addWeighted(overlay, 0.3, frames, 0.7,0)
    frame,theta = histcalculations(frame,frames)
    return frame,theta




def masking(frame):
    roi = np.array([[(260, 380), (360, 380), (490, 460), (125, 460)]])
    # print("ROI+")
    # print(roi)
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame2)
    cv2.fillPoly(mask, roi, 255)

    return mask







def Extractor(frame,mask):
    frame =cv2.bitwise_and(frame,frame,mask=mask)
    return frame





while True:

    cv2.waitKey(1)
    ret, frame = video.read()
    if not ret:
       break

    frame = cv2.resize(frame, (640, 480))
    frame2,theta = slidingWindow(frame)

    # roi = np.array([[(260, 380), (360, 380), (490, 460), (125, 460)]])
    # cv2.fillPoly(frame, roi, (255, 255, 255))

    frame3=masking(frame)
    # cv2.imshow("tyyyy", frame3)

    pts1 = np.float32([[260, 380], [360, 380], [490, 460], [125, 460]])
    pts2 = np.float32([[0, 0], [480, 0], [480, 640], [0, 640]])

    roi = np.array([[(260, 380), (360, 380), (490, 460), (125, 460)]])
    bg_ext=cv2.fillPoly(frame, roi,(0,0,0))

    matrix = cv2.getPerspectiveTransform(pts2, pts1)

    fg_ext = cv2.warpPerspective(frame2, matrix, (640,480))
    cv2.imshow("fg_ext", fg_ext)
    cv2.imshow("bg_ext",bg_ext)
    # print(np.shape(fg_ext))
    # print(np.shape(bg_ext))


    dst=cv2.add(fg_ext,bg_ext)
    dst=drawBox(dst)



    cv2.imshow("perspective view",frame2)
    cv2.putText(dst, "STEERING ANGLE={0}".format(theta), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

    if(theta>0):
        cv2.putText(dst, "TURN RIGHT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    else:
        cv2.putText(dst, "STEERING ANGLE={0}".format(theta), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
        cv2.putText(dst, "TURN Left", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.imshow("dst", dst)
    # cv2.waitKey(0)



    key = cv2.waitKey(1)
    if key == 27:
        break






