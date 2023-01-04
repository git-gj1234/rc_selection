import numpy as np

import cv2

def centers(cont):                       
    m = cv2.moments(cont)
    Cx = 0
    Cy = 0
    if m['m00'] != 0:
        Cx = int(m['m10']/m['m00'])
        Cy = int(m['m01']/m['m00'])
    return (Cx,Cy) 

vid = cv2.VideoCapture(0)

while(True):
    ret,frame = vid.read()

    blur = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
    kernel1 = np.ones((8,8),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)

    low = np.array([100,130,45])
    high = np.array([135,170,90])

    mask = cv2.inRange(blur,low,high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    canny = cv2.Canny(mask,200,230)
    contours,heir = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
    for i in contours:
        cnt = contours[0]
        leftmost = rightmost = topmost = bottommost = 0
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])       
        topleft = (leftmost[0]-4,topmost[1]-4)
        bottomright = (rightmost[0]+4,bottommost[1]+4)
        
        if(leftmost!=0):
            cv2.rectangle(frame,topleft,bottomright,(0,255,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Bottle Cap',(topleft[0],topleft[1]-4), font, 0.6,(2,2,2),1,cv2.LINE_AA)
    
    for i in contours:
        k=0
        if(k <1):
            c1,c2 = centers(i)
            cv2.circle(frame,(c1,c2), 1, (0,0,255), -1)
            
        k=1

    cv2.imshow('frame',frame)
    cv2.imshow('frame2',mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
