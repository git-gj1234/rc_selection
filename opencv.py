##################################
"""
open cv project to detect the cetroid of a bottle cap and diplay a bounding box over it
the centroid is marked with a red dot
a green box is used around the cap
text is displayed at the top 
"""
##################################

#please refer to the image in the git repo

import numpy as np
import cv2

def centers(cont):  #function to find the centroid of the bottle cap                     
    m = cv2.moments(cont)
    Cx = 0
    Cy = 0
    if m['m00'] != 0:
        Cx = int(m['m10']/m['m00'])
        Cy = int(m['m01']/m['m00'])
    return (Cx,Cy) 

vid = cv2.VideoCapture(0) #capture video from the primary camera

while(True):

    ret,frame = vid.read() #store the current frame in variable frame

    blur = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT) #removes noise from the image 
    kernel1 = np.ones((8,8),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)

    #create low and high values for the colour of the bottle cap(obtained from an rgb colour picker using a saved image of the cap)
    low = np.array([100,130,45])
    high = np.array([135,170,90])

    #create a mask to filter the bottle cap part
    mask = cv2.inRange(blur,low,high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    canny = cv2.Canny(mask,200,230)
    #finding the edges of the detected contour
    contours,heir = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
    for i in contours:

        #finding coordinates to display the bounding box
        cnt = contours[0]
        leftmost = rightmost = topmost = bottommost = 0
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])       
        topleft = (leftmost[0]-4,topmost[1]-4)
        bottomright = (rightmost[0]+4,bottommost[1]+4)
        
        if(leftmost!=0):
            #draw a rectangle box over the image
            cv2.rectangle(frame,topleft,bottomright,(0,255,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #display text
            cv2.putText(frame,'Bottle Cap',(topleft[0],topleft[1]-4), font, 0.6,(2,2,2),1,cv2.LINE_AA)
    
    for i in contours:
        k=0
        if(k <1):
            c1,c2 = centers(i)
            #draw the center as a red dot
            cv2.circle(frame,(c1,c2), 1, (0,0,255), -1)
            
        k=1
        #quit the loop after the first countour centorid is found(avoids multiple center detection)

    cv2.imshow('frame',frame)
    cv2.imshow('frame2',mask)


    #quit when user presses q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
