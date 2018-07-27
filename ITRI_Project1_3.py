import numpy as np
import cv2
import image_processing as ip

def roi(img):
    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    #otsu threshold
    ret,thr = cv2.threshold(cl1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #filling_hole
    filling_img=ip.filling_hole_br(thr)
    #opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(filling_img, cv2.MORPH_OPEN, kernel)
    #contuour
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    roi = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    opening=cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)
    roi=np.hstack((opening,roi))
    #show
    cv2.imshow("clahe",cl1)
    cv2.imshow("otsu",thr)
    cv2.imshow("filling_hole",filling_img)
    cv2.imshow("opening",opening)
    '''
    #rect
    for cnt in contours:
        if cv2.contourArea(cnt)>128:
            x,y,w,h = cv2.boundingRect(cnt)
            roir = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("rect",roir)
    '''
    return roi

#main
img3=cv2.imread("37673579_2061362367216253_6417462665358082048_n.jpg",0)
roi3=roi(img3)
img3=cv2.cvtColor(img3,cv2.COLOR_GRAY2BGR)
res3=np.hstack((img3,roi3))
cv2.imshow("result3",res3)

cv2.waitKey(0)