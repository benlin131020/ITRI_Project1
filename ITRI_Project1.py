import numpy as np
import cv2

def roi(img):
    #sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, kernel)
    #gaussianblur
    blur = cv2.GaussianBlur(sharpen,(5,5),0)
    #otsu threshold
    ret,thr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("otsu",thr)
    #closing
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing",closing)
    #contuour
    image, contours, hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    roi = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    return roi

#main
img1=cv2.imread("37736735_2061362360549587_9024993033665380352_n.jpg",0)
img2=cv2.imread("37770404_2061362363882920_3525037766163300352_n.jpg",0)
img3=cv2.imread("37673579_2061362367216253_6417462665358082048_n.jpg",0)

roi1=roi(img1)
roi2=roi(img2)
roi3=roi(img3)
#stacking images side-by-side
img1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
img3=cv2.cvtColor(img3,cv2.COLOR_GRAY2BGR)
res1=np.hstack((img1,roi1))
res2=np.hstack((img2,roi2))
res3=np.hstack((img3,roi3))
cv2.imshow("result1",res1)
cv2.imshow("result2",res2)
cv2.imshow("result3",res3)




cv2.waitKey(0)