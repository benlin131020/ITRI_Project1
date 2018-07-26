import numpy as np
import cv2
import image_processing as ip

def roi(img):
    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    #enhance
    enhance_img=ip.enhance(cl1)
    #otsu threshold
    ret,thr = cv2.threshold(enhance_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #closing
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    #filling_hole
    filling_img=ip.filling_hole_br(closing)
    #opening
    opening = cv2.morphologyEx(filling_img, cv2.MORPH_OPEN, kernel)
    #contuour
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    roi = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    opening=cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)
    roi=np.hstack((opening,roi))
    #show
    cv2.imshow("clahe",cl1)
    cv2.imshow("enhance",enhance_img)
    cv2.imshow("otsu",thr)
    cv2.imshow("closing",closing)
    cv2.imshow("filling_hole",filling_img)
    cv2.imshow("opening",opening)

    return roi

#main
img2=cv2.imread("37770404_2061362363882920_3525037766163300352_n.jpg",0)
roi2=roi(img2)
img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
res2=np.hstack((img2,roi2))
cv2.imshow("result2",res2)

cv2.waitKey(0)