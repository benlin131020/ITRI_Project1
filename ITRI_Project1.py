import numpy as np
import cv2

def sobel(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x) 
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst

def enhance(img):
    #sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, kernel)
    cv2.imshow("sharpen",sharpen)
    #gaussianblur
    blur = cv2.GaussianBlur(sharpen,(5,5),0)
    cv2.imshow("blur",blur)
    #sharpen
    sharpen2 = cv2.filter2D(blur, -1, kernel)
    cv2.imshow("sharpen2",sharpen2)
    #gaussianblur
    blur2 = cv2.GaussianBlur(sharpen2,(5,5),0)
    cv2.imshow("blur2",blur2)
    return blur2

def filling_hole(im_in):
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    return im_out

def roi(img):
    #enhance
    enhance_img=enhance(img)
    #otsu threshold
    ret,thr = cv2.threshold(enhance_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("otsu",thr)
    #closing
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing",closing)
    #filling_hole
    filling_img=filling_hole(closing)
    cv2.imshow("filling_hole",filling_img)
    #opening
    opening = cv2.morphologyEx(filling_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening",opening)
    #contuour
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    roi = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    return roi

#main
img1=cv2.imread("37736735_2061362360549587_9024993033665380352_n.jpg",0)
roi1=roi(img1)
#stacking images side-by-side
img1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
res1=np.hstack((img1,roi1))
cv2.imshow("result1",res1)

img2=cv2.imread("37770404_2061362363882920_3525037766163300352_n.jpg",0)
img3=cv2.imread("37673579_2061362367216253_6417462665358082048_n.jpg",0)
roi2=roi(img2)
roi3=roi(img3)
img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
img3=cv2.cvtColor(img3,cv2.COLOR_GRAY2BGR)
res2=np.hstack((img2,roi2))
res3=np.hstack((img3,roi3))
cv2.imshow("result2",res2)
cv2.imshow("result3",res3)

cv2.waitKey(0)