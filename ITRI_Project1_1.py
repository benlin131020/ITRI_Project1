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
    #gaussianblur
    blur = cv2.GaussianBlur(sharpen,(5,5),0)
    return blur

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

def roi1(img):
    #enhance
    enhance_img=enhance(img)
    enhance_img=enhance(enhance_img)
    #otsu threshold
    ret,thr = cv2.threshold(enhance_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #closing
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    #filling_hole
    filling_img=filling_hole(closing)
    #opening
    opening = cv2.morphologyEx(filling_img, cv2.MORPH_OPEN, kernel)
    #contuour
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    roi = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    #stacking images side-by-side
    opening=cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)
    roi=np.hstack((opening,roi))
    '''
    #show
    cv2.imshow("enhance",enhance_img)
    cv2.imshow("otsu",thr)
    cv2.imshow("closing",closing)
    cv2.imshow("filling_hole",filling_img)
    cv2.imshow("opening",opening)
    '''
    return roi

def roi2(img):
    #CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    #enhance
    enhance_img=enhance(cl1)
    #otsu threshold
    ret,thr = cv2.threshold(enhance_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #filling hole
    filling_img=filling_hole(thr)
    #opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(filling_img, cv2.MORPH_OPEN, kernel)
    #contuour
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    roi = cv2.drawContours(img, contours, -1, (0,0,255), 1)
    #stacking images side-by-side
    opening=cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)
    roi=np.hstack((opening,roi))
    '''
    #show
    cv2.imshow("clahe",cl1)
    cv2.imshow("enchance",enhance_img)
    cv2.imshow("otsu",thr)
    cv2.imshow("filling_hole",filling_img)
    cv2.imshow("opening",opening)
    '''
    return roi

#main
img1=cv2.imread("37736735_2061362360549587_9024993033665380352_n.jpg",0)
roi_img1=roi1(img1)
roi_img2=roi2(img1)
#stacking images side-by-side
img1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
res1=np.hstack((img1,roi_img1))
res2=np.hstack((img1,roi_img2))
cv2.imshow("result1",res1)
cv2.imshow("result2",res2)

cv2.waitKey(0)