import numpy as np
import pandas as pd
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

def filling_hole_tl(img):
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w= img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    filled_img = img | im_floodfill_inv
    return filled_img

def filling_hole_br(img):
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w= img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (w-1,h-1), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    filled_img = img | im_floodfill_inv
    return filled_img

def features(roi,contours):
    index=[]
    centroid_x=[]
    centroid_y=[]
    area=[]
    perimeter=[]
    i=0
    for cnt in contours:
        #centroid
        M=cv2.moments(cnt)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        centroid_x=np.append(centroid_x,cx)
        centroid_y=np.append(centroid_y,cy)
        #area
        area=np.append(area,cv2.contourArea(cnt))
        #perimeter
        perimeter=np.append(perimeter,cv2.arcLength(cnt,True))
        #index
        index=np.append(index,i)
        cv2.putText(roi,str(i),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        i+=1

    df = pd.DataFrame({"Index":index,
    "Centroid_X":centroid_x,
    "Centroid_Y:":centroid_y,
    "Area":area,
    "Perimeter":perimeter
    })

    return df