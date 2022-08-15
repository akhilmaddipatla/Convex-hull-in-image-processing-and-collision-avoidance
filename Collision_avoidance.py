#Our aim here is to apporximate the countours. If we approximate the countours with various thresholds or accuracy factors, 
#they will help in smoothening the contour shape better. The more we reduce the accuracy and threshold factors, the more the contours will become smooth

import numpy as np
import cv2

#Load the image data and create a copy of the original image
data_image  = cv2.imread("C:/Users/makhi/OneDrive/Documents/Project - 1/images/car-image1.jpg")
copy_image_data = data_image.copy()
cv2.imshow('Original Image', copy_image_data)
cv2.waitKey(0)
 
#Now we convert the image to GREYSCALE Format and later apply our required threshold     
image_grayscale = cv2.cvtColor(data_image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image_grayscale, 127, 255, cv2.THRESH_BINARY_INV)

#Now we find the contours 
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#we create rectangles by iterating through each of the contours
for cont in contours:
    x,y,w,h = cv2.boundingRect(cont)
    cv2.rectangle(copy_image_data, (x,y), (x+w, y+h), (0,0,255),1)
    cv2.imshow('Rectangling the object', copy_image_data)
    
cv2.waitKey(0)

#Applying contour approximation
for i in contours:
    accuracy = 0.04 * cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, accuracy, True)
    cv2.drawContours(copy_image_data, [approx], 0, (127, 0, 127), 2)
    cv2.imshow("Final image after applying contours",copy_image_data)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

#Therefore here we can see that the lower the accuracy, the better will be the approximation of the car.




#Convex HUll 
# we draw the outer edges of the shapes or car and draw the lines to them based on the edges 
# if the convex hull of the car avoids collision of the obstacles. so does the car

orig_image = cv2.imread("C:/Users/makhi/OneDrive/Documents/Project - 1/images/car-image1.jpg")
orig_image_gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', orig_image)
cv2.waitKey(0)

#Applying threshold on image 
ret, tresh = cv2.threshold(orig_image_gray, 176, 255, 0)

#lets find the contours
contours, heirarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Sorting the contours by area and later remove the contour which is largest 
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse= False)[:n]

#we finally draw the convex hull by iterating 
for cont in contours:
    hull = cv2.convexHull(cont)
    cv2.drawContours(orig_image, [hull], 0, (255,0,0), 2)
    cv2.imshow('Image after Convex Hull',orig_image )
    
cv2.waitKey(0)
cv2.destroyAllWindows() 
    