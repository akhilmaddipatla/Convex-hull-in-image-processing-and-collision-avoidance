import cv2 as cv
import numpy as np

image = cv.imread("C:/Users/makhi/OneDrive/Documents/Project - 1/images/City_skyline.jpg")

b_lank = np.zeros(image.shape[:2], dtype = "uint8")
cv.imshow('Blank', b_lank)

#converting the image to greyscale 

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)



#blurring = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT )

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)


edges_canny = cv.Canny(image, 125, 175)


#Now we will be using the find contours method
contours, hierarchies = cv.findContours(edges_canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(b_lank, contours, -1, (0,0,255), 1)
#we will be drawing contours 
cv.imshow("canny_edge", edges_canny)
#cv.imshow("Blurred", blurring)
cv.imshow("Gray", gray)
cv.imshow("City Skyline", image)
cv.waitKey(0)