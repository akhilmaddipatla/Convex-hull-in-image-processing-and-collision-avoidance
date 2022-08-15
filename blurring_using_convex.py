from random import gauss
from statistics import median
import cv2 as cv
from numpy import average

image = cv.imread("C:/Users/makhi/OneDrive/Documents/Project - 1/images/tiger.jpg")
cv.imshow("My image", image)

#Averaging 
aver = cv.blur(image, (7, 7))
cv.imshow('Average Blur', aver)

#median blur 
median_blur = cv.medianBlur(image, 7)
cv.imshow('Median Blur', median_blur)

#Gaussian blur 
gauss = cv.GaussianBlur(image, (7,7), 0)
cv.imshow('Gaussian Blur', gauss)

cv.waitKey(0) 
