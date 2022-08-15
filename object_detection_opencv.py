import numpy as np
import cv2

Image_hand = cv2.imread("C:/Users/makhi/OneDrive/Documents/Project - 1/images/hand.jpg", 0)

ret, threshold = cv2.threshold(Image_hand, 10, 255, cv2.THRESH_BINARY)
cont, heir = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

c_hull = [cv2.convexHull(i) for i in cont]
final = cv2.drawContours(Image_hand, c_hull, -1, (255, 255, 255))


cv2.imshow('Original', Image_hand)
cv2.imshow('Thresh', threshold)
cv2.imshow('Conex Hull', Image_hand)
cv2.waitKey(0)

#here we will be thresholding using the openCV Binary threshold operation
#for any pixel value > 10, we are going to make the output image as white and the remaining will be black 
#we are going to find the contour information and information about the trees
#for every 'i' in our contour list we will be generating out convexhull. 
