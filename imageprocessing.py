# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:30:28 2021

@author: hp
"""
'''
import cv2
image=cv2.imread('F:/dm projects/image/person.jpg',0)
#print(image)
#cv2.imshow('person',image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
'''

'''
import cv2
import numpy as np
image=cv2.imread('F:/dm projects/image/person.jpg',0)
#print(image)
#cv2.imshow('person',image)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
    
cv2.imwrite('F:/dm projects/image/personcopy.jpg',image)
#if not cv2.imwrite(r'F:/dm projects/image/personcopy.jpg',image):
#    raise Exception('couldnt copy')
'''
import cv2
image=cv2.imread('F:/dm projects/image/rgb.png')
#print(image)
#cv2.imshow('p1',image)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()

B,G,R=cv2.split(image)
#print(B)
#cv2.imshow('blue',B)#the color becomes whitish grey if it has mosre intensity of blue
cv2.imshow('green',G)
#cv2.imshow('red',R)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
