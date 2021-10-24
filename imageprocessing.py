# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:30:28 2021

@author: hp
"""
'''
import cv2
image=cv2.imread('F:/dm projects/image/person.jpg',0)
#print(image)
cv2.imshow('person',image)
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
'''

'''
 #addition operation(black-bg.white-pic)
import cv2
image1=cv2.imread('F:/dm projects/image/i1.png')
image2=cv2.imread('F:/dm projects/image/i2.png')
sum=cv2.add(image1,image2)
cv2.imshow('addimage',sum)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
cv2.imwrite('F:/dm projects/image/addimage.jpg',sum)
'''

'''
 #weightage addition operation(black-bg.white-pic)
import cv2
image1=cv2.imread('F:/dm projects/image/i3.png')
image2=cv2.imread('F:/dm projects/image/i4.png')
sum=cv2.addWeighted(image1,0.4,image2,0.2,0)#weightage for pixel in each image.correction fctor-intensity
cv2.imshow('weightageaddimage',sum)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
cv2.imwrite('F:/dm projects/image/weightedaddimage.jpg',sum)
'''

'''
#subtraction
import cv2
image1=cv2.imread('F:/dm projects/image/i5.png')
image2=cv2.imread('F:/dm projects/image/i6.png')
diff=cv2.subtract(image1,image2)
cv2.imshow('subimage',diff)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
cv2.imwrite('F:/dm projects/image/subimage.jpg',diff)
'''

'''
#resize
import cv2
image=cv2.imread('F:/dm projects/image/person.jpg')
#print(image)
resize_image=cv2.resize(image,(114,114))
cv2.imshow('person',resize_image)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
'''

'''
#errode the data
import cv2
import numpy as np
image=cv2.imread('F:/dm projects/image/i7.jpg')
#print(image)
kernel=np.ones((100,100),np.uint8())
#print(kernel)
cv2.imshow('kernel',kernel)
errosion=cv2.erode(image, kernel)
cv2.imshow('errosion',errosion)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
 '''   

'''
#blurring nd thresholding the data
import cv2
import numpy as np
image=cv2.imread('F:/dm projects/image/person.jpg')
#guassian=cv2.GaussianBlur(image,(77,77),0)
#median=cv2.medianBlur(image,5)
bilateral=cv2.bilateralFilter(image,9,75,244)
cv2.imshow('blur1',bilateral)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
'''

'''
#thresholding
import cv2
import numpy as np
image=cv2.imread('F:/dm projects/image/person.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#ret,thresh1=cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
#ret,thresh1=cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
#ret,thresh1=cv2.threshold(gray,120,255,cv2.THRESH_TRUNC)
#ret,thresh1=cv2.threshold(gray,120,255,cv2.THRESH_TOZERO)
ret,thresh1=cv2.threshold(gray,120,255,cv2.THRESH_TOZERO_INV)
cv2.imshow('thresh1',thresh1)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
'''

'''
#countoring 
import cv2
import numpy as np
image=cv2.imread('F:/dm projects/image/i8.png')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,30,200)
cv2.imshow('edge',edges)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
contours,hierarchy=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow('contour',image)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()

cv2.imwrite('F:/dm projects/image/contour.jpg',image)
print(hierarchy)
'''

'''
#drawing
import cv2
image=cv2.imread('F:/dm projects/image/person.jpg')
#image=cv2.line(image,(0,0),(1000,1000),(255,0,0),4)
#image=cv2.circle(image,(200,200),50,(255,255,0),4)
image=cv2.rectangle(image,(100,100),(270,200),(255,0,0),4)
cv2.imshow('imgage',image)
k=cv2.waitKey(0)
if k==ord('q'):#if q is pressed the window closses
    cv2.destroyAllWindows()
'''


