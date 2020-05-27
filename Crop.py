#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pytesseract
import cv2
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import numpy as np
import string as s
import re
import PIL as pil
from matplotlib import pyplot as plt


# In[2]:


def crop(string):
    im=Image.open(string)
    img=cv2.imread(string)
    im.convert('L')
    im=ImageEnhance.Contrast(im)
    im=im.enhance(5)
    im=im.filter(ImageFilter.EDGE_ENHANCE)
    width=im.size[0]
    height=im.size[1]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel=np.ones((5,5),np.uint8)
    edges=cv2.Canny(gray,0,200,apertureSize=3)
    edges=cv2.dilate(edges,kernel)
    _,contours,heirarchy=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    area=[cv2.contourArea(c) for c in contours]
    Ar_max=np.argmax(area)
    cMax=contours[Ar_max]
    x,y,w,h=cv2.boundingRect(cMax)
    im=im.crop((x,y,x+w,y+h))
    im.save("cropped.jpeg")
    cropped=cv2.imread("cropped.jpeg")
    os.remove("cropped.jpeg")
    return cropped


# In[3]:


def Contours(full_path):
    im=Image.open(full_path)
    cropped=crop(full_path)
    gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    rec,thresh=cv2.threshold(gray,0,100,cv2.THRESH_BINARY_INV)
    dilation=cv2.dilate(thresh,kernel,iterations=1)
    dilation=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)
    _,contours,_=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours,gray,dilation


# In[56]:


def OCR(file_name):
    temp=cv2.imread(file_name)
    pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    contours,cropped,dilation=Contours(file_name)
    ret,cropped=cv2.threshold(cropped,0,255,cv2.THRESH_OTSU)
    cropped = cv2.GaussianBlur(cropped,(5,5),cv2.BORDER_DEFAULT)
    f=open("text.txt","w+")
    pattern="\d+"
    text=pytesseract.image_to_string(cropped,config='-c tessedit_char_whitelist=0123456789')
    print(text)
    f.write(text)
    f.close()
    cv2.drawContours(cropped,contours,-1,(10,255,5),2)
    return dilation,cropped,contours


# In[71]:


def main(file_name):
    os.getcwd()
    path="E:/python/images"
    direc=os.chdir(path)
    full_path=path+'/'+file_name
    im=Image.open(full_path)
    if im.size[0]<im.size[1]:
        im=im.resize((699,1106))
    else:
        im=im.resize((1106,699))
    im.save("temp.png") 
    temp=cv2.imread("temp.png")
    cv2.imwrite("temp.jpeg",temp)
    dilation,op,contours=OCR("temp.jpeg")
    os.remove("temp.jpeg")
    f=open("text.txt")
    lines=f.readlines()
    f.close
    if len(lines)>1:
        for line in lines:
            if len(line)!=12:
                lines.remove(line)
    f=open("text.txt","w+")
    text=lines[0]
    f.write(lines[0])
    f.close
    return op,contours,text


# In[74]:


op,contours,text=main("aad9.jpeg")


# In[75]:


#cv2.imwrite("temp.jpeg",op)
#Image.open("temp.jpeg")
text


# In[68]:


f=open("text.txt")
lines=f.readlines()
print(len(lines))
print(lines)


# In[ ]:


im=Image.open("E:/python/images/aad7.jpeg")
temp=cv2.imread("E:/python/images/aad7.jpeg")
temp.resize(699,1106)
temp.shape
im.size
im.resize((699,1106))

