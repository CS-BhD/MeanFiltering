import cv2
import numpy as np
from pylab import *
import scipy
import scipy.stats
import matplotlib.pyplot as plt

people = cv2.imread("1.jpg")
people = cv2.resize(cv2.cvtColor(people,cv2.COLOR_BGR2RGB),(200,200))


def histeq(im,nbr_bins = 256):
    """对一幅灰度图像进行直方图均衡化"""
    #计算图像的直方图
    #在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    imhist,bins = histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   #
    cdf = 255.0 * cdf / cdf[-1]
    #使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

#算术均值滤波
def ArithmeticMeanOperator(roi):
    return np.mean(roi)
def ArithmeticMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = ArithmeticMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)

def rgbArithmeticMean(image):
    r,g,b = cv2.split(image)
    r = ArithmeticMeanAlogrithm(r)
    g = ArithmeticMeanAlogrithm(g)
    b = ArithmeticMeanAlogrithm(b)
    return cv2.merge([r,g,b])

#几何均值滤波
def GeometricMeanOperator(roi):
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return p**(1/(roi.shape[0]*roi.shape[1]))
    
def GeometricMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = GeometricMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)

def rgbGemotriccMean(image):
    r,g,b = cv2.split(image)
    r = GeometricMeanAlogrithm(r)
    g = GeometricMeanAlogrithm(g)
    b = GeometricMeanAlogrithm(b)
    return cv2.merge([r,g,b])

#谐波均值滤波
def HMeanOperator(roi):
    roi = roi.astype(np.float64)
    if 0 in roi:
        roi = 0
    else:
        roi = scipy.stats.hmean(roi.reshape(-1))
    return roi
def HMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] =HMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)
def rgbHMean(image):
    r,g,b = cv2.split(image)
    r = HMeanAlogrithm(r)
    g = HMeanAlogrithm(g)
    b = HMeanAlogrithm(b)
    return cv2.merge([r,g,b])

#逆谐波均值滤波
def IHMeanOperator(roi,q):
    roi = roi.astype(np.float64)
    return np.mean((roi)**(q+1))/np.mean((roi)**(q))
def IHMeanAlogrithm(image,q):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = IHMeanOperator(image[i-1:i+2,j-1:j+2],q)
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)
def rgbIHMean(image,q):
    r,g,b = cv2.split(image)
    r = IHMeanAlogrithm(r,q)
    g = IHMeanAlogrithm(g,q)
    b = IHMeanAlogrithm(b,q)
    return cv2.merge([r,g,b])

plt.subplot(321)
plt.title("peple")
plt.imshow(people)
plt.axis("off")

plt.subplot(322)
plt.title("Arithmetic to Image")
plt.axis("off")
plt.imshow(rgbArithmeticMean(people))

plt.subplot(323)
plt.title("Geomotric to Image")
plt.axis("off")
plt.imshow(rgbGemotriccMean(people))


plt.subplot(324)
plt.title("H Mean to Tmage")
plt.imshow(rgbHMean(people))
plt.axis("off")

plt.subplot(325)
plt.title("IH Mean to Tmage Q=2")
plt.imshow(rgbIHMean(people,2))
plt.axis("off")

plt.subplot(326)
plt.title("IH Mean to Tmage Q=3")
plt.imshow(rgbIHMean(people,3))
plt.axis("off")

plt.show()
