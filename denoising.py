#! coding=utf-8
import numpy as np
import cv2

# Add Gaussian noise
def addnoi(img, std=0.1):
    # generate gaussian noise with std
    r, c = img.shape
    noise = np.random.normal(0, std, (r, c))*255
    # add noise
    img_noi = img+noise
    # normalize
    minVal = np.amin(img_noi)
    maxVal = np.amax(img_noi)
    img_noi = (img_noi - minVal)/(maxVal - minVal)
    
    return img_noi


# TV denoising
def tvdenoi(img, tol, maxit):
    ### Variational based Image denoising method 
    cv2.imshow("Input Image", img)
    cv2.moveWindow("Input Image", 0, 0)
    cv2.waitKey(0)
    print tol, maxit


# Test 
if __name__=='__main__':
    # loading
    inputim = cv2.imread('./images/lena.png') # '/home/cluo/Pictures/西藏经幡.jpeg'
    im = cv2.cvtColor(inputim, cv2.COLOR_BGR2GRAY)
    print np.amin(im), np.amax(im)
    cv2.imshow("Input", im)
    cv2.waitKey(0)
    # adding noise
    imnoi = addnoi(im, std=0.2)
    print np.amin(imnoi), np.amax(imnoi)
    # denoising
    tvdenoi(imnoi, 0.01, 100)
    
