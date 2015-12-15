#! /usr/bin/env python
# _*_ coding:utf-8 _*_  #or #! coding=utf-8
"""\
This is denoising module.
    
including: 1) total variation based denoising.
           2) non-local means denoising.

@author: C. Luo
2015-12-13 15:19:32
"""

# import some necessary modules
import numpy as np
import cv2


# Add Gaussian noise
def addnoi(img, std=0.1):
    """\
    Add white gaussian noise with std.
    
    Parameters
    ----------
    -img: image to be added noise
    -std: standard deviation of guassian distribution

    Returns
    -------
    img_noi: degraded image with gaussian noise
    """
    # generate gaussian noise with std
    r, c = img.shape
    noise = np.random.normal(0, std, (r, c))*255

    # add noise
    img_noi = img+noise

    # normalize
    minVal = np.amin(img_noi)
    maxVal = np.amax(img_noi)
    img_noi = (img_noi - minVal)/(maxVal - minVal)

    # return noisy
    return img_noi




# TV denoising
def tvdenoi(g, lambd, beta, tol, maxit=50):
    """\
    Variational based Image denoising method.
    
    Parameters
    ----------
    -g: image to be denoised
    -lambd: regularization parameter
    -beta: penalty parameter
    -tol: tolerance for convergence
    -maxit: the max number of iteration

    Returns
    -------
    -u: denoised image

    Notes
    -----
    the algorithm Model:1/2*||g-f||2,2 + lambd||▽f||1 (equal to ||f||TV)
                      --1/2*||g-f||2,2 + lambd||d||1 + beta/2||d-▽f-b||2,2   
    """
    
    # define a function for compute L2 norm
    norm = lambda x, p:pow(sum(sum(pow(x, p))), 1.0/p)

    # initialize
    r, c = g.shape
    u = np.copy(g)
    dx = np.zeros((r, c))
    dy = np.zeros((r, c))
    bx = np.zeros((r, c))
    by = np.zeros((r, c))
    
    loop = 0
    stopFlag = False
    eps = 0.0000001
    # main loop
    while not stopFlag:
        loop += 1
        u_pre = np.copy(u)
        dbx = dx - bx
        dby = dy - by
        
        # iterate all pixels
        for i in np.arange(1, r-1):
            for j in np.arange(1, c-1):
                
                # compute image
                u[i][j] = (g[i][j] + beta*(dbx[i][j] - dbx[i][j+1])  + beta*(dby[i][j] - dby[i+1][j]) \
                                   + beta*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1])) / (1.0+4*beta)

                # compute auxiliary variables
                tempx = (u[i][j] - u[i][j-1]) + bx[i][j]
                tempy = (u[i][j] - u[i-1][j]) + by[i][j]
                tempm = np.sqrt(tempx*tempx + tempy*tempy) + eps
                dx[i][j] = max(tempm - lambd*1.0/beta, 0)*tempx/tempm # '*1.0' because two ints make division by '/' may be rounded
                dy[i][j] = max(tempm - lambd*1.0/beta, 0)*tempy/tempm

                # update bregman variables
                bx[i][j] = tempx - dx[i][j]
                by[i][j] = tempy - dy[i][j]

        # compute error
        err = norm(u - u_pre, 2)/norm(u, 2)
        print "loop:{0}, err:{1}".format(loop, err)#or print 'loop:', loop,', err:', err #or print "loop:%-8d err:%-8.4f" % (loop, err)
        
        # check if converge
        if loop == maxit:
            stopFlag = True
            print "Max iteration is reached."
        elif err<tol:
            stopFlag = True
            print "Converge at ", loop
        
    # return denoised result
    return u




# Test 
if __name__ == "__main__":
    # loading
    inputim = cv2.imread('./images/lena.png')      # '/home/cluo/Pictures/西藏经幡.jpeg'
    im = cv2.cvtColor(inputim, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (256, 256))
    cv2.imshow("Input", im)
    cv2.moveWindow("Input", 0, 0)
    #cv2.waitKey(0)

    # adding noise
    imnoi = addnoi(im, std=0.1)
    cv2.imshow("Addnoi", imnoi)
    cv2.moveWindow("Addnoi", 256, 0)
    #cv2.waitKey(0)

    # denoising
    # the smaller lambd and the bigger beta, the worse denoising effect.
    # beta almost equal to 100*lambd
    imdenoi = tvdenoi(imnoi, 0.2, 20, 0.001, 10)
    cv2.imshow("Denoi", imdenoi)
    cv2.moveWindow("Denoi", 256*2, 0)
    cv2.waitKey(0)
    
