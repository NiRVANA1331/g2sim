#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


#Need to change the way noise adds up. Generally, SNR is defined for the total signal and not the indivisual components


def tempfn(x,mu,sigma):
        return (x-mu)**2/(2*sigma**2)

def gaussian3D(X,mu_X,sigma_X,noise):
    #Noisy gaussian constructed by adding a noisy signal to 3D gaussian
    #technically, it can handle nD
    f = 0
    for i in range(len(X)):
        f = f + tempfn(X[i],mu_X[i],sigma_X[i])
    f = np.exp(-f)
    noisy_f =  f #+ np.random.normal(0, noise, f.shape) 
    return noisy_f

def gaussian3D_n(X,mu_Xlist,sigma_Xlist,noise):
    #mu_Xlist = [[mux,muy,muz],[mux,muy,muz]]
    #similar for sigma
    gauss = 0
    for i in range(len(mu_Xlist)):
        gauss = gauss + gaussian3D(X,mu_Xlist[i],sigma_Xlist[i],noise=0)
    #gauss = gauss+np.random.normal(0, noise)
    gauss = gauss + np.random.normal(0,noise)
    return gauss

def psf_intensity(X,muX_list,sigmaX_list,bkg,noise, normalize = 1):
    f = gaussian3D_n(X,muX_list,sigmaX_list, noise=0)
    final = f+bkg
    if normalize:
        temp = len(muX_list)+bkg
        return final/temp + np.random.normal(0, noise)
    return final+ np.random.normal(0, noise)

#g2 noise code incomplete 
def psf_g2(X,muX_list,sigmaX_list,bkg,noise, normalize = 1):
    #actually for 1-g2 value
    num = 0
    for i in range(len(muX_list)):
        num = num + gaussian3D(X,muX_list[i],sigmaX_list[i],noise=0)**2
    den = 0
    for i in range(len(muX_list)):
        den = den + gaussian3D(X,muX_list[i],sigmaX_list[i],noise=0)
    den = den + bkg
    den = den**2
    final = num/den
    if normalize:
        temp = len(muX_list)**2/(len(muX_list)+bkg)**2
        return (final)/temp + np.random.normal(0, noise)
    return final + np.random.normal(0, noise)

def psf_G2(X,muX_list,sigmaX_list,bkg,noise, normalize = 1):
    #1-g2 x intenisty^2 = psf^2
    #only basic noise function
    num = 0
    for i in range(len(muX_list)):
        num = num + gaussian3D(X,muX_list[i],sigmaX_list[i],noise=0)**2
    final = num
    if normalize:
        temp = len(muX_list)**2
        return (final)/temp + np.random.normal(0, noise)
    return final + np.random.normal(0, noise)


def combined_psf(X,muX_list,sigmaX_list,bkg,noise,lam = 10):
    p = psf_intensity(X,muX_list,sigmaX_list,bkg,noise)
    g = psf_g2(X,muX_list,sigmaX_list,bkg,noise)
    return lam*p+g

