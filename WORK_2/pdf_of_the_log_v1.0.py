#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:10:36 2022

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt

if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=False)
%config InlineBackend.figure_format='retina' # very useful command for high-res images

x=np.random.uniform(0.1,10,1000)

plt.subplot(111)
plt.hist(x, bins=10, label=r'$p_x(x)=Uniform(x)=10/9.9$')
plt.hline(y=np.mean(x),c='red')
plt.hline(y=np.median(x), c='blue')
plt.ylabel(r'$p_x(x)$')
plt.xlabel('x')

y=np.log(x)

plt.subplot(112)
plt.hist(y,bins=10, label=r'$y=log_{10}(x) \n p_y(y)=...$')
plt.plot(np.linspace(min(y),max(y),100), (10/9.9)*(1/x*np.ln(10)))
plt.xlabel('y')
plt.ylabel(r'$p_y(y)$')

print('mean of y %1.3f' % (np.mean(y)), '\n log of the mean of x %1.3f' % (np.log(np.mean(x))), '\n median of y %1.3f' % (np.median(y)), '\n log of the median of x %1.3f' % (np.log(np.median(x))))





