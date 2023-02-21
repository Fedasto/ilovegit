#%%

import numpy as np
import matplotlib.pyplot as plt

if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True)
%config InlineBackend.figure_format='retina' # very useful command for high-res images


N=10000

x=np.random.uniform(0.1,10,N)

plt.figure(figsize=[8,8])


plt.hist(x,bins=20, fill=False, density=True,histtype='step',lw=2, color='black' )
pdf_x=1/(10-0.1)
plt.hlines(pdf_x,0,10, colors='red')
plt.xlabel('x')
plt.ylim(0,0.12)
plt.xlim(0,10.2)

#plt.hlines(np.mean(x), 0,10, colors='red', label='mean')
#plt.hlines(np.median(x), 0,10, colors='blue', label='median')
#plt.legend(loc='best')

y=np.log10(x)

plt.figure(figsize=[8,8])
plt.hist(y,bins=20, fill=False, density=True, histtype='step',lw=2, color='black')


def pdf_y(y):
    return (10**y)*np.log(10)*(1/9.9)

y_grid=np.linspace(-1,1,100)
plt.plot(y_grid,pdf_y(y_grid), c='red')
plt.xlabel('y')
plt.xlim(-1.1,1.1)

mean_log_x=np.log10(np.mean(x))
mean_y=np.mean(y)

median_log_x=np.log10(np.median(x))
median_y=np.median(y)

print('mean of log of x = %1.5f \n mean of y = %1.5f \n median of log of x=%1.5f \n median of y= %1.5f' % (mean_log_x, mean_y, median_log_x, median_y))


