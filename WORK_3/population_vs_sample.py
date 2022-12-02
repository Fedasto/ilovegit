import numpy as np
import matplotlib.pyplot as plt

N=100000
mean_x=[]
s_sample=[]
n_sample=1000

for i in range(n_sample):
    x=np.random.normal(0,10,N)
    x_mean = np.mean(x)
    s=np.sqrt(1/(N-1))*np.sum((x-x_mean)**2)
    mean_x.append(x_mean)
    s_sample.append(s)

plt.figure(figsize=[8,8])    
plt.hist(mean_x, bins=30, density=True, fill=False)

plt.figure(figsize=[8,8])
plt.hist(s_sample, bins=30, density=True, fill=False)

err_x_mean=s_sample/np.sqrt(n_sample)
err_s_mean=s_sample/np.sqrt(2*(n_sample - 1))

