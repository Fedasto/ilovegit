import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.integrate import quad

N=1000
 
integrals=0
err_integrals=0

integrals_1=0
err_integrals_1=0

integrals_2=0


integranda = lambda x, s: x**3 *np.exp(-((x**2)/(2*s**2)))

f_p = lambda x,s: (x**3 *np.exp(-((x**2)/(2*s**2)))) * uniform.pdf(x)    

data=np.random.normal(0.1,10,N)
sigma=np.random.random_integers(0,10,1)

#plt.figure(figsize=[8,8])
#plt.hist(data, density=True, fill=False, color='black')
#plt.plot(data, uniform.pdf(data), color='red')



integrals, err_integrals = (quad(f_p ,0,np.infty, args=(sigma,)))

inte_1=[]
err_inte_1=[]

for i in range(1000):
    i_1, err_i_1=(quad(integranda,0,np.infty, args=(sigma,))) #integrates from 0 to infinity
    inte_1.append(i_1)
    err_inte_1.append(err_i_1)

integrals_1, err_integrals_1 = np.mean(inte_1), np.std(inte_1)/np.sqrt(1000)
    
integrals_2 = 1/N * np.sum(integranda(data,sigma))

print('quad method f(x)p(x): ', integrals, '\n quad method only integrand: ' , integrals_1,  '\n Montecarlo method: ' , integrals_2)

print(r'$2 sigma^4$= ', (2*(int(sigma)**4)))



