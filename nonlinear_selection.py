import math
import numpy as np
from pylab import *

def simulate_trajectory(N,dt,alpha,beta):
    ns = [1]
    while ns[-1] > 0 and ns[-1] < N:
        ns.append(update_n(ns[-1],N,dt,alpha,beta))
    return np.array(ns)

def update_n(n,N,dt,alpha,beta):
    def f(y):
        return alpha*y**2
    y = 1.0*n/N
    delta_n_plus = np.random.binomial(n,f(y)/y*dt)
    delta_n_minus = np.random.binomial(n,beta*dt)
    return n + delta_n_plus - delta_n_minus

def simulate_trajectory_adaptive(N,alpha,beta):
    ns = [1]
    ts = [0]
    while ns[-1] > 0 and ns[-1] < N:
        n,dt = update_n_adaptive(ns[-1],N,alpha,beta)
        ns.append(n)
        ts.append(ts[-1]+dt)
    return np.array(ns),np.array(ts)

def update_n_adaptive(n,N,alpha,beta):
    p_desired = 0.1
    def f(y):
        return alpha*y**2
    y = 1.0*n/N
    birth_rate = 1.0 + f(y)/y
    death_rate= 1.0 + beta
    max_rate = np.max([birth_rate,death_rate])
    dt = p_desired/max_rate
    delta_n_plus = np.random.binomial(n,(birth_rate)*dt)
    delta_n_minus = np.random.binomial(n,(death_rate)*dt)
    return n + delta_n_plus - delta_n_minus,dt
    
def get_epidemic_size(ns,ts):
    return np.sum(ns[:-1]*np.diff(ts))

def run_epidemics(N,alpha,beta,num_trials = 10000,plotting=False,trajectory_fn = simulate_trajectory_adaptive):
    y_n, y_minus, y_plus, y_p,critical_determinant = get_parameters(N,alpha,beta)
    if plotting:
        #close('all')
        figure()
        hold(1)
    T_ave = 0
    fixed = np.zeros(num_trials)
    epidemic_size = np.zeros(num_trials)
    for i in range(num_trials):
        ns,ts = trajectory_fn(N,alpha,beta)
        fixed[i] = ns[-1] >= N
        epidemic_size[i] = get_epidemic_size(ns,ts)
        if plotting:
            semilogy(ts,1.0*ns/N,'-')
        T_ave += ts[-1]
    T_ave /= 1.0*num_trials
    p_fix = sum(fixed)/num_trials
    print 'T_ave = {}, P_fix = {}'.format(T_ave,p_fix)
    if plotting:
        axhline(y_n,color='k',label=r'$y_n$')
        axhline(y_minus,color='b',label=r'$y_-$')
        axhline(y_plus,color='b',label=r'$y_+$')
        axhline(y_p,color='r',label=r'$y_p$')
        legend()
        grid()
    return epidemic_size,fixed

def get_parameters(N,alpha,beta):
    critical_determinant = 4*alpha/(N*beta**2)
    y_n = beta/alpha
    if critical_determinant < 1:
        y_minus = beta/(2*alpha)*(1 -  np.sqrt(1 - critical_determinant))
        y_plus = beta/(2*alpha)*(1 +  np.sqrt(1 - critical_determinant))
    else:
        y_minus = -1
        y_plus = -1
    y_p = beta/(2*alpha)*(1 +  np.sqrt(1 + critical_determinant))
    print r'y_n = {}, y_- = {}, y_+ = {}, y_p = {}, critical determinant = {}'.format(y_n,y_minus,y_plus,y_p,critical_determinant)
    print r'n_n = {}'.format(y_n*N)
    return y_n, y_minus, y_plus, y_p,critical_determinant


def P_w_th(w,s):
    return np.exp(-s**2*w/4)*w**(-1.5)/(2*np.sqrt(np.pi)*(1 + s))

from scipy import integrate

def s(y,alpha,beta):
    def f(y):
        return alpha*y**2
    return f(y)/y - beta

def P_fix(y0,alpha,beta,N):
    def f(y):
        return alpha*y**2

    def a(y):
        return f(y) - beta*y

    def b(y):
        return 1.0/N*(f(y) + (2 + beta)*y)

    def psi(y,a,b):
        return np.exp(-2*integrate.quad(lambda x: a(x)/b(x),0,y)[0])
    return integrate.quad(lambda x: psi(x,a,b),0,y0)[0]/integrate.quad(lambda x: psi(x,a,b),0,1)[0]