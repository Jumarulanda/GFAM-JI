import numpy as np
import scipy as sp

def fac(n):
    if n>1:
        return n*fac(n-1)
    else:
        return 1

# General fotonic superposition atomic inversion, with general
# initial conditions

def Sigma(n,lamb,det):
    return np.sqrt(det**2 + 4*lamb**2 * n)

############################## Ground state transition probability ################################################

def gcoef_0(t,n,lamb,det,ce,cg,cn,*args):
    s1 = abs(cg)**2 * abs(cn(n,*args))**2 * np.cos(Sigma(n,lamb,det)*t*0.5)**2
    s2 = abs(ce)**2 * abs(cn(n-1,*args))**2 * np.sin(Sigma(n,lamb,det)*t*0.5)**2

    return 4*lamb**2 *n * (s1 + s2) / Sigma(n,lamb,det)**2

def gcoef_1(t,n,lamb,det,ce,cg,cn,*args):
    s1 = abs(cg)**2 * abs(cn(n,*args))**2

    return det**2 * s1 / Sigma(n,lamb,det)**2

def gcoef_2(t,n,lamb,det,ce,cg,cn,*args):
    s1 = det * np.sin(Sigma(n,lamb,det)*t*0.5) / Sigma(n,lamb,det)
    s2 = -1j*np.cos(Sigma(n,lamb,det)*t*0.5)

    s3 = cg*cn(n,*args) * (ce*cn(n-1,*args)).conjugate()

    s4 = ((s1+s2) * s3).real * np.sin(Sigma(n,lamb,det)*t*0.5)

    return 4*lamb*np.sqrt(n) * s4 / Sigma(n,lamb,det)

def psi_G(t,lamb,det,c_eg,cn,N=100,*args):
    ce,cg = c_eg

    psiG_sum = 0

    for n in range(1,N):
        psiG_sum += gcoef_0(t,n,lamb,det,ce,cg,cn,*args) + gcoef_1(t,n,lamb,det,ce,cg,cn,*args) + gcoef_2(t,n,lamb,det,ce,cg,cn,*args)

    return psiG_sum


############################## Exited state transition probability ################################################

def ecoef_0(t,n,lamb,det,ce,cg,cn,*args):
    s1 = abs(ce)**2 * abs(cn(n,*args))**2 * np.cos(Sigma(n+1,lamb,det)*t*0.5)**2
    s2 = abs(cg)**2 * abs(cn(n+1,*args))**2 * np.sin(Sigma(n+1,lamb,det)*t*0.5)**2

    return 4*lamb**2 *(n+1) * (s1 + s2) / Sigma(n+1,lamb,det)**2

def ecoef_1(t,n,lamb,det,ce,cg,cn,*args):
    s1 = abs(ce)**2 * abs(cn(n+1,*args))**2

    return det**2 * s1 / Sigma(n+1,lamb,det)**2

def ecoef_2(t,n,lamb,det,ce,cg,cn,*args):
    s1 = det * np.sin(Sigma(n+1,lamb,det)*t*0.5) / Sigma(n+1,lamb,det)
    s2 = 1j*np.cos(Sigma(n+1,lamb,det)*t*0.5)

    s3 = cg*cn(n+1,*args) * (ce*cn(n,*args)).conjugate()

    s4 = ((s1+s2) * s3).real * np.sin(Sigma(n+1,lamb,det)*t*0.5)

    return 4*lamb*np.sqrt(n+1) * s4 / Sigma(n+1,lamb,det)

def psi_E(t,lamb,det,c_eg,cn,N=100,*args):
    ce,cg = c_eg

    psiE_sum = 0

    for n in range(1,N):
        psiE_sum += ecoef_0(t,n,lamb,det,ce,cg,cn,*args) + ecoef_1(t,n,lamb,det,ce,cg,cn,*args) + ecoef_2(t,n,lamb,det,ce,cg,cn,*args)

    return psiE_sum

#####################################################################################################################


def W(t,lamb,det,c_eg,c_n,N=100,*args):
    return psi_E(t,lamb,det,c_eg,c_n,N,*args) - psi_G(t,lamb,det,c_eg,c_n,N,*args)


def psi_mod(t,lamb,det,c_eg,c_n,N=100,*args):
    return psi_E(t,lamb,det,c_eg,c_n,N,*args) + psi_G(t,lamb,det,c_eg,c_n,N,*args)


# Expansion coefitients for the genear fotonic superposition

def coherent_states(n,*args):
    alpha = args[0]

    exp = np.exp(-1*abs(alpha)**2 * 0.5)
    alph_n = alpha**n
    term = np.sqrt(sp.special.gamma(n))
    
    return exp*alph_n/term


# ---------- pure squeezed states -----------

def div_fac(n,nh,d):
    if n>1:
        return div_fac(n-2,nh-1,d)* (n/d/nh) * ((n-1)/d/nh)
    else:
        return 1

def ps_coef0(n):
    s1 = (-1)**n
    # s2 = np.sqrt(sp.special.gamma(1+2*n))
    # s3 = 2**n * sp.special.gamma(1+n)
    n_term = np.sqrt(div_fac(2*n,n,2))

    return s1*n_term #s2/s3

def ps_coef1(n,r,theta):
    s1 = np.exp(1j*n*theta)
    s2 = np.tanh(r)

    return s1 * s2**n

def squeezed_states(n,*args):
    r,theta = args[0],args[1]

    if n%2 == 0:
        n = n//2
        ps_n = ps_coef0(n) * ps_coef1(n,r,theta) / np.sqrt(np.cosh(r))
    else:
        ps_n = 0

    return ps_n


# ----------- squeezed coherent states ------

def N_sq(r,theta,alpha):

    exp_arg1 = abs(alpha)**2
    exp_arg2 = alpha.conjugate()**2 * np.exp(1j*theta) * np.tanh(r)
    t_arg = -1*(exp_arg1 + exp_arg2) * 0.5

    return np.exp(t_arg)

def gamm(r,theta,alpha):
    mu = np.cosh(r)
    nu = np.sinh(r) * np.exp(1j*theta)

    gam = mu*alpha + nu*alpha.conjugate()

    return gam


def squeezed_coherent_states(n,*args):
    r,theta,alpha,N = args[0],args[1],args[2],args[3]

    t1 = np.exp(1j*theta)*np.tanh(r)*0.5
    t2 = np.sqrt(np.cosh(r))

    t3 = t1**(n/2) / t2

    t_n = H_n(gamm(r,theta,alpha)/np.sqrt(np.exp(1j*theta)*np.sinh(2*r)),n) / np.sqrt(sp.special.gamma(1+n))

    return N*t3*t_n


### HERMITE POLYNOMIALS ####

def h_m(x,m,n):
    h1 = (-1)**m
    h2 = sp.special.gamma(m+1) * sp.special.gamma((n - 2*m)+1) * (2*x)**(2*m)

    return h1/h2

def H_n(x,n):
    H_sum = 0
    N = n//2

    for m in range(0,N):
        H_sum += h_m(x,m,n)

    retH = sp.special.gamma(n+1) * (2*x)**n * H_sum

    return retH


