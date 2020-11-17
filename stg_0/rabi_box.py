import numpy as np
import scipy as sp
from scipy.special import hermite as H_n
import rk_int as rk

# Atomic units
hbar,m_e,e = 1,1,1

# to calculate complex number square modulos

def sq_mod(z):
    return z.real**2 + z.imag**2

# Matrix elements of position operator

def coup(n,m):
	s1 = np.sin((n + m) * np.pi/2)/(n + m)**2
	s2 = np.sin((n - m) * np.pi/2)/(n - m)**2
	return s1 - s2
    
def x_nm(n,m,L):
	return 2 * L * coup(n,m) / np.pi**2


# Energy n-th level

def E_n(n,L):
	return np.pi**2 * n**2 / (2 * L**2)


# Transition frequency

def omega_nm(n,m,L):
	return E_n(n,L) - E_n(m,L)


# Electric field coupling 

def alpha(q,E0,x_nm):
	return -q * E0 * x_nm


# Rabi frequecy

def R_Omega(det,alpha):
	return np.sqrt(det**2 + alpha**2)


# Rabi RWA transition probabilities

def prob_1(t,det,alpha_nm):
	p1_1 = np.cos(R_Omega(det,alpha_nm)*t/2)**2
	p1_2 = np.sin(R_Omega(det,alpha_nm)*t/2)**2 * det**2 / R_Omega(det,alpha_nm)**2
    
	return p1_1 + p1_2

def prob_2(t,det,alpha_nm):
    
	return np.sin(R_Omega(det,alpha_nm)*t/2)**2 * alpha_nm**2 / R_Omega(det,alpha_nm)**2


# Rabi no-RWA complex differential system

def c1_dot(t,c2,omega_0,omega_E,alpha):
	return -1j * c2 * np.exp(-1j * omega_0 * t) * np.cos(omega_E * t) * alpha

def c2_dot(t,c1,omega_0,omega_E,alpha):
	return -1j * c1 * np.exp( 1j * omega_0 * t) * np.cos(omega_E * t) * alpha

def ode_syst_comp(t,y,params):
	c1,c2 = y
	omega_0,omega_E,alpha = params
    
	diffs = np.array([c1_dot(t,c2,omega_0,omega_E,alpha),c2_dot(t,c1,omega_0,omega_E,alpha)],dtype='complex_')
    
	return diffs


# Rabi no-RWA real differential system (split into real and imaginary parts)

def a1_dot(t,y,omega_0,omega_E,alpha):
	a1,b1,a2,b2 = y
    
	a11 = np.cos(omega_E*t)*np.cos(omega_0*t)*b2
	a12 = -1.*np.cos(omega_E*t)*np.sin(omega_0*t)*a2
    
	return (a11+a12)*alpha

def b1_dot(t,y,omega_0,omega_E,alpha):
	a1,b1,a2,b2 = y
    
	b11 = -1.*np.cos(omega_E*t)*np.cos(omega_0*t)*a2
	b12 = -1.*np.cos(omega_E*t)*np.sin(omega_0*t)*b2
    
	return (b11+b12)*alpha

def a2_dot(t,y,omega_0,omega_E,alpha):
	a1,b1,a2,b2 = y
    
	a21 = np.cos(omega_E*t)*np.cos(omega_0*t)*b1
	a22 = np.cos(omega_E*t)*np.sin(omega_0*t)*a1
    
	return (a21+a22)*alpha

def b2_dot(t,y,omega_0,omega_E,alpha):
	a1,b1,a2,b2 = y
    
	b21 = -1.*np.cos(omega_E*t)*np.cos(omega_0*t)*a1
	b22 = np.cos(omega_E*t)*np.sin(omega_0*t)*b1
    
	return (b21+b22)*alpha

def ode_syst_reim(t,y,params):
	omega_0,omega_E,alpha = params
    
	a1_d = a1_dot(t,y,omega_0,omega_E,alpha)
	b1_d = b1_dot(t,y,omega_0,omega_E,alpha)
	a2_d = a2_dot(t,y,omega_0,omega_E,alpha)
	b2_d = b2_dot(t,y,omega_0,omega_E,alpha)
    
	diffs = np.array([a1_d,b1_d,a2_d,b2_d])
    
	return diffs


# Integration of the complex diff system

def comp_noRWA_rk(n1,n2,det,q,E0,L,t_lim,dt,C0,rk_order):
    
	omega_n2n1 = omega_nm(n2,n1,L)
	x_n2n1 = x_nm(n1,n2,L)
	alpha_n2n1 = alpha(q,E0,x_n2n1)
	omega_E = omega_n2n1 + det
    
	sol = rk.solve_rk(ode_syst_comp,t_lim,dt,C0,rk_order,omega_n2n1,omega_E,alpha_n2n1)
    
	return sol


# Integration of the real diff system

def reim_noRWA_rk(n1,n2,det,q,E0,L,t_lim,dt,y0,rk_order):
    
	omega_n2n1 = omega_nm(n2,n1,L)
	x_n2n1 = x_nm(n1,n2,L)
	alpha_n2n1 = alpha(q,E0,x_n2n1)
	omega_E = omega_n2n1 + det
    
	sols = rk.solve_rk(ode_syst_reim,t_lim,dt,y0,rk_order,omega_n2n1,omega_E,alpha_n2n1)
    
	return sols


# ----------------- Quantum rabi --------------------------

# Quantum interaction term

def lamb(alpha,n):
	
	return alpha/(2*np.sqrt(n+1))

# Quantum rabi freq

def omega_q(det,lamb,n):
	
	return np.sqrt(det**2 + 4*lamb**2*(n+1))

# exited state coeff

def c_f_q(t,omega,det,lamb,n):
	
	c1 = -2j*lamb*np.sqrt(n+1)/omega
	c2 = np.exp(-1j*det*t/2)
	c3 = np.sin(omega*t/2)

	return c1*c2*c3

# ground state coeff

def c_i_q(t,omega,det,lamb,n):

	c1 = np.exp(-1j*det*t/2)
	c2 = np.sin(omega*t/2)*det/omega
	c3 = np.cos(omega*t/2) 

	return c1*(c2 + c3)

# excited state transition probability

def pe_q(t,omega,det,lamb,n):

	return c_f_q(t,omega,det,lamb,n).real**2 + c_f_q(t,omega,det,lamb,n).imag**2

# ground state transition probability

def pg_q(t,omega,det,lamb,n):

	return c_i_q(t,omega,det,lamb,n).real**2 + c_i_q(t,omega,det,lamb,n).imag**2


# general fotonic superposition probability transition
# with only the exited state initially poblated and
# with the coherent states for the electric field

def w_n(t,n,n_bar,lamb):
    w1 = np.cos(2*lamb*t*np.sqrt(n+1))
    w2 = n_bar**n / np.math.factorial(n)

    return w2*w1

def w(t,N,n_bar,lamb):
    w_sum = 0
    for n in range(N):
        w_sum += w_n(t,n,n_bar,lamb)

    return np.exp(-n_bar)*w_sum

def w_sigfig(t,sf,N,n_bar,lamb):
    reach_sigfig = False
    for n in range(6,N):
        w_n = w(t,n,n_bar,lamb) 
        w_nm1 = w(t,n-1,n_bar,lamb) 
        if abs(w_n - w_nm1).all() < 10**(-(sf+1)):
            reach_sigfig = True
            return w_n

    if not reach_sigfig:
        return w(t,N,n_bar,lamb)


# General fotonic superposition atomic inversion, with general
# initial conditions

def Sigma(n,lamb,det):
    return np.sqrt(det**2 + 4*lamb**2 * n)

def psi_G(t,lamb,det,c_eg,c_n,N=100,*args):
    Ce,Cg = c_eg

    psiG_sum = 0

    for n in range(1,N):
        f1 = (sq_mod(Cg)*sq_mod(c_n(n,*args)) * np.cos(Sigma(n,lamb,det)*t/2)**2 + sq_mod(Ce)*sq_mod(c_n(n-1,*args)) * np.sin(Sigma(n,lamb,det)*t/2)**2) * 4 * lamb**2 * n / Sigma(n,lamb,det)**2
        f2 = sq_mod(Cg)*sq_mod(c_n(n,*args)) * det**2 / Sigma(n,lamb,det)**2
        f3 = ((np.sin(Sigma(n,lamb,det)*t/2)*det/Sigma(n,lamb,det) - 1j * np.cos(Sigma(n,lamb,det)*t/2)) * Cg*c_n(n,*args) * Ce.conjugate()*c_n(n-1,*args).conjugate()).real * 4*lamb*np.sqrt(n)/Sigma(n,lamb,det)
       
        psiG_sum += f1 + f2 + f3*np.sin(Sigma(n,lamb,det)*t/2)

    return psiG_sum

def psi_E(t,lamb,det,c_eg,c_n,N=100,*args):
    Ce,Cg = c_eg

    psiE_sum = 0

    for n in range(int(N)):
        f1 = (sq_mod(Ce)*sq_mod(c_n(n,*args)) * np.cos(Sigma(n+1,lamb,det)*t/2)**2 + sq_mod(Cg)*sq_mod(c_n(n+1,*args)) * np.sin(Sigma(n+1,lamb,det)*t/2)**2) * 4 * lamb**2 * (n+1) / Sigma(n+1,lamb,det)**2
        f2 = sq_mod(Ce)*sq_mod(c_n(n,*args)) * det**2 / Sigma(n+1,lamb,det)**2
        f3 = ((np.sin(Sigma(n+1,lamb,det)*t/2)*det/Sigma(n+1,lamb,det) + 1j * np.cos(Sigma(n+1,lamb,det)*t/2)) * Cg*c_n(n+1,*args) * Ce.conjugate()*c_n(n,*args).conjugate()).real * 4*lamb*np.sqrt(n+1)/Sigma(n+1,lamb,det)
       
        psiE_sum += f1 + f2 + f3*np.sin(Sigma(n+1,lamb,det)*t/2)

    return psiE_sum


def W(t,lamb,det,c_eg,c_n,N=100,*args):
    return psi_E(t,lamb,det,c_eg,c_n,N,*args) - psi_G(t,lamb,det,c_eg,c_n,N,*args)


def psi_mod(t,lamb,det,c_eg,c_n,N=100,*args):
    return psi_E(t,lamb,det,c_eg,c_n,N,*args) + psi_G(t,lamb,det,c_eg,c_n,N,*args)


# Expansion coefitients for the genear fotonic superposition

def coherent_states(n,*args):
    alpha = args[0]

    exp = np.exp(- abs(alpha)**2 / 2)
    alph_n = alpha**n
    term = np.sqrt(float(np.math.factorial(n)))

    return exp*alph_n/term

def squeezed_states(n,*args):
    r,theta = args[0],args[1]

    if n%2 == 0:
        s1 = np.sqrt(float(sp.math.factorial(n)))/(2**(n//2) * sp.math.factorial(n//2)**2)
        s2 = np.exp(1j*(n//2)*theta)
        s3 = np.tanh(r)**(n//2)

        return (-1)**(n//2) * s1*s2*s3 / np.sqrt(np.cosh(r))

    else:
        return 0


def squeezed_coherent_states(n,*args):
    alpha,r,theta = args[0],args[1],args[2]
    gamma = alpha*np.cosh(r) + alpha.conjugate() * np.exp(1j*theta) * np.sinh(r)

    sc1 = (0.5*np.exp(1j*theta)*np.tanh(r))**(n/2.)/np.sqrt(sp.math.factorial(n))
    sc2 = Hermite_n(gamma*(np.exp(1j*theta)*np.sinh(2*r))**(-0.5),n)


    sct = np.exp(-0.5*(abs(alpha)**2 + alpha.conjugate()**2 * np.exp(1j * theta)*np.tanh(r)))/np.sqrt(np.cosh(r))

    return sct*sc1*sc2


### HERMITE POLYNOMIALS ####

def c_l_even(x,l,n):
   c1 = (-1)**(0.5*n - l)
   c2 = sp.math.factorial(2*l)
   c3 = sp.math.factorial(0.5*n - l)
   c4 = (2*x)**(2*l)

   return c1*c4/c2/c3

def c_l_odd(x,l,n):
   c1 = (-1)**(0.5*(n-1) - l)
   c2 = sp.math.factorial(2*l+1)
   c3 = sp.math.factorial(0.5*(n-1) - l)
   c4 = (2*x)**(2*l+1)

   return c1*c4/c2/c3


def Hermite_n(x,n):
    h_sum = 0

    if n%2 == 0:
        N = n//2
        for l in range(N):
            h_sum += c_l_even(x,l,n)

    else:
        N = (n-1)//2
        for l in range(N):
            h_sum += c_l_odd(x,l,n)

    return sp.math.factorial(n)*h_sum
