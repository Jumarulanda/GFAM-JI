import numpy as np
import rk_int as rk

# Atomic units
hbar,m_e,e = 1,1,1

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