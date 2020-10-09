import numpy as np

def rk4_step(f,y_n,t_n,h,*args):
    ''' Runge-kutt of fourth order integration step to get y_n+1

        INPUTS: f - first derivative function on dy/dt = f(y,t)
                y_n - initial condition
                t_n - initial condition
                h - time step
                args - arguments to pass to f  '''

    k1 = h * f(t_n, y_n, *args)
    k2 = h * f(t_n + h*0.5, y_n + k1*0.5, *args)
    k3 = h * f(t_n + h*0.5, y_n + k2*0.5, *args)
    k4 = h * f(t_n + h    , y_n + k3, *args)

    return y_n + (k1 + 2.*k2 + 2.*k3 + k4)/6.

def rk6_step(f,y_n,t_n,h,*args):
	''' Runge-kutta of sixth order integration step to get y_n+1

		INPUTS: f - first derivative function on dy/dt = f(y,t)
				y_n - initial condition
				t_n - initial condition
				h - time step
				args - arguments to pass to f  '''

	k1 = h * f(t_n,y_n,*args)
	k2 = h * f(t_n + (1./4.)*h  , y_n + (1./4.)*k1*h,*args)
	k3 = h * f(t_n + (3./8.)*h  , y_n + (3./32.)*h*(k1 + 3*k2),*args)
	k4 = h * f(t_n + (12./13.)*h, y_n + (12./2197.)*h*(161*k1 - 600*k2 + 608*k3),*args)
	k5 = h * f(t_n + h          , y_n + (1./4104.)*h*(8341*k1 - 32832*k2 + 29440*k3 - 845*k4),*args)
	k6 = h * f(t_n + (0.5)*h    , y_n + h*(-(8./27.)*k1 + 2*k2 - (3544./2565.)*k3 + (1859./4104.)*k4 - (11./40.)*k5),*args)

	return y_n + 1./.5 * ((16./27.)*k1 + (6656./2565.)*k3 + (28561./11286.)*k4 - (9./10.)*k5 + (2./11.)*k6)*h

def rk8_step(f,y_n,t_n,h,*args):
    ''' Runge-kutta of eight order integration step to get y_n+1

        INPUTS: f - first derivative function on dy/dt = f(y,t)
                y_n - initial condition
                t_n - initial condition
                h - time step
                args - arguments to pass to f  '''

	k_1 = func(t_n, y_n,*args)                                                               
	k_2 = func(t_n + h*(4./27.), y_n + (h*4./27.)*k_1,*args)                                                  
	k_3 = func(t_n + h*(2./9.) , y_n + (h/18.)*(k_1 + 3.*k_2),*args)                                          
	k_4 = func(t_n + h*(1./3.) , y_n + (h/12.)*(k_1 + 3.*k_3),*args)                                          
	k_5 = func(t_n + h*(1./2.) , y_n + (h/8.)*(k_1 + 3.*k_4),*args)                                          
	k_6 = func(t_n + h*(2./3.) , y_n + (h/54.)*(13.*k_1 - 27.*k_3 + 42.*k_4 + 8.*k_5,*args)                         
	k_7 = func(t_n + h*(1./6.) , y_n + (h/4320.)*(389.*k_1 - 54.*k_3 + 966.*k_4 - 824.*k_5 + 243.*k_6),*args)             
	k_8 = func(t_n + h         , y_n + (h/20.)*(-234.*k_1 + 81.*k_3 - 1164.*k_4 + 656.*k_5 - 122.*k_6 + 800.*k_7),*args)   
	k_9 = func(t_n + h*(5./6.) , y_n + (h/288.)*(-127.*k_1 + 18.*k_3 - 678.*k_4 + 456.*k_5 - 9.*k_6 + 576.*k_7 + 4.*k_8),*args)
	k_10= func(t_n + h         , y_n + (h/820.)*(1481.*k_1 - 81.*k_3 + 7104.*k_4-3376.*k_5 + 72.*k_6 - 5040.*k_7 - 60.*k_8 + 720.*k_9),*args)

	return y_n + h/840.*(41.*k_1 + 27.*k_4 + 272.*k_5 + 27.*k_6 + 216.*k_7 + 216.*k_9 + 41.*k_10);


def solve_rk(f_rhs,t_f,t_step,ci,rk_ord,*f_args):
    ''' Rk4 implementation to solve a system of linear first order differential equations  
        
        INPUTS: f_rhs - rist of right hand side of the differential equation system
                t_f - final time to integrate the system
                t_step - time step
                ci - initial conditions [y0.t0]
                f_args - argumets to pass to the rhs functions'''
    
    in_cond = [np.array(ci).reshape(-1,1)]

    while in_cond[-1][-1] <= t_f:

        t,y = in_cond[-1][-1], in_cond[-1][:-1]

		if rk_ord == 4:
        	y = rk4_step(f_rhs,y,t,t_step,[*f_args])
		elif rk_ord == 6:
        	y = rk6_step(f_rhs,y,t,t_step,[*f_args])
		elif rk_ord == 8:
        	y = rk8_step(f_rhs,y,t,t_step,[*f_args])

        ci_arr = np.concatenate((y.reshape(-1,1),np.array([t+t_step]).reshape(-1,1)))

        in_cond.append(ci_arr)

    return in_cond

