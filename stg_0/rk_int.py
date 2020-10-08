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
    k4 = h * f(t_n + h, y_n + k3, *args)

    return y_n + (k1 + 2.*k2 + 2.*k3 + k4)/6.

def rk6_step(f,y_n,t_n,h,*args):
    ''' Runge-kutta of sixth order integration step to get y_n+1

        INPUTS: f - first derivative function on dy/dt = f(y,t)
                y_n - initial condition
                t_n - initial condition
                h - time step
                args - arguments to pass to f  '''

    k1 = h * f(t_n,y_n,*args)
	k2 = h * f(t_n+(1/4)*h,y_n+(1/4)*k1*h)
	k3 = h * f(t_n+(3/8)*h,y_n+(3/32)*h*(k1+3*k2))
  	k4 = h * f(t_n+(12/13)*h,y_n+(12/2197)*h*(161*k1-600*k2+608*k3))
  	k5 = h * f(t_n+h,y_n+(1/4104)*h*(8341*k1-32832*k2+29440*k3-845*k4))
  	k6 = h * f(t_n+(0.5)*h,y_n+h*(-(8/27)*k1+2*k2-(3544/2565)*k3+(1859/4104)*k4-(11/40)*k5))

    return y_n + (k1 + 2.*k2 + 2.*k3 + k4)/6.


def rk8_step(f,y_n,t_n,h,*args):
    ''' Runge-kutta of eight order integration step to get y_n+1

        INPUTS: f - first derivative function on dy/dt = f(y,t)
                y_n - initial condition
                t_n - initial condition
                h - time step
                args - arguments to pass to f  '''

    k1 = h * 
    k2 = h * 
    k3 = h * 
    k4 = h * 

    return y_n + (k1 + 2.*k2 + 2.*k3 + k4)/6.


def solve_rk4(f_rhs,t_f,t_step,ci,*f_args):
    ''' Rk4 implementation to solve a system of linear first order differential equations  
        
        INPUTS: f_rhs - rist of right hand side of the differential equation system
                t_f - final time to integrate the system
                t_step - time step
                ci - initial conditions [y0.t0]
                f_args - argumets to pass to the rhs functions'''
    
    in_cond = [np.array(ci).reshape(-1,1)]

    while in_cond[-1][-1] <= t_f:

        t,y = in_cond[-1][-1], in_cond[-1][:-1]

        y = rk4_step(f_rhs,y,t,t_step,[*f_args])

        ci_arr = np.concatenate((y.reshape(-1,1),np.array([t+t_step]).reshape(-1,1)))

        in_cond.append(ci_arr)

    return in_cond

