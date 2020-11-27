import numpy as np

def rk4_step(f,y_i,y_n,t_n,h,*args):
    ''' Runge-kutta of fourth order integration step to get y_n+1
        INPUTS: f - first derivative function on dy/dt = f(y,t)
                y_n - initial condition
                t_n - initial condition
                h - time step
                args - arguments to pass to f  '''

    k1 = h * f(t_n, y_n, *args)[0]
    k2 = h * f(t_n + h*0.5, y_n + k1*0.5, *args)[0]
    k3 = h * f(t_n + h*0.5, y_n + k2*0.5, *args)[0]
    k4 = h * f(t_n + h, y_n + k3, *args)[0]

    return y_i + (k1 + 2.*k2 + 2.*k3 + k4)/6.


def solve_rk4(f_rhs,t_f,t_step,ci,*f_args):
    ''' Rk4 implementation to solve a system of linear first order differential equations  
        
        INPUTS: f_rhs - list of right hand side of the differential equation system
                t_f - final time to integrate the system
                t_step - time step
                ci - initial conditions [y0.t0]
                f_args - argumets to pass to the rhs functions'''
    
    in_cond = [np.array(ci).reshape(-1,1)]
    all_cond = np.array(in_cond).reshape(1,-1)

    while in_cond[-1][-1] <= t_f:

        t,y = in_cond[-1][-1], in_cond[-1][:-1].reshape(-1,1)

        y_np1 = []
        for f,y_i in zip(f_rhs,y):
            y_np1.append(rk4_step(f,y_i,y,t,t_step,*f_args)[0])

        ci_arr = np.concatenate((np.array(y_np1).reshape(-1,1),np.array([t+t_step]).reshape(-1,1)))

        in_cond.append(ci_arr)
        all_cond = np.concatenate((all_cond,ci_arr.T),axis=0)

    return all_cond
