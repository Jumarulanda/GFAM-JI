import numpy as np

def c_n_dot(t,c_n,n,*args):
    cn1,cn2,omeg,det,lamb = args

    p1 = omeg*(n + 0.5) - det*0.5
    p2 = lamb*np.sqrt(n+1)*cn2[n+1]
    p3 = lamn*np.sqrt(n)*cn2[n-1]*(n>0)

    return p1+p2+p3


class noRWA_rabi:
    def __init__(self, n_tunc):
        self.n_trunc = n_trunc


    def set_ODE_syst(self):
        ode_s = []

        for n in range(self.n_trunk+1):
            f_n = lambda 
