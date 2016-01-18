import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

def dm_dt(v,m):
    return alpha_m(v)*(1-m) - beta_m(v)*m

def dn_dt(v,n):
    return alpha_n(v)*(1-n)-beta_n(v)*n

def dh_dt(v,h):
    return alpha_h(v)*(1-h)-beta_h(v)*h

def alpha_m(v):
    return (2.5 - 0.1*v)/(np.exp(2.5 - 0.1*v)-1.0)

def beta_m(v):
    return 4*np.exp(-v/18.0)

def alpha_n(v):
    return (0.1 - 0.01*v)/(np.exp(1.0 - 0.1*v)-1.0)

def beta_n(v):
    return 0.125*np.exp(-v/80.0)

def alpha_h(v):
    return 0.07*np.exp(-v/20.0)

def beta_h(v):
    return 1.0/(np.exp(3.0 - 0.1*v)+1.0)

## Parameters of the Hodgkin-Huxley model

# ENa = 115 - 65
# EK  = -12 - 65
# EL  = 10.6 - 65
params = [120.0,36.0,0.3,115.0,-12.0,10.6,1.0]

def dv_dt(v,m,n,h,params):
    gNa,gK,gL,ENa,EK,EL,C = params
    s1 = gNa*np.power(m,3)*h*(v - ENa)
    s2 = gK*np.power(n,4)*(v - EK)
    s3 = gL*(v - EL)
    return (-(s1 + s2 + s3) + I)/C

def s_dt(s, I,params):
    gNa,gK,gL,ENa,EK,EL,C = params

    # v, m, n, h = s1

    v=s[0,:]
    m=s[1,:]
    n=s[2,:]
    h=s[3,:]

    Ik = gNa*(m**3)*h*(v-ENa) + gK*(n**4)*(v-EK) + gL*(v-EL)

    v_dt = (I - Ik)/C
    m_dt = alpha_m(v)*(1-m) - beta_m(v)*m
    n_dt = alpha_n(v)*(1-n) - beta_n(v)*n
    h_dt = alpha_h(v)*(1-h) - beta_h(v)*h

    return np.array([v_dt, m_dt, n_dt, h_dt])

def s_dt_simple(s, I,params):
    gNa,gK,gL,ENa,EK,EL,C = params

    v, m, n, h = s
    Ik = gNa*(m**3)*h*(v-ENa) + gK*(n**4)*(v-EK) + gL*(v-EL)
    v_dt = (I - Ik)/C
    m_dt = alpha_m(v)*(1-m) - beta_m(v)*m
    n_dt = alpha_n(v)*(1-n) - beta_n(v)*n
    h_dt = alpha_h(v)*(1-h) - beta_h(v)*h

    return np.array([v_dt, m_dt, n_dt, h_dt])
