import numpy as np

def sech_sqrd(x):
    return np.cosh(x) ** (-2)

def get_real_solutions(exact_wave_fn, x_array, time_array):
    real_sol = np.zeros((len(time_array), len(x_array)))
    for i in range(len(time_array)):
        real_sol[i] = exact_wave_fn(x_array, time_array[i])
    return real_sol

# unforced n soltion solution of the KdV equation
def unforced_kdv_solution(x, t):
    """ Reduces the forcing term to zero allowing the KdV equation to be solved analytically. """
    return np.zeros_like(x)  # Placeholder for the actual analytical solution of the unforced KdV equation

def n_soltion_exact( x, t, n = 3):
    return n *(n+1) * sech_sqrd(x)

# Xiao's exact solution for the forced KdV equation
def xiao_traveling_forcing_exact( alpha_zg, beta, delta, kappa, b, x, t):
    """ t = time"""
    wave_const = np.exp(b[2]*(t**2) + b[1]*t + b[0])
    return ((12*kappa*beta) / alpha_zg) * ( (kappa**3)*(4*beta - delta) - (2*t*b[2] + b[1])*wave_const) * sech_sqrd(kappa * (x - delta * (kappa**2) * t) - wave_const)

def xiao_traveling_wave_exact( beta, delta, kappa, b, x, t):
    wave_const = np.exp(b[2]*(t**2) + b[1]*t + b[0])
    return (12*beta) * (kappa**2) * sech_sqrd(kappa*(x - delta*(kappa**2)*t) - wave_const)

def xiao_special_forcing_exact(alpha, beta, delta, k, A, b, x, t):
    """ t = time"""
    _wave_const = b[2]*np.arctan(A*t) + b[1]*t + b[0]
    return ((12*k*beta / alpha)
            * ( (k**3) * (4*beta - delta) - ( b[2]*A / (1 + (A**2) * (t**2)) ) - b[1])
            * sech_sqrd(k * (x - delta*(k**2)*t) - _wave_const)
)

def xiao_special_wave_exact( beta, k, A, b, x, t):
    """ t = time"""
    _wave_const = b[2] * np.arctan(A*t) + b[1] * t + b[0]
    return 12 * beta * (k ** 2) * sech_sqrd(k * (x - 4 * (k ** 2) * t) - _wave_const)

# Wu's exact solution for the forced KdV equation
def wu_wave_exact(A, K, x, t):
    return A * (np.cosh(K * x) ** (-2))

def wu_forcing_exact( b1, K, x, t):
    return 2 * b1 * (np.cosh(K * x) ** (-2))