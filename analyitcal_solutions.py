

def fkdv_exact_solution(a, x, K = 0.6124):
    return a / (np.cosh(K * x)**2)

def exact_n_solutions(x, N):
    return N * (N + 1) * (np.cosh(x)**(-2))

def unforced_soluiton(x, t, )