from sympy import * 
from sympy import Poly
import numpy as np
from numpy.linalg import qr
import scipy.sparse as sparse
import numbers
from pysindy.optimizers import STLSQ, SSR

def integrate(eq, x, num_trials=10):
	if isinstance(eq, Number) or isinstance(eq, numbers.Number):
		return eq*x
		
	basis = grinder(eq, x)

	for i in range(num_trials):		
		sol = solve_sparse(eq, x, basis)
		if sol != 0:
			print(f"grinder ({i})")
			return sol
		
	basis = generate_ansatz(eq, x)

	for i in range(num_trials):		
		sol = solve_sparse(eq, x, basis)
		if sol != 0:
			print(f"ansatz ({i})")
			return sol
			
	basis = expand_basis(basis, x)	
	
	for i in range(num_trials):
		sol = solve_sparse(eq, x, basis)
		if sol != 0:
			print(f"expanded ({i})")
			return sol

	return 0
	
##################### ansatz generation #################################

def equivalent(y):
	y = expand(y)
	op = y.func
	
	if op == Add:
		return sum([equivalent(u) for u in y.args])
	elif op == Mul:
		return prod([1 if isinstance(u, Number) else u for u in y.args])
	elif isinstance(y, Number):
		return 1
	else:
		return y

def split_terms(basis):
	if basis.func == Add:
		return [equivalent(u) for u in basis.args]
	else:
		return [equivalent(basis)]

	
def grinder(y, x, n=5):
	basis = y
	for i in range(n):
		basis += diff(basis, x)
	return split_terms(basis)


def generate_ansatz(y, x):
	# y = equivalent(y)
	basis = equivalent(x + ansatz(y, x))
	return split_terms(basis)


def ansatz(y, x):
	# print(f"ansatz({y})")
	op = y.func
	
	if op == Add:	
		return ansatz_Add(y, x)
	elif op == Mul:
		return ansatz_Mul(y, x)
	elif op == Pow:
		return ansatz_Pow(y, x)
	elif is_fun(y):
		return integrate_fun(y, x)
	elif is_poly(y):
		return ansatz_poly(y, 1, x)
	else:
		return y
	
	
def ansatz_Add(y, x):
	# print(f"ansatz_Add({y})")
	assert y.func == Add
	return sum([ansatz(u, x) for u in y.args])
	
def ansatz_Mul(y, x):
	# print(f"ansatz_Mul({y})")
	assert y.func == Mul
	w = 0
	for u in y.args:
		if u.func == Pow:
			q = ansatz_Pow(u, x)
		elif is_fun(u):			
			q = integrate_fun(u, x)
		elif is_poly(u):
			q = ansatz_poly(u, 1, x)
		else:
			q = u
		w += (q + 1) * (y / u + 1)
	return w
	
def ansatz_Pow(y, x):
	# print(f"ansatz_Pow({y})")
	assert y.func == Pow
	p = y.args[0]
	k = y.args[1]	
	
	if is_fun(p):
		if k>0:
			return integrate_fun(p, x) * p**(k-1)
		else:
			return integrate_fun_inv(p, x) * p**(k+1)
	elif is_poly(p):
		if isinstance(k, Rational) and denom(k) == 2:
			return candidate_sqrt(p, k, x)
		elif k > 0:
			return ansatz_poly(p, k, x)
		else:
			return ansatz_poly_inv(p, k, x)
	else:
		if k > 0:
			return expand((1+x)*y)
		else:
			return y + log(1 / y)

def is_fun(y):
	return len(y.args) == 1
	
def is_poly(y):
	return y.is_polynomial()
	
def ansatz_poly(p, k, x):
	# print(f"ansatz_poly({y})")
	assert is_poly(p)
	return expand((1+x) * p**k)

def ansatz_poly_inv(p, k, x, abstol=1e-8):
	# print(f"ansatz_poly({y})")
	assert is_poly(p)

	h = p**k + log(p)
	for i in range(0,-degree(p)*k):
		h += x**i * p**k
		
	roots = solve(p)
	n = len(roots)
	i = 0
	while i < n:
		r = roots[i]
		if abs(complex(r).imag) < abstol:
			i += 1
			h += log(x - r)			
		else:	# complex
			s = roots[i + 1]
			i += 2
			I = solve(x**2+1)[1]
			ℜ = simplify(r + s) / 2
			ℑ = simplify(r - s) / (2*I)
			h += atan((x - ℜ) / ℑ)
			h += log(x**2 - 2*x*ℜ + ℜ**2 + ℑ**2)
	return h

	
def expand_basis(basis, x):	
	δb = split_terms(diff(sum(basis), x))
	basis += [eq for eq in δb if eq not in basis]
	basis += [eq*x for eq in δb if eq*x not in basis]
	return basis

def candidate_sqrt(p, k, x):
	assert is_poly(p)
	h = p**k + p**(k+1)
	if degree(p) == 2: 
		Δ = diff(p, x)
		h += log(Δ + 2*sqrt(p))
	return h


####################### Integration rules ##############################

def integrate_fun(y, x):
	# print(f"integrate_fun({y})")
	op = y.func
	μ = y.args[0]
	dμ = diff(μ, x)

	if dμ == 0:
		return 0
	
	match str(op):
		# exponential functions
		case "exp":		
			h = (exp(μ) + Ei(μ)) / dμ
			if is_poly(μ) and degree(μ) == 2:
				return h + erfi(x)
			else:
				return h
		case "log":
			h = (μ + μ*log(μ)) / dμ
			if is_poly(μ):
				return h + ansatz_poly_inv(μ, 1, x)
			else:
				return h
		# trigonometrical functions
		case "sin":
			return cos(μ) / dμ + Si(μ)
		case "cos":
			return sin(μ) / dμ + Ci(μ)
		case "tan":
		    return log(cos(μ)) / dμ
		case "csc":
			return log(csc(μ) + cot(μ)) / dμ
		case "sec":
			return log(sec(μ) + tan(μ)) / dμ
		case "cot":
		    return log(sin(μ)) / dμ
		# hyperbolic functions
		case "sinh":
			return cosh(μ) / dμ
		case "cosh":
			return sinh(μ) / dμ
		case "tanh":
			return log(cosh(μ)) / dμ
		case "csch":
			return log(tanh(μ)) / dμ
		case "sech":
			return atan(sinh(μ)) / dμ
		case "coth":
			return log(sinh(μ)) / dμ
		# inverse trigonometrical functions
		case "asin":
			return (μ * asin(μ) + sqrt(1 - μ**2)) / dμ
		case "acos":
			return (μ * acos(μ) + sqrt(1 - μ**2)) / dμ
		case "atan":
			return (μ * atan(μ) + log(1 + μ**2)) / dμ
		case "acsc":
			return (μ * acsc(μ) + asinh(μ)) / dμ
		case "asec":
			return (μ * asec(μ) + acosh(μ)) / dμ
		case "acot":
			return (μ * acot(μ) + log(1 + μ**2)) / dμ
		# inverse hyperbolic functions
		case "asinh":
			return (μ * asinh(μ) + sqrt(μ**2 - 1)) / dμ
		case "acosh":
			return (μ * acosh(μ) + sqrt(μ**2 - 1)) / dμ
		case "atanh":
			return (μ * atanh(μ) + log(1 + μ)) / dμ
		case "acsch":
			return (μ * acsch(μ) + asinh(μ)) / dμ
		case "asech":
			return (μ * asech(μ) + acosh(μ)) / dμ
		case "acoth":
			return (μ * acoth(μ) + log(1 + μ)) / dμ
			



def integrate_fun_inv(y, x):
	# print(f"integrate_fun_inv({y})")
	op = y.func
	μ = y.args[0]
	dμ = diff(μ, x)

	if dμ == 0:
		return 0
	
	match str(op):
		# exponential and logarithmic functions
		case "log":
			return log(log(μ)) / dμ # need to add li(μ)
		# reciprocal of trigonometric functions
		case "sin":
			return (log(cos(μ) + 1) + log(cos(μ) - 1) + log(sin(μ))) / dμ
		case "cos":
			return (log(sin(μ) + 1) + log(sin(μ) - 1) + log(cos(μ))) / dμ
		case "tan":
			return (log(sin(μ)) + log(tan(μ))) / dμ
		case "csc":
			return (cos(μ) + log(csc(μ))) / dμ
		case "sec":
			return (sin(μ) + log(sec(μ))) / dμ
		case "cot":
			return (log(cos(μ)) + log(cot(μ))) / dμ
		# reciprocal of hyperbolic functions
		case "sinh":
			return (log(tanh(μ / 2)) + log(sinh(μ))) / dμ
		case "cosh":
			return (atan(sinh(μ)) + log(cosh(μ))) / dμ
		case "tanh":
			return (log(sinh(μ)) + log(tanh(μ))) / dμ
		case "csch":
			return (cosh(μ) + log(csch(μ))) / dμ
		case "sech":
			return (sinh(μ) + log(sech(μ))) / dμ
		case "coth":
			return (log(cosh(μ)) + log(coth(μ))) / dμ
			


		
##################### numerical utils ##############################

def init_basis_matrix(eq, x, basis, radius=5.0, abstol=1e-6, m=2):
	if basis[0] == 1:
		basis = basis[1:]	
	n = len(basis)

	eq_λ = lambdify(x, eq)
	Δbasis_λ = [lambdify(x, diff(u, x)) for u in basis]	
	# Δbasis_λ = [lambdify(x, simplify(diff(u, x))) for u in basis]	

	p = test_points(m*n)
	b = eq_λ(p)
	A = np.zeros((m*n,n,), dtype=complex)
	
	for i in range(n):	
		A[:,i] = Δbasis_λ[i](p) 
	 
	return A, b, basis
	
	 
def test_points(n, radius=5.0):
	return np.exp(2j*np.pi*np.random.rand(n)) * np.sqrt(radius*np.random.rand(n))


def filter_list(elems, bools):
	return [e for e, b in zip(elems, bools) if b]	


def solve_sparse(eq, x, basis, radius=5.0, abstol=1e-6):
	A, b, basis = init_basis_matrix(eq, x, basis, radius=radius, abstol=abstol)
	A, b = normalize(A, b)	
	l = find_independent_subset(A)
	A = A[:, l]
	basis = filter_list(basis, l)
	
	for j in range(-10, -5):
		# ker, q = find_kernel_STLSQ(A, b, threshold = exp(j))
		ker, q = find_kernel_lsqr(A, b, threshold = exp(j))				
		ϵ = sum(abs(A[:,ker] @ q - b))
		
		if ϵ < abstol:				
			β = filter_list(basis, ker)
			sol = 0	
			for i in range(len(q)):
				if np.abs(q[i]) > abstol:				
					if np.abs(q[i].imag) < abstol:
						sol += nice(q[i].real) * β[i]
					else:
						sol += q[i] * β[i]
			print(j, ' -> ', ϵ)
			return sol
	return 0

	
def normalize(A, b):
	B = np.empty_like(A)
	for i in range(A.shape[1]):
		B[:, i] = A[:, i] / b
	return B, np.ones_like(b)	

	
def nice(x, abstol=1e-6, M=20):
    a = floor(x)
    r = frac(x)
    
    for den in range(2, M):
    	if abs(round(r * den) - r * den) < abstol:
    		s = Rational(round(r * den), den)
    		if denom(s) == 1:
    			return a + numer(s)
    		else:
    			return a + s
    return x

def find_independent_subset(A, abstol=1e-3):
	Q, R = qr(A)
	d = np.diag(R)
	return np.abs(d) > abstol
	
def find_kernel_STLSQ(A, b, threshold=1e-6):
	AA = np.vstack((A.real, A.imag))
	bb = np.hstack((b.real, b.imag))	
	opt = STLSQ(threshold)	
	# opt = SSR()
	fit = opt.fit(AA, bb)
	ind = fit.ind_[0]
	q = sparse.linalg.lsqr(A[:,ind], b, x0 = fit.coef_[0][ind])[0]		
	return ind, q
		

def find_kernel_lsqr(A, b, threshold=1e-6):	
	l0 = np.ones(A.shape[1]) > 0
	while true:
		q = sparse.linalg.lsqr(A[:,l0], b)[0]
		l1 = abs(q) > threshold
		if all(l1):
			return l0, q
		l0[l0] = l1

	
