"""Hybrid (Symbolic_Numeric) indefinite integration"""

import numbers
from math import isclose
import numpy as np

from sympy import Number, Rational, Integer
from sympy import solve, expand, simplify, prod, diff, lambdify
from sympy import exp, log, sqrt, floor, frac, denom, numer, degree
from sympy import sin, cos, tan, csc, sec, cot
from sympy import asin, acos, atan, acsc, asec, acot
from sympy import sinh, cosh, tanh, csch, sech, coth
from sympy import asinh, acosh, atanh, acsch, asech, acoth
from sympy import Si, Ci, Ei, erfi

from numpy.linalg import qr
from scipy.sparse.linalg import lsqr


def integrate(eq, x, num_trials=10, depth=1, abstol=1e-6, verbose=False,
              radius=5.0, ret_hints=False):
    """
    Computes indefinite integral of univariable expressions with constant 
    coeficients. 

    The algorithm uses the symbolic-numeric methodolgy. The symbolic part 
    is similar to the Risch-Bronstein's poor man's integrator and generates 
    a list of ansatz (candidate terms). The numerical part uses sparse 
    regression adopted from Sparse identification of nonlinear dynamics 
    (SINDy) algorithm. 	
    
    Args:
        eq: integrand as a univariate expression with constant coefficients
        x: independent variable
        
    Options:
        num_trials (default 10): the number of numerical trials for each basis
        abstol (default 1e-6): the tolerance used for various numerical routines
        radius: radius of the disk in the complex plane used to generate test points
        verbose (default False): print additional and debugging information
        depth (default 1): the depth of ansatz expansion by repetitive differentiation
        ret_hints (default False): returns a list of pruned ansatzes (see hints() below)
        
    Returns:
        the solution (integral of eq) or 0 if no solution is found
    
    """
    if not eq.has_free(x):
        return eq*x

    if is_holonomic(eq, x):
        basis = blender(eq, x)

        for i in range(num_trials):
            sol = solve_sparse(eq, x, basis, abstol=abstol, radius=radius,
                               verbose=verbose, ret_hints=ret_hints)
            if sol != 0:
                if verbose:
                    print(f'solution found by grinder in {i+1} steps')
                return sol

    basis = generate_ansatz(eq, x)

    for i in range(num_trials):
        sol = solve_sparse(eq, x, basis, abstol=abstol, radius=radius,
                           verbose=verbose, ret_hints=ret_hints)
        if sol != 0:
            if verbose:
                print(f'solution found by the base ansatz generator in {i+1} steps')
            return sol

    for _ in range(depth):
        basis = expand_basis(basis, x)

        for i in range(num_trials):
            sol = solve_sparse(eq, x, basis, abstol=abstol, radius=radius,
                               verbose=verbose, ret_hints=ret_hints)
            if sol != 0:
                if verbose:
                    print(f'solution found by the expanded ansatz generator in {i+1} steps')
                return sol

    return 0


def hints(eq, x, **kwargs):
    """
    Ansatz generator. This function provides a filtered list of 
    ansatzes to be used by other integration routines. Specifically, 
    it is designed to work with sympy.heurisch function by passing 
    the results of hyint.hints to the hints argument of heurisch.
    
    This function also works with expressions containing symbolic 
    constants.
    
    Examples
    ========
    
    >>> from sympy import sqrt
    >>> from sympy.integrals.heurisch import heurisch
    >>> from sympy.abc import a, x, y
    >>> import hyint   

    >>> y = log(log(x) + a) / x

    >>> heurisch(y, x)
    # no solution

    >>> hints = hyint.hints(y, x)
    >>> print(hints)
    [log(a + log(x)), log(x)*log(a + log(x)), log(x)]
    
    >>> heurisch(y, x, hints=hints)
    a*log(a + log(x)) + log(x)*log(a + log(x)) - log(x)
    
    
    Args:
        eq: integrand as a univariate expression with constant coefficients
        x: independent variable
        
    Options:
        kwargs is passed to integrate and includes the same options
        
    Returns:
        a list of ansatzes or 0 if None if no solution found
    
    """
    sol = integrate(eq, x, ret_hints=True, **kwargs)
    return sol if sol else None


##################### ansatz generation #################################

def equivalent(y, x):
    """Removes numerical coefficients from terms"""
    y = expand(y)

    if y.is_Add:
        return sum(equivalent(u, x) for u in y.args)

    if y.is_Mul:
        return prod([u if u.has_free(x) else 1 for u in y.args])

    if isinstance(y, Number):
        return 1

    return y


def split_terms(eq, x):
    """Splits an expression into a list of terms"""
    if eq.is_Add:
        return [equivalent(u, x) for u in eq.args]
    return [equivalent(eq, x)]


def is_holonomic(y, x):
    """
    Checks whether y is a holonomic function, i.e., is closed 
    under differentiation w.r.t. x
    
    For our purpose, we define a holonomic function as one composed
    of the positive powers of x, sin, cos, exp, sinh, and cosh
    
    Args:
        y: the expression to check for holonomy
        x: independent variable
        
    Returns:
        True if y is holonomic
    """
    if isinstance(y, (Number, numbers.Number)):
        return True

    if y == x:
        return True

    if y.is_Function:
        return y.func in [sin, cos, sinh, cosh, exp]

    if y.is_Pow:
        p = y.args[0]
        k = y.args[1]
        return p.is_polynomial() and isinstance(k, Integer) and k > 0

    if y.is_Add or y.is_Mul:
        return all(is_holonomic(t, x) for t in y.args)

    return False


def blender(y, x, n=5):
    """
    Generates a list of ansatzes based on repetative differentiation. 
    It works for holonomic functions, which are closed under differentiation.
    """
    basis = y
    for _ in range(n):
        for t in split_terms(basis, x):
            basis += equivalent(diff(t, x), x)
    return split_terms(basis, x)


def generate_ansatz(y, x):
    """
    The main ansatz generation entry point.

    Note that all different ansatz_**** functions below have the same signature. 

    Args:
        y: a univariate expression 
        x: indepedent variable

    Returns:
        a list of ansatzes
    """
    basis = equivalent(x + ansatz(y, x), x)
    return split_terms(basis, x)


def ansatz(y, x):
    """The generic ansatz generator for a sympy expression."""

    if y.is_Add:
        return ansatz_Add(y, x)

    if y.is_Mul:
        return ansatz_Mul(y, x)

    if y.is_Pow:
        return ansatz_Pow(y, x)

    if y.is_Function:
        return integrate_fun(y, x)

    if y.is_polynomial():
        return ansatz_poly(y, 1, x)

    return y


def ansatz_Add(y, x):
    """Ansatz generator for an Add node."""
    assert y.is_Add
    return sum(ansatz(u, x) for u in y.args)


def ansatz_Mul(y, x):
    """Ansatz generator for a Mul node."""
    assert y.is_Mul
    w = 0
    for u in y.args:
        if u.is_Pow:
            q = ansatz_Pow(u, x)
        elif u.is_Function:
            q = integrate_fun(u, x)
        elif u.is_polynomial():
            q = ansatz_poly(u, 1, x)
        else:
            q = u
        # the first 1 accounts for the contant of integration
        w += (q + 1) * (y / u + 1)
    return w


def ansatz_Pow(y, x):
    """
    Ansatz generator for a Pow node.
    
    Compared to ansatz_Add and ansatz_Mul, ansatz_Pow has a more
    complex logic and is the central component of ansatz generation
    machinery.
    """
    assert y.is_Pow
    p = y.args[0]
    k = y.args[1]

    if p.is_Function:
        if k>0:
            return integrate_fun(p, x) * (1 + p**(k-1))

        return integrate_fun_inv(p, x) * (1 + p**(k+1))

    if p.is_polynomial():
        if isinstance(k, Rational) and denom(k) == 2:
            return ansatz_sqrt(p, k, x)

        if k > 0:
            return ansatz_poly(p, k, x)

        return ansatz_poly_inv(p, k, x)

    # TODO: needs to expand the default processing
    if k > 0:
        return expand((1+x) * y)

    return y + log(1 / y)


def ansatz_poly(p, k, x):
    """
    Returns the antiderivative ansatz for for p**k, where
    p is a polynomial and k is a positive integer
    """
    assert p.is_polynomial() and k > 0
    return expand((1+x) * p**k)


def ansatz_poly_inv(p, k, x, abstol=1e-8):
    """
    Returns the antiderivative ansatz for for p**k, where
    p is a polynomial and k is a negative integer
    """
    assert p.is_polynomial() and k < 0

    h = p**k + log(p)
    for i in range(0, int(-degree(p,x) * k)):
        h += x**i * p**k

    if len(p.free_symbols) > 1:
        return h

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
            J = solve(x**2+1)[1]    # symbolic sqrt(-1)
            real = simplify(r + s) / 2
            imag = simplify(r - s) / (2*J)
            h += atan((x - real) / imag)
            h += log(x**2 - 2*x*real + real**2 + imag**2)
    return h


def ansatz_sqrt(p, k, x):
    """
    Ansatz generator for expressions with a square root.
    The input expression is p**k, where k is a Rational
    with denominator of 2.
    """
    assert p.is_polynomial()
    h = p**k + p**(k+1)
    if degree(p, x) == 2:
        Δ = diff(p, x)
        h += log(Δ + 2*sqrt(p))

        if len(p.free_symbols) == 1:
            roots = solve(p)
            if len(roots) == 2 and isclose(sum(roots), 0):
                a = np.abs(roots[0])
                h += asinh(x / a) + asin(x / a)

    return h


def expand_basis(basis, x):
    """
    Expands a given basis (list of ansatz) by differentiation and introducing x
    
    Args:
        basis: the old basis as a list of ansatzes
        x: independent variable
        
    Returns:
        an expanded list of ansatzes, which is a superset of basis
    """
    B0 = sum(basis)
    basis = split_terms(expand((B0 + diff(B0, x))), x)
    basis += [x*y for y in basis if x*y not in basis]
    return basis


####################### Integration rules ##############################

def integrate_fun(y, x):
    """
    Integration table 
    if y = f(u(x)), it returns F(u(x))/u', where F is the anti-derivative of f
    """
    op = y.func
    u = y.args[0]
    du = diff(u, x)

    if du == 0:
        return 0

    match str(op):
        # exponential functions
        case "exp":
            h = (exp(u) + Ei(u)) / du
            if u.is_polynomial() and degree(u, x) == 2:
                return h + erfi(x)
            return h
        case "log":
            h = (u + u*log(u)) / du
            if u.is_polynomial():
                return h + ansatz_poly_inv(u, -1, x)
            return h
        # trigonometrical functions
        case "sin":
            return cos(u) / du + Si(u)
        case "cos":
            return sin(u) / du + Ci(u)
        case "tan":
            return (log(cos(u)) + u*tan(u)) / du
        case "csc":
            return log(csc(u) + cot(u)) / du
        case "sec":
            return log(sec(u) + tan(u)) / du
        case "cot":
            return (log(sin(u)) + u*cot(u)) / du
        # hyperbolic functions
        case "sinh":
            return cosh(u) / du
        case "cosh":
            return sinh(u) / du
        case "tanh":
            return log(cosh(u)) / du
        case "csch":
            return log(tanh(u)) / du
        case "sech":
            return atan(sinh(u)) / du
        case "coth":
            return log(sinh(u)) / du
        # inverse trigonometrical functions
        case "asin":
            return (u * asin(u) + sqrt(1 - u**2)) / du
        case "acos":
            return (u * acos(u) + sqrt(1 - u**2)) / du
        case "atan":
            return (u * atan(u) + log(1 + u**2)) / du
        case "acsc":
            return (u * acsc(u) + asinh(u)) / du
        case "asec":
            return (u * asec(u) + acosh(u)) / du
        case "acot":
            return (u * acot(u) + log(1 + u**2)) / du
        # inverse hyperbolic functions
        case "asinh":
            return (u * asinh(u) + sqrt(u**2 - 1)) / du
        case "acosh":
            return (u * acosh(u) + sqrt(u**2 - 1)) / du
        case "atanh":
            return (u * atanh(u) + log(1 + u)) / du
        case "acsch":
            return (u * acsch(u) + asinh(u)) / du
        case "asech":
            return (u * asech(u) + acosh(u)) / du
        case "acoth":
            return (u * acoth(u) + log(1 + u)) / du


def integrate_fun_inv(y, x):
    """
    Integration table for reciprocal of standard functions
    if y = 1 / f(u(x)), it returns F(u(x))/u', where F is the anti-derivative of 1 / f
    """
    op = y.func
    u = y.args[0]
    du = diff(u, x)

    if du == 0:
        return 0

    match str(op):
        # exponential and logarithmic functions
        case "log":
            return (log(log(u)) + Ei(log(u))) / du # need to add li(u)
        # reciprocal of trigonometric functions
        case "sin":
            return (log(cos(u) + 1) + log(cos(u) - 1) + log(sin(u))) / du
        case "cos":
            return (log(sin(u) + 1) + log(sin(u) - 1) + log(cos(u))) / du
        case "tan":
            return (log(sin(u)) + log(tan(u))) / du
        case "csc":
            return (cos(u) + log(csc(u))) / du
        case "sec":
            return (sin(u) + log(sec(u))) / du
        case "cot":
            return (log(cos(u)) + log(cot(u))) / du
        # reciprocal of hyperbolic functions
        case "sinh":
            return (log(tanh(u / 2)) + log(sinh(u))) / du
        case "cosh":
            return (atan(sinh(u)) + log(cosh(u))) / du
        case "tanh":
            return (log(sinh(u)) + log(tanh(u))) / du
        case "csch":
            return (cosh(u) + log(csch(u))) / du
        case "sech":
            return (sinh(u) + log(sech(u))) / du
        case "coth":
            return (log(cosh(u)) + log(coth(u))) / du


##################### numerical utils ##############################

def init_basis_matrix(eq, x, basis, radius=5.0, m=2):
    """
    Generates test matrixes
    
    Args:
        eq: the integrand (a univariate expression with constant coefficients)
        x: indendent variable
        basis: the list of ansatzes
        
    Options:
        radius (default 5.0): radius of the disk in the complex plane used for test points
        m (default 2): the ratio of the test points to the number of ansatzes

    Returns:
        A: a complex matrix (m*n-by-n), where n is the number of ansatzes, and
            A[i, j] is the value of basis[j] at test point i.
        b: a complex vector (m*n elements), where b[i] is the value of eq at 
            test point i.
    """
    n = len(basis)

    s = subs_symbols(eq, x)

    diff_eq = lambdify(x, eq.subs(s))
    diff_basis = [lambdify(x, diff(u.subs(s), x)) for u in basis]
	# diff_basis = [lambdify(x, simplify(diff(u, x))) for u in basis]

    p = test_points(m*n, radius=radius)
    b = diff_eq(p)
    A = np.zeros((m*n,n,), dtype=complex)

    for i in range(n):
        A[:,i] = diff_basis[i](p)

    return A, b


def strip_basis(basis, x):
    """Strips constants from basis"""
    return [y for y in basis if not isinstance(y, numbers.Number) and y.has_free(x)]


def filter_list(basis, bools):
    """Filters basis (a list) by a boolean vector bools"""    
    return [y for y, b in zip(basis, bools) if b]


def test_points(n, radius=5.0):
    """
    Generates n complex random test points over a disk of radius r 
    centered at the origin
    """
    r = np.sqrt(radius*np.random.rand(n))
    θ = 2*np.pi*np.random.rand(n)
    return r * np.exp(1j * θ)


def solve_sparse(eq, x, basis, radius=5.0, abstol=1e-6, verbose=False, ret_hints=False):
    """
    The main entry point to find the integral of eq using sparse regression.
    
    Args:
        eq: the integrands, a univariate expression with constant coefficients
        x: independent variable
        basis: the list of ansatzes

    Options:
        radius: radius of the disk in the complex plane used for test points
        abstol: tolerance to find sparse representations
        
    Returns:
        the integer of eq or 0 if no solution is found
    """
    basis = strip_basis(basis, x)
    A, b = init_basis_matrix(eq, x, basis, radius=radius)
    A, b = normalize(A, b)
    l = find_independent_subset(A)
    A = A[:, l]
    basis = filter_list(basis, l)

    for j in range(-10, -5):
        ker, q = find_kernel_lsqr(A, b, threshold = exp(j))

        if (ϵ := sum(abs(A[:,ker] @ q - b))) < abstol:
            if verbose:
                print(f'found a solution with ϵ = {ϵ}')

            kernel = filter_list(basis, ker)

            if ret_hints:
                return kernel

            sol = 0
            for (coef, y) in zip(q, kernel):
                if np.abs(coef) > abstol:   # may not be needed
                    sol += nice(coef) * y
            return sol
    return 0


def normalize(A, b):
    """Normalizes A by dividing row-wise with b
    """
    B = np.empty_like(A)
    for i in range(A.shape[1]):
        B[:, i] = A[:, i] / b
    return B, np.ones_like(b)


def nice(x, abstol=1e-6, M=20):
    """Converts a number x to a nice (|denominator| < M) rational"""
    if np.abs(x.imag) > abstol:
        return x    # no rational complex number

    a = floor(x.real)
    r = frac(x.real)

    for den in range(2, M):
        if abs(round(r * den) - r * den) < abstol:
            s = Rational(round(r * den), den)
            if denom(s) == 1:
                return a + numer(s)
            return a + s
    return x


def find_independent_subset(A, abstol=1e-3):
    """
    Finds the set of linearly independent columns of A
    
    Args:
        A: a complex matrix 
        
    Options:
        abstol: tolerance used to define linear dependence
        
    Returns:
        a boolean list, where True means the corresponding column of A
            is in the independent list
    """
    _, R = qr(A)
    d = np.diag(R)
    return np.abs(d) > abstol


def find_kernel_lsqr(A, b, threshold=1e-6):
    """A simple sparse-regression routine
    
    It solves the linear equation A @ q = b by enforcing sparcity of
    q; i.e., tries to set as many elements of q to 0 as possible.
    
    Args:
        A: a complex matrix (m-by-n)
        b: a complex vector (m elements)
        
    Options:
        threshold: elements of q with absolute value below the threshold are
            set to 0
        
    Returns:
        q: a complex vector (n elements)
    """
    l0 = np.ones(A.shape[1]) > 0
    while True:
        q, *_ = lsqr(A[:,l0], b)
        l1 = abs(q) > threshold
        if all(l1):
            return l0, q
        l0[l0] = l1

#####################################################################

def subs_symbols(eq, x):
    """
    Returns a dictionary of the symbolic constants in the expression
    and random real value assignments.

    Args:
        eq: the integrands
        x: independent variable

    Returns:
        a dict of sym : value pairs
    """
    s = {}
    for v in eq.free_symbols:
        if v != x:
            s[v] = np.random.randn()
    return s

