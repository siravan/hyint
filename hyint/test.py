from sympy import *
from sympy.integrals.heurisch import heurisch 
from . import hyint

x = Symbol('x')

######################################################

basic_integrals = [
    # Basic Forms
    1,
    x**2,
    4*x**3,
    # Integrals of Rational Functions
    1 / x,
    1 / (2*x + 5),
    1 / (x + 1)**2,
    (x + 3)**3,
    x * (x - 2)**4,
    1 / (1 + x**2),
    1 / (9 + x**2),
    x / (4 + x**2),
    x**2 / (16 + x**2),
    x**3 / (1 + x**2),
    1 / (x**2 - 5*x + 6),
    1 / (x**2 + x + 1),
    x / (x + 4)**2,
    x / (x**2 + x + 1),
    # Integrals with Roots
    sqrt(x - 2),
    1 / sqrt(x - 1),
    1 / sqrt(x + 1),
    1 / sqrt(4 - x),
    x * sqrt(x - 3),
    sqrt(2*x + 5),
    (3*x - 1)**1.5,
    x / sqrt(x - 1),
    x / sqrt(x + 1),
    sqrt(x / (4 - x)),
    sqrt(x / (4 + x)),
    x * sqrt(2*x + 3),
    sqrt(x * (x + 2)),
    sqrt(x**3 * (x + 3)),
    sqrt(x**2 + 4),
    sqrt(x**2 - 4),
    sqrt(4 - x**2),
    x * sqrt(x**2 + 9),
    x * sqrt(x**2 - 9),
    1 / sqrt(x**2 + 4),
    1 / sqrt(x**2 - 4),
    1 / sqrt(4 - x**2),
    x / sqrt(x**2 + 4),
    x / sqrt(x**2 - 4),
    x / sqrt(4 - x**2),
    x**2 / sqrt(x**2 + 4),
    x**2 / sqrt(x**2 - 4),
    sqrt(x**2 - 5*x + 6),
    x * sqrt(x**2 - 5*x + 6),
    1 / sqrt(x**2 - 5*x + 6),
    1 / (4 + x**2)**1.5,
    # Integrals with Logarithms
    log(x),
    x * log(x),
    x**2 * log(x),
    log(2*x) / x,
    log(x) / x**2,
    log(2*x + 1),
    log(x**2 + 4),
    log(x**2 - 4),
    log(x**2 - 5*x + 6),
    x * log(x + 2),
    x * log(9 - 4*x**2),
    log(x)**2,
    log(x)**3,
    x * log(x)**2,
    x**2 * log(x)**2,
    # Integrals with Exponentials
    exp(x),
    sqrt(x) * exp(x),
    x * exp(x),
    x * exp(3*x),
    x**2 * exp(x),
    x**2 * exp(5*x),
    x**3 * exp(x),
    x**3 * exp(2*x),
    exp(x**2),
    x * exp(x**2),
    # Integrals with Trigonometric Functions
    sin(4*x),
    sin(x)**2,
    sin(x)**3,
    cos(3*x),
    cos(x)**2,
    cos(2*x)**3,
    sin(x) * cos(x),
    sin(3*x) * cos(5*x),
    sin(x)**2 * cos(x),
    sin(3*x)**2 * cos(x),
    sin(x) * cos(x)**2,
    sin(x) * cos(5*x)**2,
    sin(x)**2 * cos(x),
    sin(x)**2 * cos(x)**2,
    sin(4*x)**2 * cos(4*x)**2,
    tan(x),
    tan(7*x),
    tan(x)**2,
    tan(x)**3,
    sec(x),
    sec(x) * tan(x),
    sec(x)**2 * tan(x),
    csc(x),
    sec(x) * csc(x),
    # Products of Trigonometric Functions and Monomials
    x * cos(x),
    x * cos(3*x),
    x**2 * cos(x),
    x**2 * cos(5*x),
    x * sin(x),
    x * sin(3*x),
    x**2 * sin(x),
    x**2 * sin(5*x),
    x * cos(x)**2,
    x * sin(x)**2,
    x * tan(x)**2,
    x * sec(x)**2,
    x**3 * sin(x),
    x**4 * cos(2*x),
    sin(x)**2 * cos(x)**3,
    # Products of Trigonometric Functions and Exponentials
    exp(x) * sin(x),
    exp(3*x) * sin(2*x),
    exp(x) * cos(x),
    exp(2*x) * cos(7*x),
    x * exp(x) * sin(x),
    x * exp(x) * cos(x),
    # Integrals of Hyperbolic Functions
    cosh(x),
    exp(x) * cosh(x),
    sinh(3*x),
    exp(2*x) * sinh(3*x),
    tanh(x),
    exp(x) * tanh(x),
    cos(x) * cosh(x),
    cos(x) * sinh(x),
    sin(x) * cosh(x),
    sin(x) * sinh(x),
    sinh(x) * cosh(x),
    sinh(3*x) * cosh(5*x),
    # Misc
    exp(x) / (1 + exp(x)),
    cos(exp(x)) * sin(exp(x)) * exp(x),
    cos(exp(x))**2 * sin(exp(x)) * exp(x),
    1 / (x * log(x)),
    1 / (exp(x) - 1),
    1 / (exp(x) + 5),
    sqrt(x) * log(x),
    log(log(x)) / x,
    x**3 * exp(x**2),
    sin(log(x)),
    x * cos(x) * exp(x),
    log(x - 1)**2,
    1 / (exp(2*x) - 1),
    exp(x) / (exp(2*x) - 1),
    x / (exp(2*x) - 1),
    # derivative-divide examples (Lamangna 7.10.2)
    exp(x) * exp(exp(x)),
    exp(sqrt(x)) / sqrt(x),
    log(log(x)) / (x * log(x)),
    log(cos(x)) * tan(x),
    # rothstein-Trager examples (Lamangna 7.10.9)
    1 / (x**3 - x),
    1 / (x**3 + 1),
    1 / (x**2 - 8),
    (x + 1) / (x**2 + 1),
    x / (x**4 - 4),
    x**3 / (x**4 + 1),
    1 / (x**4 + 1),
    # exponential/trigonometric/logarithmic integral functions
    exp(2*x) / x,
    exp(x + 1) / (x + 1),
    x * exp(2*x**2 + 1) / (2*x**2 + 1),
    sin(3*x) / x,
    sin(x + 1) / (x + 1),
    cos(5*x) / x,
    x * cos(x**2 - 1) / (x**2 - 1),
    1 / log(3*x - 1),
    1 / (x * log(log(x))),
    x / log(x**2),
    # bypass = true
    # β,      # turn of bypass = true
    (log(x - 1) + (x - 1)**-1) * log(x),
    exp(x) / x - exp(x) / x**2,
    cos(x) / x - sin(x) / x**2,
    1 / log(x) - 1 / log(x)**2,
]

#####################################################################

def run_tests(try_heurisch=False):
    n = 0
    ks = 0
    kh = 0
	
    for eq in basic_integrals:
        n += 1
        try:
            sol = hyint.integrate(eq, x)
		    
            if sol:
                ks += 1

            if try_heurisch:
                h = heurisch(eq, x)		        
                if h:
                    kh += 1		            
                print(f'{eq} => {sol} ({h})')	
            else:
                print(f'{eq} => {sol}')	

        except Exception as e:
            print(e)

    print(f"{ks} out of {n}")
    if try_heurisch:
    	print(f"{kh} out of {n} for heurisch")
	
	
