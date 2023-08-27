# hyint: Hybrid (Symbolic-Numeric) Integration Package 

**hyint** is a Python package to computes indefinite integral of univariable expressions 
with constant coeficients using symbolic-numeric methodolgy. It is built on top of **sympy** 
symbolic manipulation ecosystem of Python, but applies numerical methods to solve integral
problems. 

**hyint** can solve a large subset of basic standard integrals (polynomials, exponential/logarithmic, 
trigonometric and hyperbolic, inverse trigonometric and hyperbolic, rational and square root) (
see [The Basis of Symbolic-Numeric Integration](https://github.com/SciML/SymbolicNumericIntegration.jl/blob/main/docs/theory.ipynb)
for a brief introduction to the algorithm. It can even find some integrals not found by the
current version of **sympy.integrate**.
	
The symbolic part of the algorithm is similar (but not identical) to the Risch-Bronstein's poor man's integrator 
and generates a list of ansatzes (candidate terms). The numerical part uses sparse regression 
adopted from the Sparse identification of nonlinear dynamics (SINDy) algorithm to prune down the 
ansatzes and find the corresponding coefficients. 

# Prerequisites

**hyint** requires **numpy**/**scipy** and **sympy** to have be installed.

# Installation

Install **hyint** as

```python
pip install hyint
```

# Tutorial

The main function exported by **hyint** is `integrate(eq, x)`. It accepts two arguments, where 
`eq` is a univariable expression in `x'. It results either the integral or 0 otherwise. 

Some examples:

```python
from sympy import *
import hyint

x = Symbol('x')

In: hyint.integrate(x**3 - x + 1, x)
Out: x**4/4 - x**2/2 + x

In: hyint.integrate(x**2 * sin(2*x), x)
Out: -x**2*cos(2*x)/2 + x*sin(2*x)/2 + cos(2*x)/4

In: hyint.integrate(sqrt(x**2 + x - 1), x)
Out: x*sqrt(x**2 + x - 1)/2 + sqrt(x**2 + x - 1)/4 - 5*log(2*x + 2*sqrt(x**2 + x - 1) + 1)/8

In: hyint.integrate(x/(x**2 + 4), x)
Out: log(x**2 + 4)/2

In: hyint.integrate(x**2*log(x)**2 , x)
Out: x**3*log(x)**2/3 - 2*x**3*log(x)/9 + 0.0740740740740739*x**3

In: hyint.integrate(1 / (x**3 - 2*x + 1), x)
Out: 2.17082039324994*log(x - 1) + 1.34164078649988*log(x + 1/2 + sqrt(5)/2) - 1.17082039324993*log(x**3 - 2*x + 1)

In: hyint.integrate(x*exp(x)*cos(2*x), x)
Out: 2*x*exp(x)*sin(2*x)/5 + x*exp(x)*cos(2*x)/5 - 0.16*exp(x)*sin(2*x) + 0.12*exp(x)*cos(2*x)

In: hyint.integrate(log(log(x))/x, x)
Out: log(x)*log(log(x)) - log(x)

In: hyint.integrate(log(cos(x))*tan(x), x)
Out: -log(cos(x))**2/2

In: hyint.integrate(exp(x + 1)/(x + 1), x)
Out: Ei(x + 1)

In: hyint.integrate(exp(x)/x - exp(x)/x**2 , x)
Out: exp(x)/x

In: hyint.integrate(exp(x**2) , x)
Out: 0.886226925452758*erfi(x)

# sympy.integrate does not solve this example:
In: hyint.integrate(sqrt(1 - sin(x)), x)
Out: 2*cos(x)/sqrt(1 - sin(x))

```

# Citation

**hyint** is a adopted from and is a rewrite of **SymbolicNumericIntegration.jl**. 
Citation: [Symbolic-Numeric Integration of Univariate Expressions based on Sparse Regression](https://arxiv.org/abs/2201.12468):

```
@article{Iravanian2022,
author = {Shahriar Iravanian and Carl Julius Martensen and Alessandro Cheli and Shashi Gowda and Anand Jain and Julia Computing and Yingbo Ma and Chris Rackauckas},
doi = {10.48550/arxiv.2201.12468},
month = {1},
title = {Symbolic-Numeric Integration of Univariate Expressions based on Sparse Regression},
url = {https://arxiv.org/abs/2201.12468v2},
year = {2022},
}
```
