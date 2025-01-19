# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:22:09 2024

@author: s2714589
"""

import pytest
import numpy as np
from scipy.optimize import minimize

def func(x):
    return x**2 + 2*x + 1

def test_optimisation_minimize():
    # Use scipy.optimize.minimize to find the minimum of the quadratic function
    result = minimize(func, 0)  # Start at x = 0
    assert result.success, "Optimisation did not succeed"  
    assert np.isclose(result.x, -1), f"Expected -1, but got {result.x}" 
    
from scipy.linalg import solve

def test_linear_algebra_solve():
    # Define a system of linear equations: Ax = b
    A = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    
    # The expected solution is x = [1, 0]
    expected_solution = np.array([1, 0])
    
    # Solve the system
    solution = solve(A, b)
    
    # Assert the solution is correct
    np.testing.assert_almost_equal(solution, expected_solution, decimal=5)
    
from scipy import stats

def test_statistical_ttest_ind():
    # Create two random samples
    sample1 = np.random.normal(loc=0, scale=1, size=100)
    sample2 = np.random.normal(loc=0, scale=1, size=100)
    
    # Perform the two-sample t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    
    # Assert the p-value is above 0.05 (no significant difference between the samples)
    assert p_value > 0.05, f"P-value {p_value} is not greater than 0.05"
    
from scipy.integrate import quad

def test_integration_quad():
    # Define a simple function to integrate: f(x) = x^2
    def f(x):
        return x**2
    
    # Perform the integration of f(x) from 0 to 1
    result, error = quad(f, 0, 1)
    
    # The integral of x^2 from 0 to 1 is 1/3
    expected_result = 1/3
    assert np.isclose(result, expected_result, atol=1e-6), f"Expected {expected_result}, but got {result}"
    


