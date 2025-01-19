import pytest
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
from scipy import stats
from scipy.integrate import quad
import time

# Unit Test for Optimization Function: scipy.optimize.minimize
def func(x):
    return x**2 + 2*x + 1

def test_optimization_minimize():
    result = minimize(func, 0)  # Start at x = 0
    assert result.success, "Optimization did not succeed"
    assert np.isclose(result.x, -1), f"Expected -1, but got {result.x}"

# Unit Test for Linear Algebra Function: scipy.linalg.solve
def test_linear_algebra_solve():
    A = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    expected_solution = np.array([1, 0])
    solution = solve(A, b)
    np.testing.assert_almost_equal(solution, expected_solution, decimal=5)

# Unit Test for Statistical Function: scipy.stats.ttest_ind
def test_statistical_ttest_ind():
    sample1 = np.random.normal(loc=0, scale=1, size=100)
    sample2 = np.random.normal(loc=0, scale=1, size=100)
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    assert p_value > 0.05, f"P-value {p_value} is not greater than 0.05"

# Unit Test for Integration Function: scipy.integrate.quad
def test_integration_quad():
    def f(x):
        return x**2
    result, error = quad(f, 0, 1)
    expected_result = 1/3
    assert np.isclose(result, expected_result, atol=1e-6), f"Expected {expected_result}, but got {result}"

# Functional Test for Optimization: Testing with multiple starting points
def test_optimization_function():
    def func(x):
        return x**2 + 2*x + 1
    start_points = [0, 5, -5]
    for start in start_points:
        result = minimize(func, start)
        assert result.success, "Optimization did not succeed"
        assert np.isclose(result.x, -1), f"Expected -1, but got {result.x} for start point {start}"

# Functional Test for Linear Algebra: Testing solving different systems of linear equations
def test_linear_algebra_systems():
    # First system: 2x + y = 1, x + 3y = 3
    A1 = np.array([[2, 1], [1, 3]])
    b1 = np.array([1, 3])
    expected_solution1 = np.array([1, 0])
    
    # Second system: 3x - 2y = 4, x + y = 2
    A2 = np.array([[3, -2], [1, 1]])
    b2 = np.array([4, 2])
    expected_solution2 = np.array([2, 0])
    
    # Solve both systems
    solution1 = solve(A1, b1)
    solution2 = solve(A2, b2)
    
    # Assert that solutions are correct
    np.testing.assert_almost_equal(solution1, expected_solution1, decimal=5)
    np.testing.assert_almost_equal(solution2, expected_solution2, decimal=5)

# Functional Test for Integration: Testing a few different integrals
def test_integration_functions():
    # First function: f(x) = x^3
    def f1(x):
        return x**3
    result1, _ = quad(f1, 0, 1)
    expected1 = 1/4  # The integral of x^3 from 0 to 1 is 1/4
    assert np.isclose(result1, expected1, atol=1e-6), f"Expected {expected1}, but got {result1}"
    
    # Second function: f(x) = e^(-x^2)
    def f2(x):
        return np.exp(-x**2)
    result2, _ = quad(f2, -np.inf, np.inf)
    expected2 = np.sqrt(np.pi)  # The integral of e^(-x^2) over the real line is sqrt(pi)
    assert np.isclose(result2, expected2, atol=1e-6), f"Expected {expected2}, but got {result2}"

# Integration Test for SciPy's Integration Functions and External Libraries
def test_integration_with_numpy():
    # Ensure that NumPy and SciPy integrate well together
    # Solve a linear system using NumPy and SciPy together
    A = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    solution = solve(A, b)
    
    # Now using NumPy for array operations
    assert np.allclose(np.dot(A, solution), b), "SciPy and NumPy integration failed"

# Performance Test for Optimization Function
def test_performance_optimization():
    def func(x):
        return np.sum(x**2)
    
    # Create a large input for optimization
    large_input = np.random.rand(10000)  # A 10000-dimensional vector
    
    start_time = time.time()
    result = minimize(func, large_input)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Optimization performance test took {execution_time} seconds")
    
    # Assert that the optimization converges and the execution time is reasonable
    assert result.success, "Optimization did not succeed"
    assert execution_time < 1, f"Optimization took too long: {execution_time} seconds"

# Performance Test for Integration Function
def test_performance_integration():
    def f(x):
        return np.exp(-x**2)
    
    start_time = time.time()
    result, error = quad(f, -np.inf, np.inf)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Integration performance test took {execution_time} seconds")
    
    # Assert that the integration was successful and the execution time is reasonable
    assert np.isclose(result, np.sqrt(np.pi), atol=1e-6), f"Expected sqrt(pi), but got {result}"
    assert execution_time < 1, f"Integration took too long: {execution_time} seconds"

# Running the tests with pytest
if __name__ == "__main__":
    pytest.main()
