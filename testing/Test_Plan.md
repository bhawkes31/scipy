# Test Plan for SciPy

## Introduction

The aim of this document is to set out the testing plan for testing the Python package SciPy. This test plan will aim to test the main requirements of the SciPy package, as identified in the testing requirements document. It will also outline how the results of this testing will be communicated.

## Testing Scope

The main functions of the package that will be tested for accuracy and robustness will be the optimisation functions, integration evaluation functions, statistical packages, signal processing operations and linear algebra functions. These functions have been chosen as they are some of the most widely used and applicable functions from the SciPy library.

These functions will be tested using functional testing, unit testing, performance testing, integration testing and useability testing.

## Testing Strategy

### Functional Testing

Functional testing will be carried out to ensure that SciPy's functions produce accurate results for a range of different use cases. This will be done by using a variety of inputs into functions and checking to see that the results are as expected using both large and small inputs. Testing will be automated by comparing results to known or theoretical solutions.

### Unit Testing

Unit testing will be focused on individual functions to ensure their correctness. This will focus on normal inputs into these functions. Unit tests will also verify that each function's error handling and exceptions are raised appropriately when given invalid inputs.

### Performance Testing

Performance testing will focus on testing that functiosn complete their operations in a timely manner. This will be done by inputting large datasets into the functions and measuring their time to completion. This will then be compared to acceptable completion time metrics that a typical user would expect to ensure that SciPy has acceptable performance levels.  

### Integration Testing

This will test that SciPy both integrates well with other Python packages as well as integrates well with itself. This will be done by ensuring outputs from SciPy functions are useable in a wide variety of scenarios and by using workflows that invlove multiple SciPy and, for example, NumPy functionms to test SciPy's integration.

### Useability Testing

This will be a more manual test to ensure that SciPy is useable to inexperienced users of it. This will be done by going through some SciPy documentation to see if it is clear and understandable as well as looking at error messages in functions to test if it is clear what is incorrect.

## Communication of Results

The results of these tests will be communicated in a testing report. This report will outline the specifics of the testing carried out, the results of the test and, if applicable, any areas where the software could be improved.
