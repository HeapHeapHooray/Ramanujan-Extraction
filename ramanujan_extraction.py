import sympy as sp

def extract_ramanujan_constant(func_str: str, max_iterations: int = 10, verbose: bool = False):
    """
    Parses a divergent series generator function and extracts its Ramanujan structural constant.

    Args:
        func_str: A string representing the function f(x). Use 'x' as the variable.
        max_iterations: Safety limit for non-polynomial functions (which have infinite non-zero derivatives).
        verbose: If True, prints the execution trace of the extraction process.

    Returns:
        The exact fractional constant extracted from the series.
    """
    # Define our symbolic variable
    x = sp.Symbol('x')

    # Parse the string into a SymPy mathematical expression
    try:
        f = sp.sympify(func_str)
    except sp.SympifyError:
        raise ValueError(f"Could not parse the function string: '{func_str}'")

    print(f"--- Compiling Series for f(x) = {f} ---")
    first_6 = [f.subs(x, i) for i in range(1, 7)]
    print(f"First 6 elements of the series: {first_6}")

    # Step 1: Evaluate the Base State (-f(0) / 2)
    f_0 = f.subs(x, 0)
    constant = -f_0 / sp.Rational(2, 1)

    if verbose:
        print(f"Base state (-f(0)/2): {constant}")

    # Step 2: The Derivative Loop
    # We loop through k to find the (2k-1)th derivatives.
    for k in range(1, max_iterations + 1):
        derivative_order = 2 * k - 1

        # Calculate the symbolic derivative
        f_prime = sp.diff(f, x, derivative_order)

        # Optimize: If the derivative function becomes exactly 0,
        # all subsequent derivatives will be 0. We can safely break the loop.
        if f_prime == 0:
            if verbose:
                print(f"Index {k}: Derivative is 0. Halting extraction loop.")
            break

        # Evaluate the derivative at x = 0
        f_prime_0 = f_prime.subs(x, 0)

        # If the derivative at 0 is 0, this term adds nothing. Skip the heavy math.
        if f_prime_0 == 0:
            continue

        # Fetch the Bernoulli weight and the dampening factorial
        B_2k = sp.bernoulli(2 * k)
        fact = sp.factorial(2 * k)

        # Calculate the current term: (B_2k / (2k)!) * f^(2k-1)(0)
        term = (B_2k / fact) * f_prime_0

        # Subtract from the running constant
        constant -= term

        if verbose:
            print(f"Index {k} | {derivative_order}th Deriv @ 0: {f_prime_0} | B_{2*k}: {B_2k} | Term extracted: {-term}")

    if verbose:
        print(f"Final Structural Constant: {constant}\n")

    return constant

# ==========================================
# Execution / Test Cases
# ==========================================
if __name__ == "__main__":
    # Test Case 1: The Constant Series (1 + 1 + 1 + 1...)
    # Function: f(x) = 1
    # Expected: -1/2
    ans_const = extract_ramanujan_constant("1", verbose=True)

    # Test Case 2: The Linear Series (1 + 2 + 3 + 4...)
    # Function: f(x) = x
    # Expected: -1/12
    ans_linear = extract_ramanujan_constant("x", verbose=True)

    # Test Case 3: The Cubic Series (1^3 + 2^3 + 3^3 + 4^3...)
    # Function: f(x) = x**3
    # Expected: 1/120
    ans_cubic = extract_ramanujan_constant("x**3", verbose=True)

    # Test Case 4: The Quadratic Series (1^2 + 2^2 + 3^2 + 4^2...)
    # Function: f(x) = x**2
    # Expected: 0
    ans_quad = extract_ramanujan_constant("x**2", verbose=True)
