# Newton's Method Example for Optimization

# Define the function and its derivatives
f(x) = (x - 3)^2 + 2        # function
f_prime(x) = 2 * (x - 3)     # first derivative
f_doubleprime(x) = 2         # second derivative

# Initial guess
x = 0.0

# Newton's method iterations
for i in 1:10
    x_new = x - f_prime(x) / f_doubleprime(x)
    println("Iteration $i: x = $x_new, f(x) = $(f(x_new))")
    if abs(x_new - x) < 1e-6
        break
    end
    x = x_new
end

println("\nApproximate minimum at x = ", x)