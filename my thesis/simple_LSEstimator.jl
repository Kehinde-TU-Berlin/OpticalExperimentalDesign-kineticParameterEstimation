# Simple Least Squares Example in Julia

# Sample data points
x = [1, 2, 3, 4, 5]
y = [2.2, 2.8, 4.5, 3.7, 5.5]

# Construct the design matrix for y = a + b*x
X = [ones(length(x)) x]

# Solve for coefficients (a, b) using the normal equation: (X'X)\(X'y)
β = X \ y

# Extract coefficients
a, b = β
println("Intercept (a): ", a)
println("Slope (b): ", b)

# Predict y values
y_pred = X * β

# Show fitted line
using Plots
scatter(x, y, label="Data Points")
plot!(x, y_pred, label="Fitted Line", lw=2, color=:red)