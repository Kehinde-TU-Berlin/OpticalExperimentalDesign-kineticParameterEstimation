# Weighted Least Squares Example

# Sample data
x = [1, 2, 3, 4, 5]
y = [2.2, 2.8, 4.5, 3.7, 5.5]

# Weights (higher weight = more important / more reliable data)
w = [0, 2, 1, 0.5, 10]

# Construct design matrix for model y = a + b*x
X = [ones(length(x)) x]

using LinearAlgebra      # for Diagonal()
using Plots              # for visualization

# Construct diagonal weight matrix
W = Diagonal(w)

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