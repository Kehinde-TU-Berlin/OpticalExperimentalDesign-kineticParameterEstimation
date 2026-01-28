using LinearAlgebra
using Plots

# Step 1: Generate synthetic data
A0 = 1.0                   # initial concentration
k_true = 0.5               # true rate constant
t_data = [0, 1, 2, 3, 4, 5]  # experimental time points
A_data = A0 .* exp.(-k_true .* t_data) .+ 0.02 .* randn(length(t_data))  # add noise

# Step 2: Define the model function
model(t, k) = A0 .* exp.(-k .* t)

# Step 3: Define residuals and least squares cost function
function residuals(k)
    return A_data .- model.(t_data, k)
end

function cost(k)
    return sum(residuals(k).^2)
end

# Step 4: Perform simple least squares minimization (grid search)
k_values = range(0.1, 1.0, length=100)
cost_values = [cost(k) for k in k_values]
k_est = k_values[argmin(cost_values)]

println("Estimated k = ", k_est)
println("True k = ", k_true)

# Step 5: Plot results
A_fit = model.(t_data, k_est)
scatter(t_data, A_data, label="Data", xlabel="Time", ylabel="[A]")
plot!(t_data, A_fit, label="Fitted Model", lw=2, color=:red)