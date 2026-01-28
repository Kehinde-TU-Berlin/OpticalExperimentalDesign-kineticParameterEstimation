using LinearAlgebra
using Plots

# Step 1: Generate synthetic data
A0 = 1.0
k_true = 0.5
t_data = [0, 1, 2, 3, 4, 5]
A_data = A0 .* exp.(-k_true .* t_data) .+ 0.02 .* randn(length(t_data))  # noisy data

# Step 2: Define the model
model(t, k) = A0 .* exp.(-k .* t)

# Step 3: Define weights (inverse of measurement variance)
# Example: later measurements are noisier → smaller weights
σ = [0.01, 0.01, 0.02, 0.03, 0.05, 0.05]    # assumed std deviation for each point
w = 1.0 ./ (σ .^ 2)                         # weight = 1/variance

# Step 4: Weighted least squares cost function
function cost(k)
    residuals = A_data .- model.(t_data, k)
    return sum(w .* residuals.^2)
end

# Step 5: Estimate parameter k (simple grid search)
k_values = range(0.1, 1.0, length=100)
cost_values = [cost(k) for k in k_values]
k_est = k_values[argmin(cost_values)]

println("True k = ", k_true)
println("Estimated k (Weighted LS) = ", k_est)

# Step 6: Plot data and fitted curve
A_fit = model.(t_data, k_est)
scatter(t_data, A_data, label="Data", xlabel="Time", ylabel="[A]", legend=:topright)
plot!(t_data, A_fit, label="Weighted Fit", lw=2, color=:red)