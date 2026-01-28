# Problem: Estimate the Rate Constant k for a First-Order Reaction
# We have:

#   

# 1. Generate noisy experimental data.
# 2. Estimate 𝑘 using nonlinear least squares.
# 3. Use QuasiMonteCarlo to test different experimental designs (sets of time points).
# 4. Evaluate which design gives the smallest parameter uncertainty.

using Pkg
Pkg.add("Plots")


using NLsolve
using Optim
using QuasiMonteCarlo
using Distributions
using Statistics
using Plots

# Step 1: True model and parameters
A0 = 1.0
k_true = 0.5

# Define kinetic model
model(t, k) = A0 .* exp.(-k .* t)

# Step 2: Generate synthetic noisy data
t_exp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]          # initial experimental design
noise = Normal(0, 0.02)
A_exp = model.(t_exp, k_true) .+ rand(noise, length(t_exp))  # noisy observations

# Step 3: Define the nonlinear least squares cost function
function cost(k)
    y_pred = model.(t_exp, k[1])
    residuals = A_exp .- y_pred
    return sum(residuals .^ 2)
end

# Step 4: Estimate k using Optim
result = optimize(cost, [0.3], NelderMead())
k_est = result.minimizer[1]

println("True k = ", k_true)
println("Estimated k = ", k_est)

# Step 5: Visualize fit
A_fit = model.(t_exp, k_est)
scatter(t_exp, A_exp, label="Data", xlabel="Time", ylabel="[A]", legend=:topright)
plot!(t_exp, A_fit, label="Nonlinear LS Fit", lw=2, color=:red)

# ------------------------------------------------------------
# Step 6: Simple Optimal Experimental Design (OED)
# ------------------------------------------------------------
# Goal: find time points that minimize uncertainty in k estimate

# Fisher Information Matrix for a single parameter (k)
#function fisher_information(t_points, k)
    # derivative of model wrt k: dA/dk = -t*A0*exp(-k*t)
  #  sensitivities = [-t * A0 * exp(-k * t) for t in t_points]
   # return sum(sensitivities .^ 2)
#end

# Generate random experimental designs using quasi-Monte Carlo sampling
#sampler = SobolSample()
#designs = sample(sampler, 50, 3) .* 5  # 50 candidate designs with 3 time points in [0,5]
#designs = QuasiMonteCarlo.sample(50, [0.0, 0.1, 0.2], [3.0, 4.0, 5.0], sampler)


# Evaluate Fisher information for each design
# F_values = [fisher_information(design, k_true) for design in eachrow(designs)]

# # Pick best design (maximizes Fisher Information)
# best_idx = argmax(F_values)
# best_design = designs[best_idx, :]

# println("\nOptimal time points (OED): ", best_design)
# println("Fisher Information = ", F_values[best_idx])

# # Plot comparison
# histogram(F_values, bins=15, label="Fisher Info")
# vline!([F_values[best_idx]], label="Best Design", color=:red)