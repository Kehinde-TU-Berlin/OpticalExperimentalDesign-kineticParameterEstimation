using Pkg
Pkg.add("GlobalSensitivity")

using GlobalSensitivity
using QuasiMonteCarlo

param_ranges = [0.5 1.5; 5.0 15.0; 200.0 500.0; 0.1 1.0]  # θ + ξ
function wrapped_model(x)
    θ = x[1:2]; ξ = x[3:4]
    return model(θ, ξ)
end

res = gsa(wrapped_model, Sobol(), param_ranges, samples=1000)
sensitive_ξ = filter_ξ_by_sensitivity(res.S1[3:4])  # Custom filter for high indices
println("Sensitivity-optimal points: ", sensitive_ξ)


###############################################################################################

using GlobalSensitivity
using QuasiMonteCarlo
using Statistics

# 1. Define the underlying model function (if not already defined)
function model(θ, ξ)
    return θ[1] * log(ξ[1]) + θ[2] * ξ[2]
end

# 2. Corrected Parameter Ranges: [Lower Upper]
# Rows 1-2 are θ (parameters), Rows 3-4 are ξ (experimental settings)
param_ranges = [0.5 1.5;   # θ₁
                5.0 15.0;  # θ₂
                200.0 500.0; # ξ₁
                0.1 1.0]   # ξ₂

# 3. Wrapped Model: GlobalSensitivity passes a matrix if batching, 
# or a vector if not. Here we handle the vector input.
function wrapped_model(x)
    θ = x[1:2] 
    ξ = x[3:4]
    return model(θ, ξ)
end

# 4. Run Global Sensitivity Analysis (GSA)
# Sobol() calculates S1 (first-order) and ST (total-order) indices
res = gsa(wrapped_model, Sobol(), param_ranges, samples=1000)

# 5. Extract and Filter Sensitivity
# res.S1 contains the first-order indices for each input
# [3:4] corresponds to the settings ξ₁ and ξ₂
s1_indices = res.S1[3:4]

# Define the filter: e.g., keep indices where sensitivity is above a threshold
threshold = 0.1
sensitive_ξ_indices = findall(x -> x > threshold, s1_indices)

println("First-order indices for ξ: ", s1_indices)
println("Settings with high impact: ", sensitive_ξ_indices)