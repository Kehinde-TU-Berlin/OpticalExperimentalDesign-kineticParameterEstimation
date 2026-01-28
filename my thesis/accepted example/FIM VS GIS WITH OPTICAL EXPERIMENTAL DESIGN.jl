using LinearAlgebra
using QuasiMonteCarlo
using Combinatorics
using Statistics

# Model
model(t, θ) = θ[1] * exp(-θ[2] * t)

# Analytical Jacobian
function jacobian(t, θ)
    A, k = θ
    return [
        exp(-k*t)
        -A*t*exp(-k*t)
    ]
end


# PART I — Fisher Information Matrix (Local Design)
# 1. Fisher Information Matrix

function FIM(times, θ; σ2 = 1.0)
    F = zeros(2,2)
    for t in times
        J = jacobian(t, θ)
        F += (1/σ2) * (J * J')
    end
    return F
end

# 2. D-optimal design using FIM
# Candidate time grid

t_grid = 0.0:0.5:5.0
ns = 3  # number of measurements
θ_nominal = [1.0, 0.5]

# Objective function

function fim_objective(indices)
    times = collect(t_grid)[indices]
    F = FIM(times, θ_nominal)
    return det(F + 1e-8I)
end

# Optimization (enumeration)

combos = collect(combinations(1:length(t_grid), ns))
values_fim = [fim_objective(c) for c in combos]

best_fim_idx = argmax(values_fim)
optimal_times_fim = collect(t_grid)[combos[best_fim_idx]]

println("FIM-optimal times:")
println(optimal_times_fim)


# Interpretation
# Optimizes local parameter precision
# Sensitive to incorrect nominal parameters


# PART II — Global Information Matrix (Bayesian Design)
# 3. Global Information Matrix

function GIM(times, θ_samples; σ2 = 1.0)
    G = zeros(2,2)
    for θ in eachcol(θ_samples)
        G += FIM(times, θ, σ2=σ2)
    end
    return G / size(θ_samples, 2)
end

# 4. Parameter uncertainty (prior)

θ_lower = [0.8, 0.3]
θ_upper = [1.2, 0.7]

n_mc = 300
θ_samples = QuasiMonteCarlo.sample(
    n_mc, θ_lower, θ_upper, SobolSample()
)

# 5. D-optimal design using GIM
# Objective function

function gim_objective(indices)
    times = collect(t_grid)[indices]
    G = GIM(times, θ_samples)
    return det(G + 1e-8I)
end

# Optimization

values_gim = [gim_objective(c) for c in combos]

best_gim_idx = argmax(values_gim)
optimal_times_gim = collect(t_grid)[combos[best_gim_idx]]

println("\nGIM-optimal times:")
println(optimal_times_gim)

# PART III — Comparison
# Numerical comparison

println("\nDeterminants:")
println("FIM design det = ", values_fim[best_fim_idx])
println("GIM design det = ", values_gim[best_gim_idx])



# Practical guidance

# Use FIM-based OED when:

  # Reliable nominal parameters exist

  # Model is weakly nonlinear

# Use GIM-based OED when:

  # Parameters are uncertain

  # Noise is correlated or misspecified

  # Robust design is required