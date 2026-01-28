using Pkg
Pkg.add("Combinatorics")

using QuasiMonteCarlo
using LinearAlgebra
using Combinatorics
using Statistics

# Step 2: Define the logistic model (proxy for kinetic reaction)
function model(t, θ)
    r, K, C0 = θ
    return K / (1 + (K / C0 - 1) * exp(-r * t))
end

# Step 3: Compute FIM at a single time t (numerical Jacobian)
function fim(t, θ)
    ε = 1e-6
    J = zeros(1, 3)  # Single output, 3 parameters
    for i in 1:3
        θ_plus = copy(θ)
        θ_plus[i] += ε
        J[i] = (model(t, θ_plus) - model(t, θ)) / ε
    end
    return J' * J  # Assuming σ^2 = 1 (scalar factor doesn't affect optimization)
end

# Step 4: Define grid and parameters
Δt = 2.0
t_start = 0.0
t_final = 20.0
k = floor(Int, (t_final - t_start) / Δt) + 1
times = t_start .+ (0:(k-1)) .* Δt
ns = 5  # Fixed number of experiments (adjust as needed)

# Step 5: QMC samples for θ prior
lower_θ = [0.14, 35.0, 3.15]
upper_θ = [0.26, 65.0, 5.85]
n_mc = 500  # Number of QMC samples
# θ_samples = sample(n_mc, lower_θ, upper_θ, SobolSample())
θ_samples = QuasiMonteCarlo.sample(n_mc, lower_θ, upper_θ, SobolSample())

# Step 6-7: Objective and optimization via enumeration
function objective(subset_indices)
    # Precompute FIM for all times and samples for efficiency
    FIM_total = zeros(3, 3, n_mc)
    for (s, θ) in enumerate(eachcol(θ_samples))
        for idx in subset_indices
            FIM_total[:, :, s] += fim(times[idx], θ)
        end
    end
    dets = [det(FIM_total[:, :, s] + 1e-6 * I(3)) for s in 1:n_mc]  # Regularized det
    return mean(dets)
end

# Enumerate all combinations and find max
all_combos = collect(combinations(1:k, ns))
objectives = [objective(combo) for combo in all_combos]
best_index = argmax(objectives)
best_subset = all_combos[best_index]
optimal_times = times[best_subset]
optimal_value = objectives[best_index]

println("Optimal observation times: ", optimal_times)
println("Maximized expected det(G): ", optimal_value)