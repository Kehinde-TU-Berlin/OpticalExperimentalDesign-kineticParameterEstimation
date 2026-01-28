Pkg.add("StatsBase")

using QuasiMonteCarlo
using StatsBase
using Distributions
using LinearAlgebra
using Statistics

lower_θ = [0.5, 5.0]
upper_θ = [1.5, 15.0]
n_mc = 500
#θ_qmc = sample(n_mc, lower_θ, upper_θ, SobolSample())
θ_qmc = QuasiMonteCarlo.sample(n_mc, lower_θ, upper_θ, SobolSample())

# 2. Define the physics/system model (Crucial Fix)
function model(θ, ξ)
    # Replace this with your actual physical equation
    return θ[1] * log(ξ[1]) + θ[2] * ξ[2]
end

# 3. Define the information metric
function info_from_y(y_samples)
    # Variance of predicted outputs is a common proxy for information gain
    return var(y_samples)
end

# 4. Define Utility
function utility(ξ)
    # NOTE: QMC returns a matrix where columns are samples, so use eachcol
    y_qmc = [model(θ, ξ) + rand(Normal(0, 0.05)) for θ in eachcol(θ_qmc)]
    return info_from_y(y_qmc)
end

#function utility(ξ)
#    y_qmc = [model(θ, ξ) + rand(Normal(0,0.05)) for θ in eachrow(θ_qmc)]
#    return mean(info_from_y(y_qmc))  # Placeholder for info metric
#end

candidates = [[rand(200:500), rand(0.1:0.01:1.0)] for _ in 1:20]
utilities = utility.(candidates)
optimal_ξ = candidates[argmax(utilities)]
println("Optimal QMC-based point: ", optimal_ξ)