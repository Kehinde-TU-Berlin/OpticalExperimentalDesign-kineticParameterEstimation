using Pkg

Pkg.add("MonteCarloMeasurements")

import Pkg
Pkg.add("MonteCarlo")

using MonteCarlo
using Distributions
using LinearAlgebra
using Statistics

θ_prior = MvNormal([1.0, 10.0], diagm([0.1, 1.0]))
noise_dist = Normal(0, 0.05)

#function info_gain(ξ, n_mc=1000)
#    θ_samples = rand(θ_prior, n_mc)
#    y_samples = [model(θ, ξ) + rand(noise_dist) for θ in eachcol(θ_samples)]
#    # Approximate KL divergence (simplified)
#    return mean(logpdf(θ_prior, θ_samples) - logpdf(posterior(θ_samples, y_samples), θ_samples))
#end

function info_gain_proxy(ξ, n_mc=1000)
    θ_samples = rand(θ_prior, n_mc)
    y_samples = [model(θ, ξ) for θ in eachcol(θ_samples)]
    return var(y_samples) # More variance = more potential information
end

# candidates = [rand(Uniform(200,500)), rand(Uniform(0.1,1))] for _ in 1:50
# This creates a Vector of Vectors
candidates = [[rand(Uniform(200, 500)), rand(Uniform(0.1, 1))] for _ in 1:50]
gains = [info_gain(c) for c in candidates]
optimal_ξ = candidates[argmax(gains)]
println("Optimal point: ", optimal_ξ)



############################################################################################

using Pkg
import Pkg

Pkg.add("Distributions")

using Distributions
using LinearAlgebra
using Statistics

# 1. Setup
θ_prior = MvNormal([1.0, 10.0], diagm([0.1, 1.0]))
noise_dist = Normal(0, 0.05)

# Example model: replace with your actual function
function model(θ, ξ)
    return θ[1] * ξ[1] + θ[2] * ξ[2] 
end

# 2. Revised Info Gain (Using Nested Monte Carlo Estimator)
function info_gain(ξ, n_outer=100, n_inner=100)
    # Sample 'true' parameters from prior
    θ_outer = rand(θ_prior, n_outer)
    
    eig = 0.0
    for i in 1:n_outer
        θ_true = θ_outer[:, i]
        # Simulate an observation
        y_obs = model(θ_true, ξ) + rand(noise_dist)
        
        # Likelihood of the observation given the true θ
        log_p_y_given_theta_true = logpdf(Normal(model(θ_true, ξ), 0.05), y_obs)
        
        # Evidence: Average likelihood of y_obs over the prior
        θ_inner = rand(θ_prior, n_inner)
        log_p_y_evidence = logsumexp([logpdf(Normal(model(θ_inner[:, j], ξ), 0.05), y_obs) for j in 1:n_inner]) - log(n_inner)
        
        eig += log_p_y_given_theta_true - log_p_y_evidence
    end
    
    return eig / n_outer
end

# Helper for stable log-averaging
function logsumexp(x)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

# 3. Execution
candidates = [[rand(Uniform(200, 500)), rand(Uniform(0.1, 1))] for _ in 1:50]
gains = [info_gain(c) for c in candidates]
optimal_ξ = candidates[argmax(gains)]
println("Optimal point: ", optimal_ξ)