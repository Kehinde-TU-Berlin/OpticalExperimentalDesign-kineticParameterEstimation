using Pkg

Pkg.add("LatinHypercubeSampling")
Pkg.add("Distributions")

using LatinHypercubeSampling
using Statistics
using Distributions

#bounds = [200.0 500.0; 0.1 1.0]
bounds = [(200.0, 500.0), (0.1, 1.0)]
# lower = bounds[:, 1]
# upper = bounds[:, 2]
n_runs = 5  # Minimal experiments
plan, _ = LHCoptim(n_runs, 2, 1000)  # 2D space, 1000 iterations
ξ_opt = scaleLHC(plan, bounds)
#ξ_opt = scaleLHC(plan, lower, upper)

function sensitivity(ξ)
    # Proxy: variance of model outputs over parameter prior
    θ_samples = rand(Uniform(0.5, 1.5), 100, 2)  # Prior samples
    outputs = [model(θ, ξ) for θ in eachrow(θ_samples)]
    return var(outputs)
end

info_gains = [sensitivity(ξ_opt[i,:]) for i in 1:n_runs]
optimal_indices = sortperm(info_gains, rev=true)[1:3]  # Select top 3
println("Selected optimal points: ", ξ_opt[optimal_indices,:])