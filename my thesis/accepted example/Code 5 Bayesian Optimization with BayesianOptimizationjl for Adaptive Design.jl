using Pkg

Pkg.add("BayesianOptimization")

using BayesianOptimization
using GaussianProcesses

function obj(ξ)
    return det(fim(θ_nom, ξ))  # Maximize det(FIM)
end

model = ElasticGPE(2, mean=MeanConst(0.), kernel=SE(0.,0.), logNoise=-Inf)
gp_surrogate = ElasticGPE(2, mean=MeanConst(0.), kernel=SE(0.,0.), logNoise=-Inf)
# opt = BOpt(obj, gp_surrogate, UpperConfidenceBound(), maxeval=10)

# 1. Define the search space bounds
lower = [200.0, 0.1]
upper = [500.0, 1.0]

# 2. Define the GP hyperparameter optimizer (MAP is standard)
model_optimizer = MAPOptimize() 

# 3. Corrected BOpt construction
opt = BOpt(obj, gp_surrogate, UpperConfidenceBound(), model_optimizer, lower, upper; iterations = 10, initializer_iterations = 5) # Random samples to seed the GP

result = boptimize!(opt)
optimal_ξ = result.observed_optimizer
println("Bayesian optimal points: ", optimal_ξ)