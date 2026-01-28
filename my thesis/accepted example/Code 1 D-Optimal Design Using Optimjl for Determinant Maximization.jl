using Pkg
Pkg.add("Optim")
Pkg.add("LinearAlgebra")

using Optim
using LinearAlgebra

function model(θ, ξ)
    return θ[1] * exp(-θ[2] / ξ[1]) * ξ[2]  # Simplified kinetic model
end

function fim(θ, ξ)
    # Numerical Jacobian approximation
    ε = 1e-6
    J = zeros(1, 2)  # For single output
    J[1] = (model(θ + [ε, 0], ξ) - model(θ, ξ)) / ε
    J[2] = (model(θ + [0, ε], ξ) - model(θ, ξ)) / ε
    return J' * J
end

θ_nom = [1.0, 10.0]  # Nominal parameters
function obj(ξ_vec)
    n_points = length(ξ_vec) ÷ 2  # Assuming 2D ξ
    ξ = reshape(ξ_vec, 2, n_points)'
    F = sum(fim(θ_nom, ξ[i,:]) for i in 1:n_points)
    return -log(det(F + 1e-6 * I))  # Regularized for stability
end

lower = [200.0, 0.1]
upper = [500.0, 1.0]  # Bounds for temp, conc
n_points = 2
lower_vec = repeat(lower, n_points)
upper_vec = repeat(upper, n_points)
initial_ξ = rand(2 * n_points) .* (upper_vec .- lower_vec) .+ lower_vec
# initial_ξ = rand(2) .* (upper .- lower) .+ lower  # For 2 points
# initial_ξ = rand(4) .* repeat(upper .- lower, 2) .+ repeat(lower, 2)
#result = optimize(obj, lower .* ones(4), upper .* ones(4), initial_ξ, Fminbox(LBFGS()))
result = optimize(
    obj,
    lower_vec,
    upper_vec,
    initial_ξ,
    Fminbox(LBFGS())
)

optimal_ξ = reshape(Optim.minimizer(result), 2, 2)'
println("Optimal points: ", optimal_ξ)