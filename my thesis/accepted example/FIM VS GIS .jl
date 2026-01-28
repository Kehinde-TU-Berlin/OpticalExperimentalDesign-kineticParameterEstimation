# Below is a clear Julia-based demonstration that differentiates the Fisher Information Matrix (FIM) and 
# the Global Information Matrix (GIM) using the same nonlinear model

# The goal is to make it obvious what each matrix measures and how they differ.

# 1. Fisher Information Matrix (FIM)
#Local, pointwise sensitivity



using LinearAlgebra

# Model
model(t, θ) = θ[1] * exp(-θ[2] * t)

# Jacobian (analytical)
function jacobian(t, θ)
    A, k = θ
    return [
        exp(-k*t)          # ∂y/∂A
        -A*t*exp(-k*t)     # ∂y/∂k
    ]
end

# Fisher Information Matrix
function FIM(times, θ; σ2 = 1.0)
    F = zeros(2,2)
    for t in times
        J = jacobian(t, θ)
        F += (1/σ2) * (J * J')
    end
    return F
end

# Nominal parameter values
θ_nominal = [1.0, 0.5]
times = [0.0, 1.0, 2.0, 3.0]

F_fim = FIM(times, θ_nominal)
println("Fisher Information Matrix:")
println(F_fim)





# 2. Global Information Matrix (GIM)

# Global, variance-based sensitivity

# Here we average the FIM over the entire parameter space, which approximates a global information measure.

using QuasiMonteCarlo
using Statistics

# Parameter bounds
θ_lower = [0.8, 0.3]
θ_upper = [1.2, 0.7]

# Sample parameter space
n_samples = 500
θ_samples = QuasiMonteCarlo.sample(
    n_samples, θ_lower, θ_upper, SobolSample()
)

# Global Information Matrix
function GIM(times, θ_samples; σ2 = 1.0)
    G = zeros(2,2)
    for θ in eachcol(θ_samples)
        G += FIM(times, θ, σ2=σ2)
    end
    return G / size(θ_samples, 2)
end

G_gim = GIM(times, θ_samples)
println("\nGlobal Information Matrix:")
println(G_gim)



# 3. Key numerical comparison

println("\nDeterminants:")
println("det(FIM) = ", det(F_fim))
println("det(GIM) = ", det(G_gim))



# 5. One-line mathematical summary

# FIM:
#F(θ) = Σ J(t,θ)'J(t,θ)

# GIM:
#G = Eθ[F(θ)]




# When to use which (practical rule)

# Use FIM when:

 # Parameters are well-known

 # Model is weakly nonlinear

 # Fast computation is needed

# Use GIM when:

 # Parameters are uncertain

 # Model is strongly nonlinear

 # Designing experiments robustly