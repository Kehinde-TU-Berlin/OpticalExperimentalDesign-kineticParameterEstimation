using QuasiMonteCarlo
using Printf
using Distributions
using Optim
using Plots

# Interval for time: [0, 5]
lb = [0.0]
ub = [5.0]
N = 6   # we want exactly 3 points

# The four samplers
samplers = [
    ("Random (Uniform)", RandomSample()),
    ("Latin Hypercube",  LatinHypercubeSample()),
    ("Halton",           HaltonSample()),
    ("Sobol",            SobolSample())
]

println("5 experimental time points (t_exp) generated with each method:\n")

for (name, sampler) in samplers
    # Generate the samples (returns a matrix of size 1 × N)
    t_samples = QuasiMonteCarlo.sample(N, lb, ub, sampler)
    
    # Flatten to a vector and sort for readability (common in experiments)
    t_exp = sort(vec(t_samples))
    
    # Print nicely
    print(name, ": ")
    print("[")
    for (i, t) in enumerate(t_exp)
        @printf("%.3f", t)
        i < length(t_exp) && print(", ")
    end
    println("]")
end


# 3 Data generated

Random (Uniform): [0.164, 0.980, 2.162]
Latin Hypercube: [0.833, 2.500, 4.167]
Halton: [0.833, 2.083, 3.333]
Sobol: [1.875, 3.125, 4.375]

# 4 Data generated

Random (Uniform): [1.039, 1.316, 2.024, 4.716]
Latin Hypercube: [0.625, 1.875, 3.125, 4.375]
Halton: [0.625, 1.875, 3.125, 4.375]
Sobol: [0.625, 1.875, 3.125, 4.375]

# 5 Data generated

Random (Uniform): [0.131, 0.646, 2.031, 2.749, 4.719]
Latin Hypercube: [0.500, 1.500, 2.500, 3.500, 4.500]
Halton: [0.500, 1.125, 1.750, 3.000, 4.250]
Sobol: [0.625, 0.938, 1.875, 3.125, 4.375]

# 6 Data generated

Random (Uniform): [0.596, 2.141, 2.338, 3.376, 3.898, 4.110]
Latin Hypercube: [0.417, 1.250, 2.083, 2.917, 3.750, 4.583]
Halton: [0.417, 1.042, 1.667, 2.917, 3.542, 4.167]
Sobol: [0.625, 0.938, 1.875, 3.125, 3.438, 4.375]


# Step 1: True model and parameters
A0 = 1.0
k_true = 0.5

# Define kinetic model
model(t, k) = A0 .* exp.(-k .* t)

# Step 2: Generate synthetic noisy data
# 1. t_exp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]          # initial experimental design
t_exp = [0.625, 0.938, 1.875, 3.125, 3.438, 4.375]
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