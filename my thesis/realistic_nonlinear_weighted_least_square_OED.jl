# below is a compact, runnable Julia example that ties together weighted nonlinear least squares for kinetic parameter estimation with:

# data generation using Distributions,
# parameter estimation by both Optim (minimization) and NLsolve (solve gradient = 0),
# sampling candidate designs with QuasiMonteCarlo (Sobol),
# simple OED selection using the Fisher Information Matrix,
# and utilities from Statistics / LinearAlgebra.

# 


using NLsolve
using Optim
using QuasiMonteCarlo
using Distributions
using Statistics
using LinearAlgebra
using Plots
using Random

# -------------------------
# 1) True system + generate noisy data
# -------------------------
A0_true = 1.2
k_true  = 0.6

model(t, p) = p[1] .* exp.(-p[2] .* t)   # p = [A0, k]

# Experimental times (initial design)
t_exp = [0.0, 0.5, 1.0, 2.0, 4.0]                  # chosen times
σ = [0.01, 0.01, 0.02, 0.03, 0.05]                 # assumed std deviations
w = 1.0 ./ (σ .^ 2)                                # weights = 1/variance

rng = MersenneTwister(1234)
noise = Normal(0, 1)
y_clean = model.(t_exp, Ref([A0_true, k_true])) |> vec
y_obs = y_clean .+ σ .* rand(rng, noise, length(t_exp))  # heteroscedastic noise

# -------------------------
# 2) Weighted nonlinear LS: cost and gradient
# -------------------------
# cost(p) = sum_i w_i * (y_obs_i - model(t_i,p))^2
function cost(p)
    ypred = model.(t_exp, Ref(p)) |> vec
    r = y_obs .- ypred
    return sum(w .* (r .^ 2))
end

# gradient of cost (2 components) — used by NLsolve (no factor 2 needed)
function grad!(g, p)
    ypred = model.(t_exp, Ref(p)) |> vec
    r = y_obs .- ypred
    A0, k = p[1], p[2]
    # ∂J/∂A0 = -2 * sum w_i * r_i * exp(-k t_i)
    # ∂J/∂k  =  2 * sum w_i * r_i * (A0 * t_i * exp(-k t_i))
    # we can omit the factor 2 (roots are same); here we include full gradient
    g1 = -2 * sum(w .* r .* exp.(-k .* t_exp))
    g2 =  2 * sum(w .* r .* (A0 .* t_exp .* exp.(-k .* t_exp)))
    g[1] = g1
    g[2] = g2
end

# -------------------------
# 3) Estimate with Optim (minimization)
# -------------------------
p0 = [1.0, 0.4]   # initial guess
opt_res = optimize(cost, p0, NelderMead(); iterations=1000)
opt_res = optimize(cost, p0, NelderMead(), Optim.Options(iterations = 1000))

p_opt = Optim.minimizer(opt_res)
println("Optim estimate: A0 = $(round(p_opt[1],digits=4)), k = $(round(p_opt[2],digits=4))")

# -------------------------
# 4) Estimate with NLsolve (solve gradient=0)
# -------------------------
function f!(F, p)     # NLsolve expects F(p) = 0
    grad!(F, p)
end

nl_res = nlsolve(f!, [1.0, 0.4]; ftol=1e-8, xtol=1e-8)
p_nls = nl_res.zero
println("NLsolve estimate: A0 = $(round(p_nls[1],digits=4)), k = $(round(p_nls[2],digits=4))")

# -------------------------
# 5) Plot data & fits
# -------------------------
tgrid = range(0, stop=5, length=200)
plot(tgrid, model.(tgrid, Ref([A0_true,k_true])) |> vec, label="True", lw=2)
scatter!(t_exp, y_obs, label="Observations", ms=4)
plot!(tgrid, model.(tgrid, Ref(p_opt)) |> vec, label="Optim fit", lw=2, ls=:dash)
plot!(tgrid, model.(tgrid, Ref(p_nls)) |> vec, label="NLsolve fit", lw=2, ls=:dot)
xlabel!("Time")
ylabel!("[A]")
title!("Weighted Nonlinear Least Squares Fit")


# -------------------------
# 6) Simple OED via Quasi-Monte Carlo (Sobol) sampling
#    - sample candidate designs (each design has m time points)
#    - compute Fisher Information Matrix at true params:
#        F = S' * W * S   where S_ij = ∂y(t_i)/∂p_j
#    - use D-optimality: maximize det(F)
# -------------------------
using Surrogates

m = 3                       # number of measurements per candidate design
Ndesigns = 200              # number of candidate designs to evaluate
sampler = SobolSample()     # or SobolSample(NoRand()) for deterministic

# get samples in [0,5]^m — call sample with bounds
try
    candidates = sample(sampler, Ndesigns, m, 0.0, 5.0)  # Ndesigns x m
catch
    # fallback for older API returning [0,1]^m
    candidates = sample(sampler, Ndesigns, m) .* 5.0
end

# sort each row (time points increasing)
candidates = [sort(vec(candidates[i, :]))' for i in 1:size(candidates,1)]
candidates = vcat(candidates...)    