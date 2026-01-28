# file: cstr_estimators.jl
using Random
using LinearAlgebra
using Statistics
using Distributions
using NLsolve
using Optim
using QuasiMonteCarlo

# ---------------------------
# 1) Problem setup / helpers
# ---------------------------

# True parameters
k_true = 0.5            # 1/s
C0_true = 2.0           # mol/L (or mol/m^3 scaled)
rng = MersenneTwister(42)

# Model: C(t) = C0 * exp(-k * t)
c_model(t, θ) = θ[1] * exp.(-θ[2] .* t)   # θ = [C0, k]

# Log-linearized model (for LS on ln(y)):
log_model(t, β) = β[1] .+ β[2] .* (-t)     # β = [ln C0, k]; note slope = -k

# Generate synthetic sampling times (design) using Sobol (Quasi Monte Carlo)
# function design_times_sobol(n; tmin=0.0, tmax=10.0, rng=GLOBAL_RNG)
#     s = sobolseq(1)                       # Sobol sequence generator
#     xs = QuasiMonteCarlo.sample(s, n)    # returns n × 1 matrix-like
#     ts = tmin .+ (tmax - tmin) .* xs[:,1] # map to [tmin, tmax]
#     return collect(ts)
# end

function design_times_sobol(n; tmin=0.0, tmax=10.0, rng=GLOBAL_RNG)
    s = SobolSeq(1)  # Correct constructor for Sobol sequence
    xs = QuasiMonteCarlo.sample(n, s)  # Correct sampling syntax
    ts = tmin .+ (tmax - tmin) .* xs  # xs is already a vector, no need for indexing
    return ts  # No need for collect() since it's already an array
end

# Simulate noisy measurements
# Two noise models for demonstration:
#  - multiplicative noise: y_obs = y_true * (1 + ε), ε ~ N(0, σ_rel)
#  - additive noise: y_obs = y_true + ε, ε ~ N(0, σ_abs)
function simulate_data(t, θ; noise=:multiplicative, σ_rel=0.05, σ_abs=0.05, rng=rng)
    y_true = c_model(t, θ)
    if noise == :multiplicative
        eps = rand(rng, Normal(0, σ_rel), length(t))
        y_obs = y_true .* (1 .+ eps)
        # approximate var(y_obs) = (σ_rel * y_true)^2
        var_y = (σ_rel .* y_true).^2
    elseif noise == :additive
        eps = rand(rng, Normal(0, σ_abs), length(t))
        y_obs = y_true .+ eps
        var_y = fill(σ_abs^2, length(t))
    else
        error("Unknown noise type")
    end
    return y_obs, y_true, var_y
end

# ---------------------------
# 2) Create synthetic dataset
# ---------------------------
const GLOBAL_RNG = Random.default_rng()
n = 12
t = design_times_sobol(n; tmin=0.1, tmax=8.0) |> sort   # avoid t=0 for log
θ_true = [C0_true, k_true]

# simulate multiplicative noise (5% relative)
y_obs, y_true, var_y = simulate_data(t, θ_true, noise=:multiplicative, σ_rel=0.05, rng=rng)

println("times t = ", round.(t, digits=3))
println("observations y_obs = ", round.(y_obs, digits=3))

# ---------------------------
# 3) Linear LS via log-transform
# ---------------------------
# transform: z_i = ln(y_i) = ln(C0) - k t_i + ε_i
z = log.(y_obs)                       # observed log-values
X = [ones(n) -t]                      # design matrix: columns = [1, -t]

# Ordinary least squares solution for beta = [ln C0, k]
β_hat = X \ z
lnC0_hat, k_hat_from_log = β_hat[1], β_hat[2]
C0_hat_from_log = exp(lnC0_hat)

println("\n--- Linear LS (log transform) ---")
println("ln(C0) estimate = $(round(lnC0_hat, digits=6)) , C0 = $(round(C0_hat_from_log, digits=6))")
println("k estimate = $(round(k_hat_from_log, digits=6))")

# compute fitted values & residuals (linearized)
z_fit = X * β_hat
resid_z = z - z_fit
σhat_z = sqrt(sum(resid_z.^2)/(n - length(β_hat)))   # residual std in log space
println("residual std (log-space) = $(round(σhat_z, digits=6))")

# ---------------------------
# 4) Weighted Least Squares (WLS) on original (nonlinear) model
# ---------------------------
# Objective: minimize S(θ) = sum_i w_i * (y_i - f(t_i; θ))^2
# where w_i = 1/Var(y_i). For multiplicative noise, Var(y_i) ≈ (σ_rel*y_true)^2.
w = 1.0 ./ var_y                     # weights (higher weight = more precise)

function wls_objective_vec(θ)
    # residual vector r_i = sqrt(w_i) * (y_obs - model)
    r = sqrt.(w) .* (y_obs .- c_model(t, θ))
    return r
end

# Solve normal equations (i.e., gradient = 0) by root finding using NLsolve
# Define gradient: g(θ) = -2 * J^T * W * (y - f)
# where J is Jacobian of f wrt θ: J[i,1] = ∂f/∂C0 = exp(-k t_i)
#                         J[i,2] = ∂f/∂k  = -C0 * t_i * exp(-k t_i)
function gradient_wls(θ)
    C0, k = θ[1], θ[2]
    f = c_model(t, θ)
    # Jacobian J: n x 2
    J = zeros(n, 2)
    J[:,1] .= exp.(-k .* t)                  # ∂f/∂C0
    J[:,2] .= -C0 .* t .* exp.(-k .* t)      # ∂f/∂k
    r = y_obs .- f
    # gradient (2-vector)
    grad = -2 .* (J' * (w .* r))
    return grad
end

# Use NLsolve to find θ such that gradient = 0
θ0 = [1.0, 0.2]   # initial guess
nlsol = nlsolve(x -> gradient_wls(x), θ0; method=:newton, autodiff=:forward)
θ_wls_nlsolve = nlsol.zero
println("\n--- WLS solved by NLsolve (root of gradient) ---")
println("θ (C0, k) = ", round.(θ_wls_nlsolve, digits=6))

# ---------------------------
# 5) WLS via direct minimization using Optim
# ---------------------------
obj(θ_vec) = sum((sqrt.(w) .* (y_obs .- c_model(t, θ_vec))).^2)
# We use box constraints to keep k positive
lower = [0.0, 1e-8]
upper = [Inf, 10.0]
res = optimize(obj, lower, upper, θ0, Fminbox(); optimizer = NelderMead())
θ_wls_opt = Optim.minimizer(res)
println("\n--- WLS via Optim (Fminbox + NelderMead) ---")
println("θ (C0, k) = ", round.(θ_wls_opt, digits=6))
println("objective value = ", round(Optim.minimum(res), digits=8))

# ---------------------------
# 6) Compare estimates and uncertainties (approx)
# ---------------------------
# Approx covariance from linearization: Cov(θ) ≈ (J' W J)^{-1} * σ^2
# Here σ^2 ~ 1 if residuals already incorporate weights; compute approximate
function covariance_approx(θ)
    C0, k = θ
    J = zeros(n, 2)
    J[:,1] .= exp.(-k .* t)
    J[:,2] .= -C0 .* t .* exp.(-k .* t)
    JTWJ = J' * (w .* J)
    cov = inv(JTWJ)
    # estimate scale: s2 = sum(w .* (y_obs .- c_model(t, θ)).^2) / (n - 2)
    s2 = sum((sqrt.(w) .* (y_obs .- c_model(t, θ))).^2) / (n - 2)
    return cov * s2
end

cov_nls = covariance_approx(θ_wls_nlsolve)
stderr = sqrt.(diag(cov_nls))
println("\nApprox std errors (NLsolve WLS): ", round.(stderr, digits=6))

# ---------------------------
# 7) Quick bootstrap using QuasiMonteCarlo sampling (sobol) for param uncertainty
#    (illustrative; small number of replicates)
# ---------------------------
m_boot = 200
sobol = SobolSample()  # from QuasiMonteCarlo
# We'll bootstrap by sampling noise realizations using Sobol + inverse CDF (normal),
# producing quasi-MonteCarlo noise vectors for speed & low-discrepancy coverage.
rng2 = MersenneTwister(123)
# Generate Sobol points in (0,1) of size m_boot × n
U = QuasiMonteCarlo.sample(sobol, m_boot,  n)  # m_boot × n matrix
boot_estimates = zeros(m_boot, 2)
σ_rel_assumed = 0.05
for b in 1:m_boot
    # transform U[b,:] to normal noise via quantile
    eps = quantile.(Ref(Normal(0,σ_rel_assumed)), U[b,:])
    y_boot = y_true .* (1 .+ eps)
    # simple LS on log to get initial guess
    zboot = log.(y_boot)
    βb = [ones(n) -t] \ zboot
    θ0b = [exp(βb[1]), βb[2]]
    # minimize WLS objective for this bootstrap sample (fast Nelder Mead)
    y_obs_backup = deepcopy(y_obs)
    # temporarily use y_boot for optimization
    y_obs .= y_boot
    resb = optimize(θ -> sum((sqrt.(w) .* (y_boot .- c_model(t, θ))).^2),
                   lower, upper, θ0b, Fminbox(); optimizer = NelderMead())
    θb = Optim.minimizer(resb)
    boot_estimates[b, :] .= θb
    # restore y_obs (not strictly needed because y_boot replaced in loop)
    y_obs .= y_obs_backup
end

mean_boot = mean(boot_estimates, dims=1)
std_boot = std(boot_estimates, dims=1)
println("\nBootstrap (Quasi-MC) mean estimates: ", round.(vec(mean_boot), digits=5))
println("Bootstrap std dev: ", round.(vec(std_boot), digits=5))

# ---------------------------
# 8) Summary printout
# ---------------------------
println("\n--- Summary ---")
println("True params: C0=$(C0_true), k=$(k_true)")
println("Linear (log) LS estimates: C0=$(round(C0_hat_from_log, digits=5)), k=$(round(k_hat_from_log, digits=5))")
println("WLS (NLsolve) estimates: C0=$(round(θ_wls_nlsolve[1], digits=5)), k=$(round(θ_wls_nlsolve[2], digits=5))")
println("WLS (Optim) estimates:  C0=$(round(θ_wls_opt[1], digits=5)), k=$(round(θ_wls_opt[2], digits=5))")
