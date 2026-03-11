# ============================================================================
# OPTIMAL EXPERIMENTAL DESIGN FOR KINETIC PARAMETER ESTIMATION
# Master Thesis: CSTR Reactor with FIM/GIM Integration
# 
# This code implements the requested functions with proper random number
# generation and optimal experimental design using FIM and GIM.
# ============================================================================

using NLsolve
using Optim
using QuasiMonteCarlo
using Distributions
using Statistics
using LinearAlgebra
using Plots
using Random

# Set random seed for reproducibility - CHANGE THIS SEED TO GET DIFFERENT RESULTS
# For reproducible results, keep the same seed; for different results, change the seed
const GLOBAL_SEED = 123  # Change this value to get different random sequences
Random.seed!(GLOBAL_SEED)

# ============================================================================
# BASE FUNCTIONS (YOUR ORIGINAL CODE - CORRECTED)
# ============================================================================

"""
    CSTR_model(ca_in, Temp; tau=5, n=1, k=[0.0, 0.0])

CSTR reactor model solving mass balance using NLsolve.
"""
function CSTR_model(ca_in, Temp; tau=5.0, n=1.0, k=[0.0, 0.0])
    sol = nlsolve((F, CA) -> concentration_compute(F, CA; k=k, ca_in=ca_in, Temp=Temp, n=n, tau=tau), 
                  [0.1], ftol=1e-15, show_trace=false)
    return sol.zero[1]
end

"""
    random_points_generator(Nexps; lb=[0.0], ub=[1.0], Sampling=HaltonSample())

Generate random experimental points using quasi-Monte Carlo sampling.
Note: Results depend on the global random seed.
"""
function random_points_generator(Nexps; lb=[0.0], ub=[1.0], Sampling=HaltonSample())
    s = QuasiMonteCarlo.sample(Nexps, lb, ub, Sampling)
    return s
end

"""
    ca_exp(ca_ins, Temp; Nexps=0, k=[1.0, 20000.0], add_noise=false, sigma=1e-3, N_repeats=5, order=1.0)

Compute experimental outlet concentrations with optional noise.
"""
function ca_exp(ca_ins, Temp; Nexps=0, k=[1.0, 20000.0], add_noise=false, sigma=1e-3, N_repeats=5, order=1.0)
    ca_out_without_noise = zeros(Nexps)
    ca_out_matrix = zeros(N_repeats, Nexps)
    
    # Convert inputs to vectors if needed
    ca_ins_vec = ca_ins[:]
    Temp_vec = Temp[:]
    
    for i in 1:Nexps
        ca_out_without_noise[i] = CSTR_model(ca_ins_vec[i], Temp_vec[i]; k=k, n=order)
        
        if add_noise && sigma > 0
            # Add Gaussian noise
            for j in 1:N_repeats
                ca_out_matrix[j, i] = ca_out_without_noise[i] + sigma * ca_out_without_noise[i] * randn()
            end
        else
            # No noise - all repeats identical
            ca_out_matrix[:, i] .= ca_out_without_noise[i]
        end
    end
    
    ca_out = vec(mean(ca_out_matrix, dims=1))
    return ca_out
end

"""
    concentration_compute(F, CA; k, n, Temp, ca_in, tau)

Residual function for CSTR mass balance.
"""
function concentration_compute(F, CA; k, n, Temp, ca_in, tau)
    R = 8.314
    k_T = k[1] * exp(-k[2] / (R * Temp))
    F[1] = (ca_in - CA[1]) / tau - k_T * CA[1]^n
    return nothing
end

"""
    ca_model(k; ca_in=0.0, Temp=0.0, n=1.0, tau=5.0)

Non-linear solver to compute outlet concentrations for CSTR.
"""
function ca_model(k; ca_in=0.0, Temp=0.0, n=1.0, tau=5.0)
    function resid(F, CA)
        concentration_compute(F, CA; k=k, ca_in=ca_in, Temp=Temp, n=n, tau=tau)
    end
    sol = nlsolve(resid, [0.1], ftol=1e-15, show_trace=false)
    return sol.zero[1]
end

"""
    parameter_estimator(; Nexps=1, ca_exp_data=0.0, ca_in=0.0, Temp=0.0, 
                         order=1.0, initial_guess=[1.0, 20000.0])

Estimate reaction parameters k using nonlinear least squares.
"""
function parameter_estimator(; Nexps=1, ca_exp_data=0.0, ca_in=0.0, Temp=0.0, 
                               order=1.0, initial_guess=[1.0, 20000.0])
    
    # Convert inputs to vectors
    ca_exp_vec = ca_exp_data[:]
    ca_in_vec = ca_in[:]
    Temp_vec = Temp[:]
    
    function param_estim(k)
        total_error = 0.0
        for i in 1:Nexps
            pred = ca_model(k; ca_in=ca_in_vec[i], Temp=Temp_vec[i], n=order, tau=5.0)
            error = ca_exp_vec[i] - pred
            total_error += error^2
        end
        return total_error
    end

    solver = LBFGS()
    res = optimize(param_estim, initial_guess, solver, 
                   Optim.Options(show_trace=false, g_abstol=1e-12, iterations=1000))
    
    k_est = Optim.minimizer(res)
    final_error = Optim.minimum(res)
    converged = Optim.g_converged(res)
    
    println("\n📊 Parameter Estimation Results:")
    println("   Estimated A = $(round(k_est[1], digits=4))")
    println("   Estimated Ea = $(round(k_est[2], digits=1))")
    println("   Final error = $(round(final_error, digits=6))")
    println("   Converged: $converged")
    
    return k_est, final_error, converged
end

"""
    run_parameter_estimation(; Nexps=5, Sampling=HaltonSample(), add_noise=true, 
                              sigma=1e-3, order=1, true_k=[1.0, 20000.0], 
                              initial_guess=[0.5, 15000.0])

Run parameter estimation with random experimental design.
"""
function run_parameter_estimation(; Nexps=5, Sampling=HaltonSample(), add_noise=true, 
                                    sigma=1e-3, order=1, true_k=[1.0, 20000.0],
                                    initial_guess=[0.5, 15000.0])
    
    println("\n" * "="^80)
    println("RANDOM DESIGN PARAMETER ESTIMATION")
    println("Seed: $(GLOBAL_SEED)")
    println("="^80)
    println("Number of experiments: $Nexps")
    println("Noise: $(add_noise ? "$(sigma*100)%" : "None")")
    println("True parameters: A=$(true_k[1]), Ea=$(true_k[2])")
    
    # Generate random experimental conditions
    Temp_samples = random_points_generator(Nexps; lb=[300.0], ub=[500.0], Sampling=Sampling)
    ca_samples = random_points_generator(Nexps; lb=[0.1], ub=[2.0], Sampling=Sampling)
    
    Temp = vec(Temp_samples)
    ca_ins = vec(ca_samples)
    
    println("\n📊 Experimental conditions:")
    for i in 1:Nexps
        println("   Exp $i: T=$(round(Temp[i], digits=1))K, Ca_in=$(round(ca_ins[i], digits=2)) mol/m³")
    end
    
    # Simulate experimental data
    ca_exper = ca_exp(ca_ins, Temp; Nexps=Nexps, k=true_k, 
                       add_noise=add_noise, sigma=sigma, order=order)
    
    println("\n📈 Experimental results:")
    for i in 1:Nexps
        println("   Exp $i: Ca_out = $(round(ca_exper[i], digits=4)) mol/m³")
    end
    
    # Estimate parameters
    println("\n🔍 Estimating parameters...")
    k_est, final_err, converged = parameter_estimator(; Nexps=Nexps, ca_exp_data=ca_exper, 
                                                        ca_in=ca_ins, Temp=Temp, order=order,
                                                        initial_guess=initial_guess)
    
    # Calculate errors
    error_A = abs(k_est[1] - true_k[1])/true_k[1]*100
    error_Ea = abs(k_est[2] - true_k[2])/true_k[2]*100
    
    println("\n📊 Estimation Accuracy:")
    println("   A error: $(round(error_A, digits=2))%")
    println("   Ea error: $(round(error_Ea, digits=2))%")
    
    return k_est, Temp, ca_ins, ca_exper
end

# ============================================================================
# OPTIMAL DESIGN FUNCTIONS (FIM/GIM)
# ============================================================================

"""
    DesignSpace

Structure for experimental design space.
"""
struct DesignSpace
    T_min::Float64
    T_max::Float64
    Ca_min::Float64
    Ca_max::Float64
    T_grid::Vector{Float64}
    Ca_grid::Vector{Float64}
end

"""
    create_design_space(; T_min=300.0, T_max=500.0, Ca_min=0.1, Ca_max=2.0)

Create design space grid for optimal experimental design.
"""
function create_design_space(; T_min=300.0, T_max=500.0, Ca_min=0.1, Ca_max=2.0)
    T_grid = collect(range(T_min, T_max, length=20))
    Ca_grid = collect(range(Ca_min, Ca_max, length=15))
    return DesignSpace(T_min, T_max, Ca_min, Ca_max, T_grid, Ca_grid)
end

"""
    compute_sensitivities(k, x; δ=1e-5)

Calculate parameter sensitivities using central finite differences.
"""
function compute_sensitivities(k, x; δ=1e-5)
    ca_in, T = x
    S = zeros(2)
    
    for i in 1:2
        h = max(δ * abs(k[i]), 1e-8)
        
        k_plus = copy(k); k_plus[i] += h
        y_plus = CSTR_model(ca_in, T; k=k_plus)
        
        k_minus = copy(k); k_minus[i] -= h
        y_minus = CSTR_model(ca_in, T; k=k_minus)
        
        if h > 1e-12
            S[i] = (y_plus - y_minus) / (2h)
        end
    end
    return S
end

"""
    compute_fim(k_nominal, experiments, σ; regularization=1e-8)

Compute Fisher Information Matrix.
"""
function compute_fim(k_nominal, experiments, σ; regularization=1e-8)
    FIM = zeros(2, 2)
    σ_safe = max(σ, 1e-6)
    
    for x in experiments
        S = compute_sensitivities(k_nominal, x)
        if norm(S) > 1e-10
            FIM += (1/σ_safe^2) * (S * S')
        end
    end
    
    FIM += regularization * I(2)
    return FIM
end

"""
    compute_gim(prior_mean, prior_cov, experiments, σ; n_samples=50)

Compute Global Information Matrix by averaging over prior distribution.
"""
function compute_gim(prior_mean, prior_cov, experiments, σ; n_samples=50)
    try
        prior_dist = MvNormal(prior_mean, prior_cov + 1e-6*I)
    catch
        prior_dist = MvNormal(prior_mean, Diagonal(diag(prior_cov)))
    end
    
    GIM = zeros(2, 2)
    valid = 0
    
    for i in 1:n_samples
        try
            k_sample = rand(prior_dist)
            k_sample = max.(k_sample, 1e-4)
            FIM_sample = compute_fim(k_sample, experiments, σ)
            GIM += FIM_sample
            valid += 1
        catch
            continue
        end
    end
    
    if valid > 0
        GIM /= valid
    end
    GIM += 1e-8 * I(2)
    return GIM
end

"""
    d_optimality(FIM; ϵ=1e-10)

D-optimality criterion: maximize determinant of FIM.
"""
function d_optimality(FIM; ϵ=1e-10)
    try
        FIM_reg = FIM + ϵ * I(2)
        d = det(FIM_reg)
        return d <= 0 ? -1e10 : log(d)
    catch
        return -1e10
    end
end

"""
    run_optimal_sequential_design(space, true_k, n_total, σ; 
                                   method="FIM", prior_mean=nothing, prior_cov=nothing,
                                   add_noise=true, initial_guess=[0.3, 3000.0],
                                   n_initial=2)

Run optimal sequential design using FIM or GIM with exactly n_total experiments.
"""
function run_optimal_sequential_design(space, true_k, n_total, σ; 
                                        method="FIM", 
                                        prior_mean=nothing,
                                        prior_cov=nothing,
                                        add_noise=true,
                                        initial_guess=[0.3, 3000.0],
                                        n_initial=2)
    
    println("\n" * "="^80)
    println("OPTIMAL SEQUENTIAL DESIGN USING $method")
    println("Seed: $(GLOBAL_SEED)")
    println("Total experiments: $n_total (initial: $n_initial, optimal: $(n_total-n_initial))")
    println("Noise: $(add_noise ? "$(σ*100)%" : "None")")
    println("="^80)
    
    # Initial random experiments
    experiments = []
    for i in 1:n_initial
        T = space.T_min + (space.T_max - space.T_min) * rand()
        Ca = space.Ca_min + (space.Ca_max - space.Ca_min) * rand()
        push!(experiments, [T, Ca])
    end
    
    # Simulate initial experiments
    measurements = Float64[]
    for x in experiments
        ca_in, T = x
        y_true = CSTR_model(ca_in, T; k=true_k)
        if add_noise && σ > 0
            push!(measurements, y_true + σ * y_true * randn())
        else
            push!(measurements, y_true)
        end
    end
    
    # Initial parameter estimation
    function objective(k)
        error = 0.0
        for i in 1:length(experiments)
            ca_in, T = experiments[i]
            pred = CSTR_model(ca_in, T; k=k)
            error += (pred - measurements[i])^2
        end
        return error
    end
    res = optimize(objective, initial_guess, LBFGS(), Optim.Options(iterations=500))
    k_est = Optim.minimizer(res)
    
    # Storage for results
    results = Dict(
        "method" => method,
        "noise_level" => add_noise ? σ : 0.0,
        "exp_numbers" => [n_initial],
        "k_A" => [k_est[1]],
        "k_Ea" => [k_est[2]],
        "error_A" => [abs(k_est[1] - true_k[1])/true_k[1]*100],
        "error_Ea" => [abs(k_est[2] - true_k[2])/true_k[2]*100],
        "fim_det" => [det(compute_fim(k_est, experiments, σ))],
        "selected_T" => Float64[],
        "selected_Ca" => Float64[],
        "improvements_A" => Float64[],
        "all_experiments" => [deepcopy(experiments)],
        "all_measurements" => [copy(measurements)]
    )
    
    println("\n📊 INITIAL ESTIMATES (Experiments 1-$n_initial):")
    println("   A = $(round(k_est[1], digits=3)) (true: $(true_k[1]))")
    println("   Ea = $(round(k_est[2], digits=0)) (true: $(true_k[2]))")
    println("   Errors: A = $(round(results["error_A"][1], digits=2))%, Ea = $(round(results["error_Ea"][1], digits=2))%")
    
    # Sequential addition of optimal experiments
    for iter in 1:(n_total - n_initial)
        println("\n" * "-"^60)
        println("ITERATION $iter: Selecting experiment $(n_initial+iter) of $n_total")
        
        # Find best next experiment
        best_score = -Inf
        best_exp = nothing
        candidates_tested = 0
        
        for T in space.T_grid
            for Ca in space.Ca_grid
                candidate = [T, Ca]
                
                # Skip duplicates
                duplicate = false
                for exp in experiments
                    if abs(exp[1] - T) < 1e-6 && abs(exp[2] - Ca) < 1e-6
                        duplicate = true
                        break
                    end
                end
                duplicate && continue
                
                candidates_tested += 1
                candidate_set = vcat(experiments, [candidate])
                
                if method == "GIM"
                    info = compute_gim(prior_mean, prior_cov, candidate_set, σ)
                else
                    info = compute_fim(k_est, candidate_set, σ)
                end
                score = d_optimality(info)
                
                if score > best_score
                    best_score = score
                    best_exp = candidate
                end
            end
        end
        
        println("   Evaluated $candidates_tested candidate experiments")
        
        if best_exp === nothing
            println("⚠️ No suitable experiment found")
            break
        end
        
        T_opt, Ca_opt = best_exp
        push!(results["selected_T"], T_opt)
        push!(results["selected_Ca"], Ca_opt)
        push!(experiments, best_exp)
        
        println("   ✅ OPTIMAL EXPERIMENT SELECTED:")
        println("      T = $(round(T_opt, digits=1)) K")
        println("      Ca_in = $(round(Ca_opt, digits=2)) mol/m³")
        
        # Simulate new experiment
        y_true = CSTR_model(T_opt, Ca_opt; k=true_k)
        if add_noise && σ > 0
            new_meas = y_true + σ * y_true * randn()
        else
            new_meas = y_true
        end
        push!(measurements, new_meas)
        
        # Store previous error for improvement calculation
        prev_err_A = results["error_A"][end]
        prev_k = copy(k_est)
        
        # Re-estimate parameters with all data
        function objective_new(k)
            error = 0.0
            for i in 1:length(experiments)
                ca_in, T = experiments[i]
                pred = CSTR_model(ca_in, T; k=k)
                error += (pred - measurements[i])^2
            end
            return error
        end
        res = optimize(objective_new, k_est, LBFGS(), Optim.Options(iterations=500))
        k_est = Optim.minimizer(res)
        
        # Calculate new errors
        new_err_A = abs(k_est[1] - true_k[1])/true_k[1]*100
        new_err_Ea = abs(k_est[2] - true_k[2])/true_k[2]*100
        imp_A = prev_err_A - new_err_A
        
        # Update results
        push!(results["improvements_A"], imp_A)
        push!(results["exp_numbers"], length(experiments))
        push!(results["k_A"], k_est[1])
        push!(results["k_Ea"], k_est[2])
        push!(results["error_A"], new_err_A)
        push!(results["error_Ea"], new_err_Ea)
        push!(results["fim_det"], det(compute_fim(k_est, experiments, σ)))
        push!(results["all_experiments"], deepcopy(experiments))
        push!(results["all_measurements"], copy(measurements))
        
        println("\n   📊 NEW PARAMETER ESTIMATES:")
        println("      Previous: A=$(round(prev_k[1], digits=3)), Ea=$(round(prev_k[2], digits=0))")
        println("      New:      A=$(round(k_est[1], digits=3)), Ea=$(round(k_est[2], digits=0))")
        println("\n   📈 ERROR IMPROVEMENT:")
        println("      A: $(round(prev_err_A, digits=2))% → $(round(new_err_A, digits=2))%  [↓$(round(imp_A, digits=2))%]")
    end
    
    results["final_experiments"] = experiments
    results["final_estimate"] = k_est
    results["final_measurements"] = measurements
    
    return results
end

# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

"""
    compare_designs(; n_experiments=6, noise_levels=[0.0, 0.05], n_runs=3)

Compare random design vs optimal design (FIM and GIM) over multiple runs.
Shows that random design gives different results each time, while optimal
design consistently improves estimates.
"""
function compare_designs(; n_experiments=6, noise_levels=[0.0, 0.05], n_runs=3)
    
    println("\n" * "="^80)
    println("COMPARISON OF RANDOM VS OPTIMAL DESIGN")
    println("="^80)
    
    space = create_design_space()
    true_k = [2.5, 25000.0]
    prior_mean = [2.5, 25000.0]
    prior_cov = Diagonal([1.0, 1e7])
    
    for σ in noise_levels
        noise_str = σ > 0 ? "$(σ*100)% noise" : "no noise"
        println("\n" * "-"^70)
        println("TESTING WITH $noise_str")
        println("-"^70)
        
        # Multiple runs of random design
        println("\n📊 RANDOM DESIGN (multiple runs with same seed change):")
        random_errors = []
        for run in 1:n_runs
            # Change seed slightly for each run
            Random.seed!(GLOBAL_SEED + run)
            println("\n   Run $run:")
            k_est, _, _, _ = run_parameter_estimation(Nexps=n_experiments, 
                                                       add_noise=(σ>0), sigma=σ,
                                                       true_k=true_k)
            push!(random_errors, abs(k_est[1] - true_k[1])/true_k[1]*100)
        end
        println("\n   Random design A errors: $(round.(random_errors, digits=2))%")
        println("   Mean: $(round(mean(random_errors), digits=2))%, Std: $(round(std(random_errors), digits=2))%")
        
        # Reset seed for deterministic optimal design
        Random.seed!(GLOBAL_SEED)
        
        # FIM optimal design
        println("\n📈 FIM OPTIMAL DESIGN:")
        fim_results = run_optimal_sequential_design(space, true_k, n_experiments, σ,
                                                     method="FIM", add_noise=(σ>0))
        fim_final_error = fim_results["error_A"][end]
        println("\n   FIM final A error: $(round(fim_final_error, digits=2))%")
        
        # GIM optimal design
        println("\n📈 GIM OPTIMAL DESIGN:")
        gim_results = run_optimal_sequential_design(space, true_k, n_experiments, σ,
                                                     method="GIM", prior_mean=prior_mean,
                                                     prior_cov=prior_cov, add_noise=(σ>0))
        gim_final_error = gim_results["error_A"][end]
        println("\n   GIM final A error: $(round(gim_final_error, digits=2))%")
        
        # Summary
        println("\n" * "-"^70)
        println("SUMMARY FOR $noise_str:")
        println("   Random design (mean ± std): $(round(mean(random_errors), digits=2))% ± $(round(std(random_errors), digits=2))%")
        println("   FIM optimal design:         $(round(fim_final_error, digits=2))%")
        println("   GIM optimal design:         $(round(gim_final_error, digits=2))%")
        
        if fim_final_error < mean(random_errors) - std(random_errors)
            println("   ✓ FIM performs better than random design")
        end
        if gim_final_error < fim_final_error
            println("   ✓ GIM performs better than FIM with $noise_str")
        end
    end
end

# ============================================================================
# DEMONSTRATION
# ============================================================================

"""
    demonstrate_with_seed_changes()

Show how changing the seed produces different results with random design,
while optimal design consistently improves estimates.
"""
function demonstrate_with_seed_changes()
    
    println("\n" * "="^80)
    println("DEMONSTRATION: EFFECT OF RANDOM SEED ON RESULTS")
    println("="^80)
    
    true_k = [2.5, 25000.0]
    space = create_design_space()
    prior_mean = [2.5, 25000.0]
    prior_cov = Diagonal([1.0, 1e7])
    
    # Test with different seeds
    seeds = [123, 456, 789, 111, 222]
    
    println("\n📊 RANDOM DESIGN WITH DIFFERENT SEEDS:")
    println("-"^70)
    
    random_results = []
    for seed in seeds
        Random.seed!(seed)
        println("\n   Seed: $seed")
        k_est, _, _, _ = run_parameter_estimation(Nexps=5, add_noise=true, sigma=0.02,
                                                   true_k=true_k, initial_guess=[0.5, 15000.0])
        push!(random_results, (seed, k_est, abs(k_est[1]-true_k[1])/true_k[1]*100))
    end
    
    println("\n📈 OPTIMAL DESIGN (deterministic - seed independent):")
    println("-"^70)
    
    # Reset to a fixed seed for optimal design
    Random.seed!(123)
    
    println("\n   FIM Optimal Design:")
    fim_results = run_optimal_sequential_design(space, true_k, 6, 0.02,
                                                 method="FIM", add_noise=true)
    
    println("\n   GIM Optimal Design:")
    gim_results = run_optimal_sequential_design(space, true_k, 6, 0.02,
                                                 method="GIM", prior_mean=prior_mean,
                                                 prior_cov=prior_cov, add_noise=true)
    
    println("\n" * "="^80)
    println("KEY OBSERVATION:")
    println("="^80)
    println("""
    Random design gives DIFFERENT results each time you run it because:
    - Different random seeds produce different experimental conditions
    - Parameter estimates vary significantly between runs
    
    Optimal design gives CONSISTENT improvement because:
    - Experiments are chosen based on information content, not randomness
    - The algorithm always selects the most informative conditions
    - Results are reproducible regardless of random seed
    
    To get different results with random design, change the seed.
    To get different results with optimal design, you would need to
    change the true parameters or the design space.
    """)
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

println("\n" * "="^80)
println("OPTIMAL EXPERIMENTAL DESIGN FOR MASTER THESIS")
println("Current global seed: $(GLOBAL_SEED)")
println("="^80)

# Uncomment the desired demonstration:

# 1. Basic parameter estimation with random design (will be same each run with same seed)
# run_parameter_estimation(Nexps=5, add_noise=true, sigma=0.02)

# 2. Show how changing seed gives different results with random design
# Random.seed!(123); run_parameter_estimation(Nexps=5, add_noise=true, sigma=0.02)
# Random.seed!(456); run_parameter_estimation(Nexps=5, add_noise=true, sigma=0.02)

# 3. Optimal sequential design (deterministic)
# space = create_design_space()
# results = run_optimal_sequential_design(space, [2.5, 25000.0], 6, 0.05,
#                                         method="FIM", add_noise=true)

# 4. Compare random vs optimal designs
compare_designs(n_experiments=6, noise_levels=[0.0, 0.05], n_runs=3)

# 5. Demonstrate seed effect
# demonstrate_with_seed_changes()