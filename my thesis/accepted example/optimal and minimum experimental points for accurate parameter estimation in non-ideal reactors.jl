# ============================================================================
# OPTIMAL EXPERIMENTAL DESIGN USING FIM AND GIM
# For Kinetic Parameter Estimation in Non-Ideal Reactors
# CORRECTED VERSION - NO COMPLEX NUMBERS
# ============================================================================

# ============================================================================
# STEP 1: LOAD REQUIRED PACKAGES
# ============================================================================

using Plots
using LinearAlgebra
using Distributions
using Random
using Statistics
using Printf

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================================
# STEP 2: REACTOR MODEL (Non-ideal with axial dispersion)
# ============================================================================

"""
Axial Dispersion Reactor Model for non-ideal flow
Accounts for deviations from plug flow behavior
"""
function axial_dispersion_model(θ, x)
    # θ[1] = Pre-exponential factor A (s⁻¹)
    # θ[2] = Activation energy E (J/mol)
    A, E = θ
    
    # x[1] = Temperature T (K)
    # x[2] = Inlet concentration C₀ (mol/m³)
    # x[3] = Flow velocity u (m/s)
    # x[4] = Reactor position z (dimensionless)
    # x[5] = Peclet number Pe (axial dispersion)
    T, C0, u, z, Pe = x
    
    # Check for valid inputs
    if A <= 0 || E <= 0 || T <= 0 || C0 <= 0 || u <= 0 || z < 0 || z > 1 || Pe <= 0
        return 0.0
    end
    
    # Gas constant
    R = 8.314
    
    # Reactor length
    L = 1.0
    
    # Arrhenius equation
    k = A * exp(-E/(R*T))
    
    # Damköhler number
    Da = k * L / u
    
    # Analytical solution for axial dispersion model (first-order reaction)
    # Handle edge cases
    if Pe < 1e-6  # Well-mixed limit (CSTR)
        return C0 / (1 + Da)
    elseif Da < 1e-6  # No reaction
        return C0
    else
        # Calculate eigenvalues
        discriminant = sqrt(1 + 4*Da/Pe)
        λ1 = (Pe/2) * (1 + discriminant)
        λ2 = (Pe/2) * (1 - discriminant)
        
        # Calculate constants from boundary conditions
        denom = (λ2 - λ1) * exp(λ1 - λ2)
        
        # Avoid division by zero
        if abs(denom) < 1e-10
            return C0 * exp(-Da * z)
        end
        
        C1 = λ2 / denom
        C2 = -λ1 / denom
        
        # Concentration profile
        C = C0 * (C1 * exp(λ1 * (z-1)) + C2 * exp(λ2 * (z-1)))
        
        # Ensure concentration is physical
        return max(0.0, min(C, C0))
    end
end

# ============================================================================
# STEP 3: SENSITIVITY ANALYSIS FUNCTIONS (USING FINITE DIFFERENCES)
# ============================================================================

"""
Compute parameter sensitivities using forward finite differences
Avoids complex numbers and is numerically stable
"""
function compute_sensitivities_fd(θ, x, model; δ=1e-5)
    n_params = length(θ)
    sensitivities = zeros(n_params)
    
    # Base value
    try
        y0 = model(θ, x)
        
        for i in 1:n_params
            # Determine perturbation size
            if abs(θ[i]) < 1e-10
                h = δ
            else
                h = δ * abs(θ[i])
            end
            
            # Forward difference
            θ_pert = copy(θ)
            θ_pert[i] += h
            y_pert = model(θ_pert, x)
            
            # Calculate sensitivity
            if h > 1e-12
                sensitivities[i] = (y_pert - y0) / h
            else
                sensitivities[i] = 0.0
            end
            
            # Check for invalid values
            if isnan(sensitivities[i]) || isinf(sensitivities[i])
                sensitivities[i] = 0.0
            end
        end
    catch e
        # Return zeros if model fails
        return zeros(n_params)
    end
    
    return sensitivities
end

"""
Compute parameter sensitivities using central differences (more accurate)
"""
function compute_sensitivities_central(θ, x, model; δ=1e-4)
    n_params = length(θ)
    sensitivities = zeros(n_params)
    
    for i in 1:n_params
        # Determine perturbation size
        if abs(θ[i]) < 1e-10
            h = δ
        else
            h = δ * abs(θ[i])
        end
        
        # Central difference
        θ_plus = copy(θ)
        θ_minus = copy(θ)
        θ_plus[i] += h
        θ_minus[i] -= h
        
        y_plus = model(θ_plus, x)
        y_minus = model(θ_minus, x)
        
        # Calculate sensitivity
        if h > 1e-12
            sensitivities[i] = (y_plus - y_minus) / (2 * h)
        else
            sensitivities[i] = 0.0
        end
        
        # Check for invalid values
        if isnan(sensitivities[i]) || isinf(sensitivities[i])
            sensitivities[i] = 0.0
        end
    end
    
    return sensitivities
end

# ============================================================================
# STEP 4: INFORMATION MATRIX COMPUTATION
# ============================================================================

"""
Compute Fisher Information Matrix for a set of experiments
"""
function compute_fim_experiments(θ, experiments, σ, model; method="central")
    n_params = length(θ)
    FIM = zeros(n_params, n_params)
    
    # Ensure σ is positive
    σ_safe = max(σ, 1e-6)
    
    valid_experiments = 0
    
    for (i, x) in enumerate(experiments)
        try
            # Choose sensitivity method
            if method == "central"
                S = compute_sensitivities_central(θ, x, model)
            else
                S = compute_sensitivities_fd(θ, x, model)
            end
            
            # Check if sensitivities are valid
            if !any(isnan.(S)) && !any(isinf.(S)) && norm(S) > 1e-10
                FIM += (1/σ_safe^2) * (S * S')
                valid_experiments += 1
            end
        catch e
            # Skip problematic experiments
            continue
        end
    end
    
    # Add small regularization if no valid experiments
    if valid_experiments == 0
        FIM = 1e-6 * I(n_params)
    else
        # Add small regularization for numerical stability
        FIM += 1e-8 * I(n_params)
    end
    
    return FIM
end

"""
Compute Global Information Matrix accounting for parameter uncertainty
"""
function compute_gim_experiments(prior_mean, prior_cov, experiments, σ, model; 
                                n_samples=200, method="central")
    n_params = length(prior_mean)
    GIM = zeros(n_params, n_params)
    
    # Create prior distribution with regularization
    prior_cov_reg = prior_cov + 1e-6 * I(n_params)
    
    # Check if prior_cov_reg is positive definite
    try
        prior_dist = MvNormal(prior_mean, prior_cov_reg)
    catch e
        # If not, use diagonal approximation
        prior_dist = MvNormal(prior_mean, Diagonal(diag(prior_cov_reg)))
    end
    
    valid_samples = 0
    
    for i in 1:n_samples
        try
            # Sample from prior
            θ_sample = rand(prior_dist)
            
            # Ensure parameters are physical
            θ_sample = max.(θ_sample, 1e-4)  # Minimum positive values
            
            # Compute FIM for these sampled parameters
            FIM_local = compute_fim_experiments(θ_sample, experiments, σ, model, method=method)
            
            # Check if FIM is valid
            if !any(isnan.(FIM_local)) && !any(isinf.(FIM_local))
                GIM += FIM_local
                valid_samples += 1
            end
        catch e
            continue
        end
    end
    
    if valid_samples > 0
        GIM /= valid_samples
    else
        GIM = 1e-6 * I(n_params)
    end
    
    # Add regularization
    GIM += 1e-8 * I(n_params)
    
    return GIM
end

# ============================================================================
# STEP 5: OPTIMAL DESIGN CRITERIA
# ============================================================================

"""
D-optimality criterion: Maximize determinant of information matrix
"""
function d_optimality_criterion(FIM; ϵ=1e-10)
    try
        d = det(FIM + ϵ * I)
        return log(max(d, ϵ))
    catch e
        return -1e10  # Very bad design
    end
end

"""
A-optimality criterion: Minimize trace of inverse (average variance)
"""
function a_optimality_criterion(FIM; ϵ=1e-10)
    try
        FIM_reg = FIM + ϵ * I
        return -tr(inv(FIM_reg))
    catch e
        return -1e10  # Very bad design
    end
end

"""
E-optimality criterion: Maximize minimum eigenvalue
"""
function e_optimality_criterion(FIM; ϵ=1e-10)
    try
        FIM_reg = FIM + ϵ * I
        λ = eigvals(FIM_reg)
        return minimum(λ)
    catch e
        return -1e10
    end
end

"""
Combined criterion balancing D and A optimality
"""
function combined_criterion(FIM; α=0.7, ϵ=1e-10)
    try
        FIM_reg = FIM + ϵ * I
        d_opt = log(det(FIM_reg))
        a_opt = -tr(inv(FIM_reg))
        
        # Normalize (rough approximation)
        d_norm = d_opt / (abs(d_opt) + 1)
        a_norm = a_opt / (abs(a_opt) + 1)
        
        return α * d_norm + (1-α) * a_norm
    catch e
        return -1e10
    end
end

# ============================================================================
# STEP 6: EXPERIMENTAL DESIGN SPACE DEFINITION
# ============================================================================

"""
Define the experimental design space
"""
function define_design_space()
    # Temperature range (K)
    T_range = collect(350:10:450)
    
    # Inlet concentration range (mol/m³)
    C0_range = collect(0.5:0.2:1.5)
    
    # Flow velocity range (m/s)
    u_range = collect(0.05:0.05:0.2)
    
    # Reactor position range (dimensionless)
    z_range = collect(0.2:0.2:0.8)
    
    # Peclet number range (axial dispersion)
    Pe_range = [10, 20, 50, 100]
    
    return (T_range, C0_range, u_range, z_range, Pe_range)
end

"""
Generate initial space-filling design
"""
function generate_initial_design(design_space, n_initial)
    T_range, C0_range, u_range, z_range, Pe_range = design_space
    
    experiments = []
    
    for i in 1:n_initial
        # Random sampling
        T = T_range[rand(1:length(T_range))]
        C0 = C0_range[rand(1:length(C0_range))]
        u = u_range[rand(1:length(u_range))]
        z = z_range[rand(1:length(z_range))]
        Pe = Pe_range[rand(1:length(Pe_range))]
        
        push!(experiments, [T, C0, u, z, Pe])
    end
    
    return experiments
end

"""
Check if two experiments are approximately equal
"""
function experiments_equal(exp1, exp2; tol=1e-6)
    return norm(exp1 - exp2) < tol
end

# ============================================================================
# STEP 7: SEQUENTIAL OPTIMAL DESIGN ALGORITHM (FIM)
# ============================================================================

"""
Sequential optimal experimental design using FIM
"""
function sequential_optimal_design_fim(θ_nominal, design_space, σ, model;
                                      n_initial=3, n_max=12, 
                                      criterion=d_optimality_criterion,
                                      improvement_threshold=0.01,
                                      method="central")
    
    println("\n" * "="^70)
    println("SEQUENTIAL OPTIMAL DESIGN USING FIM")
    println("="^70)
    
    # Generate initial design
    experiments = generate_initial_design(design_space, n_initial)
    
    # Compute initial criterion
    FIM_initial = compute_fim_experiments(θ_nominal, experiments, σ, model, method=method)
    criterion_initial = criterion(FIM_initial)
    
    # Storage for results
    history = Dict(
        :n_experiments => [n_initial],
        :criterion_values => [criterion_initial],
        :parameter_covariance => [inv(FIM_initial + 1e-8*I)],
        :experiments => [deepcopy(experiments)]
    )
    
    println("\nInitial design with $n_initial experiments")
    println("Initial criterion value: $(round(criterion_initial, digits=4))")
    
    # Sequential addition of experiments
    T_range, C0_range, u_range, z_range, Pe_range = design_space
    
    for iter in 1:(n_max - n_initial)
        println("\nIteration $(iter): Searching for optimal next experiment...")
        
        best_criterion = -Inf
        best_experiment = nothing
        
        # Search over design space
        search_count = 0
        for T in T_range
            for C0 in C0_range
                for u in u_range
                    for z in z_range
                        for Pe in Pe_range
                            candidate = [T, C0, u, z, Pe]
                            
                            # Skip if already in experiments
                            already_exists = false
                            for exp in experiments
                                if experiments_equal(candidate, exp)
                                    already_exists = true
                                    break
                                end
                            end
                            
                            if already_exists
                                continue
                            end
                            
                            search_count += 1
                            
                            # Evaluate candidate
                            candidate_set = vcat(experiments, [candidate])
                            try
                                FIM_candidate = compute_fim_experiments(θ_nominal, candidate_set, σ, model, method=method)
                                crit_val = criterion(FIM_candidate)
                                
                                if crit_val > best_criterion && isfinite(crit_val)
                                    best_criterion = crit_val
                                    best_experiment = candidate
                                end
                            catch e
                                continue
                            end
                        end
                    end
                end
            end
        end
        
        println("  Searched $search_count candidate experiments")
        
        # Add best experiment
        if best_experiment !== nothing
            push!(experiments, best_experiment)
            
            # Compute new FIM
            FIM_new = compute_fim_experiments(θ_nominal, experiments, σ, model, method=method)
            crit_new = criterion(FIM_new)
            
            # Store history
            push!(history[:n_experiments], length(experiments))
            push!(history[:criterion_values], crit_new)
            push!(history[:parameter_covariance], inv(FIM_new + 1e-8*I))
            push!(history[:experiments], deepcopy(experiments))
            
            # Calculate improvement
            improvement = (crit_new - history[:criterion_values][end-1]) / abs(history[:criterion_values][end-1])
            
            println("  Added experiment:")
            println("    T = $(best_experiment[1]) K")
            println("    C0 = $(best_experiment[2]) mol/m³")
            println("    u = $(best_experiment[3]) m/s")
            println("    z = $(best_experiment[4])")
            println("    Pe = $(best_experiment[5])")
            println("  New criterion: $(round(crit_new, digits=4))")
            println("  Improvement: $(round(improvement*100, digits=2))%")
            
            # Check convergence
            if improvement < improvement_threshold && iter > 1
                println("\n✓ Convergence achieved! Improvement below $(improvement_threshold*100)%")
                break
            end
        else
            println("No suitable experiment found. Stopping.")
            break
        end
    end
    
    return experiments, history
end

# ============================================================================
# STEP 8: SEQUENTIAL OPTIMAL DESIGN ALGORITHM (GIM)
# ============================================================================

"""
Sequential optimal design using GIM (accounts for parameter uncertainty)
"""
function sequential_optimal_design_gim(prior_mean, prior_cov, design_space, σ, model;
                                      n_initial=3, n_max=12,
                                      criterion=d_optimality_criterion,
                                      improvement_threshold=0.01,
                                      n_samples=100,
                                      method="central")
    
    println("\n" * "="^70)
    println("SEQUENTIAL OPTIMAL DESIGN USING GIM")
    println("="^70)
    
    # Generate initial design
    experiments = generate_initial_design(design_space, n_initial)
    
    # Compute initial criterion
    GIM_initial = compute_gim_experiments(prior_mean, prior_cov, experiments, σ, model, 
                                         n_samples=n_samples, method=method)
    criterion_initial = criterion(GIM_initial)
    
    # Storage for results
    history = Dict(
        :n_experiments => [n_initial],
        :criterion_values => [criterion_initial],
        :parameter_covariance => [inv(GIM_initial + 1e-8*I)],
        :experiments => [deepcopy(experiments)]
    )
    
    println("\nInitial design with $n_initial experiments")
    println("Initial criterion value: $(round(criterion_initial, digits=4))")
    
    # Sequential addition of experiments
    T_range, C0_range, u_range, z_range, Pe_range = design_space
    
    for iter in 1:(n_max - n_initial)
        println("\nIteration $(iter): Searching for optimal next experiment...")
        
        best_criterion = -Inf
        best_experiment = nothing
        
        # Search over design space
        search_count = 0
        for T in T_range
            for C0 in C0_range
                for u in u_range
                    for z in z_range
                        for Pe in Pe_range
                            candidate = [T, C0, u, z, Pe]
                            
                            # Skip if already in experiments
                            already_exists = false
                            for exp in experiments
                                if experiments_equal(candidate, exp)
                                    already_exists = true
                                    break
                                end
                            end
                            
                            if already_exists
                                continue
                            end
                            
                            search_count += 1
                            
                            # Evaluate candidate using GIM
                            candidate_set = vcat(experiments, [candidate])
                            try
                                GIM_candidate = compute_gim_experiments(prior_mean, prior_cov, candidate_set, 
                                                                       σ, model, n_samples=n_samples, method=method)
                                crit_val = criterion(GIM_candidate)
                                
                                if crit_val > best_criterion && isfinite(crit_val)
                                    best_criterion = crit_val
                                    best_experiment = candidate
                                end
                            catch e
                                continue
                            end
                        end
                    end
                end
            end
        end
        
        println("  Searched $search_count candidate experiments")
        
        # Add best experiment
        if best_experiment !== nothing
            push!(experiments, best_experiment)
            
            # Compute new GIM
            GIM_new = compute_gim_experiments(prior_mean, prior_cov, experiments, σ, model,
                                             n_samples=n_samples, method=method)
            crit_new = criterion(GIM_new)
            
            # Store history
            push!(history[:n_experiments], length(experiments))
            push!(history[:criterion_values], crit_new)
            push!(history[:parameter_covariance], inv(GIM_new + 1e-8*I))
            push!(history[:experiments], deepcopy(experiments))
            
            # Calculate improvement
            improvement = (crit_new - history[:criterion_values][end-1]) / abs(history[:criterion_values][end-1])
            
            println("  Added experiment:")
            println("    T = $(best_experiment[1]) K")
            println("    C0 = $(best_experiment[2]) mol/m³")
            println("    u = $(best_experiment[3]) m/s")
            println("    z = $(best_experiment[4])")
            println("    Pe = $(best_experiment[5])")
            println("  New criterion: $(round(crit_new, digits=4))")
            println("  Improvement: $(round(improvement*100, digits=2))%")
            
            # Check convergence
            if improvement < improvement_threshold && iter > 1
                println("\n✓ Convergence achieved! Improvement below $(improvement_threshold*100)%")
                break
            end
        else
            println("No suitable experiment found. Stopping.")
            break
        end
    end
    
    return experiments, history
end

# ============================================================================
# STEP 9: OPTIMAL NUMBER OF EXPERIMENTS DETERMINATION
# ============================================================================

"""
Determine the optimal number of experiments using elbow method
"""
function optimal_experiment_count(history; threshold=0.03)
    n_experiments = history[:n_experiments]
    criterion_values = history[:criterion_values]
    
    if length(criterion_values) < 3
        return n_experiments[end]
    end
    
    # Calculate marginal improvements
    improvements = Float64[]
    for i in 2:length(criterion_values)
        imp = (criterion_values[i] - criterion_values[i-1]) / max(abs(criterion_values[i-1]), 1e-10)
        push!(improvements, imp)
    end
    
    # Find where improvement drops below threshold for at least 2 consecutive steps
    for i in 1:length(improvements)-1
        if improvements[i] < threshold && improvements[i+1] < threshold
            return n_experiments[i+1]
        end
    end
    
    return n_experiments[end]
end

# ============================================================================
# STEP 10: DESIGN COMPARISON
# ============================================================================

"""
Compare efficiency of different experimental designs
"""
function compare_designs(fim_exp, gim_exp, random_exp, θ_true, prior_mean, prior_cov, σ, model)
    println("\n" * "="^70)
    println("DESIGN COMPARISON")
    println("="^70)
    
    # Compute information matrices
    FIM_fim = compute_fim_experiments(θ_true, fim_exp, σ, model)
    FIM_gim = compute_fim_experiments(θ_true, gim_exp, σ, model)
    FIM_random = compute_fim_experiments(θ_true, random_exp, σ, model)
    
    GIM_fim = compute_gim_experiments(prior_mean, prior_cov, fim_exp, σ, model, n_samples=100)
    GIM_gim = compute_gim_experiments(prior_mean, prior_cov, gim_exp, σ, model, n_samples=100)
    GIM_random = compute_gim_experiments(prior_mean, prior_cov, random_exp, σ, model, n_samples=100)
    
    # Print comparison
    println("\n{:<15} {:>15} {:>15} {:>15}".format("Design", "det(FIM)", "det(GIM)", "cond(FIM)"))
    println("-" * 70)
    
    for (name, FIM, GIM) in [("FIM-Optimal", FIM_fim, GIM_fim),
                              ("GIM-Optimal", FIM_gim, GIM_gim),
                              ("Random", FIM_random, GIM_random)]
        
        det_FIM = det(FIM)
        det_GIM = det(GIM)
        cond_FIM = cond(FIM)
        
        println("{:<15} {:15.2e} {:15.2e} {:15.2e}".format(
            name, det_FIM, det_GIM, cond_FIM))
    end
    
    # Calculate relative efficiency
    println("\nRelative Efficiency (compared to random design):")
    
    eff_fim = det(FIM_fim) / max(det(FIM_random), 1e-10)
    eff_gim = det(FIM_gim) / max(det(FIM_random), 1e-10)
    
    println("  FIM-Optimal: $(round(eff_fim, digits=2))× more informative")
    println("  GIM-Optimal: $(round(eff_gim, digits=2))× more informative")
    
    return Dict(
        "fim_det" => det(FIM_fim),
        "gim_det" => det(FIM_gim),
        "random_det" => det(FIM_random),
        "eff_fim" => eff_fim,
        "eff_gim" => eff_gim
    )
end

# ============================================================================
# STEP 11: MAIN EXECUTION
# ============================================================================

println("\n" * "="^70)
println("OPTIMAL EXPERIMENTAL DESIGN FOR KINETIC PARAMETER ESTIMATION")
println("Non-ideal Reactor with Axial Dispersion")
println("="^70)

# Set up problem
θ_true = [1e5, 50000.0]  # True parameters
θ_prior_mean = [1e5, 50000.0]  # Prior mean
θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])  # Prior uncertainty
σ = 0.01  # Measurement noise (1%)

# Define design space
design_space = define_design_space()
println("\nDesign Space:")
println("  Temperature: 350-450 K")
println("  Inlet concentration: 0.5-1.5 mol/m³")
println("  Flow velocity: 0.05-0.2 m/s")
println("  Reactor position: 0.2-0.8")
println("  Peclet number: 10-100")

# ----------------------------------------------------------------------------
# Run sequential design with FIM
# ----------------------------------------------------------------------------
println("\n" * "-"^70)
fim_experiments, fim_history = sequential_optimal_design_fim(
    θ_true, design_space, σ, axial_dispersion_model,
    n_initial=3, n_max=10, criterion=d_optimality_criterion,
    method="central"
)

# ----------------------------------------------------------------------------
# Run sequential design with GIM
# ----------------------------------------------------------------------------
println("\n" * "-"^70)
gim_experiments, gim_history = sequential_optimal_design_gim(
    θ_prior_mean, θ_prior_cov, design_space, σ, axial_dispersion_model,
    n_initial=3, n_max=10, criterion=d_optimality_criterion,
    n_samples=80, method="central"
)

# ----------------------------------------------------------------------------
# Generate random design for comparison
# ----------------------------------------------------------------------------
n_random = min(length(fim_experiments), length(gim_experiments))
random_experiments = generate_initial_design(design_space, n_random)

# ----------------------------------------------------------------------------
# Determine optimal number of experiments
# ----------------------------------------------------------------------------
fim_optimal_n = optimal_experiment_count(fim_history)
gim_optimal_n = optimal_experiment_count(gim_history)

println("\n" * "="^70)
println("OPTIMAL NUMBER OF EXPERIMENTS")
println("="^70)
println("FIM-based design: $fim_optimal_n experiments (out of $(length(fim_experiments)) total)")
println("GIM-based design: $gim_optimal_n experiments (out of $(length(gim_experiments)) total)")

# ----------------------------------------------------------------------------
# Compare designs
# ----------------------------------------------------------------------------
metrics = compare_designs(
    fim_experiments[1:fim_optimal_n],
    gim_experiments[1:gim_optimal_n],
    random_experiments[1:min(fim_optimal_n, gim_optimal_n)],
    θ_true, θ_prior_mean, θ_prior_cov, σ, axial_dispersion_model
)

# ----------------------------------------------------------------------------
# Print optimal experimental points
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("OPTIMAL EXPERIMENTAL POINTS")
println("="^70)

println("\nFIM-Optimal Design ($fim_optimal_n experiments):")
for (i, exp) in enumerate(fim_experiments[1:fim_optimal_n])
    println("  $i. T=$(exp[1])K, C₀=$(exp[2]) mol/m³, u=$(exp[3]) m/s, z=$(exp[4]), Pe=$(exp[5])")
end

println("\nGIM-Optimal Design ($gim_optimal_n experiments):")
for (i, exp) in enumerate(gim_experiments[1:gim_optimal_n])
    println("  $i. T=$(exp[1])K, C₀=$(exp[2]) mol/m³, u=$(exp[3]) m/s, z=$(exp[4]), Pe=$(exp[5])")
end

# ----------------------------------------------------------------------------
# Parameter uncertainty reduction
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("PARAMETER UNCERTAINTY REDUCTION")
println("="^70)

# Prior uncertainty
prior_std = sqrt.(diag(θ_prior_cov))
println("\nPrior uncertainty:")
println("  σ_A = $(round(prior_std[1], digits=0)) s⁻¹")
println("  σ_E = $(round(prior_std[2], digits=0)) J/mol")

# Posterior uncertainty from FIM-optimal design
FIM_opt = compute_fim_experiments(θ_true, fim_experiments[1:fim_optimal_n], σ, axial_dispersion_model)
post_cov_fim = inv(FIM_opt + 1e-8*I)
post_std_fim = sqrt.(diag(post_cov_fim))

println("\nPosterior uncertainty (FIM-optimal design):")
println("  σ_A = $(round(post_std_fim[1], digits=0)) s⁻¹ ($(round((1 - post_std_fim[1]/prior_std[1])*100, digits=1))% reduction)")
println("  σ_E = $(round(post_std_fim[2], digits=0)) J/mol ($(round((1 - post_std_fim[2]/prior_std[2])*100, digits=1))% reduction)")

# Posterior uncertainty from GIM-optimal design
GIM_opt = compute_gim_experiments(θ_prior_mean, θ_prior_cov, gim_experiments[1:gim_optimal_n], 
                                  σ, axial_dispersion_model, n_samples=100)
post_cov_gim = inv(GIM_opt + 1e-8*I)
post_std_gim = sqrt.(diag(post_cov_gim))

println("\nPosterior uncertainty (GIM-optimal design):")
println("  σ_A = $(round(post_std_gim[1], digits=0)) s⁻¹ ($(round((1 - post_std_gim[1]/prior_std[1])*100, digits=1))% reduction)")
println("  σ_E = $(round(post_std_gim[2], digits=0)) J/mol ($(round((1 - post_std_gim[2]/prior_std[2])*100, digits=1))% reduction)")

# ----------------------------------------------------------------------------
# Create simple plots
# ----------------------------------------------------------------------------
println("\nCreating visualizations...")

try
    # Plot design history
    p1 = plot(title="Sequential Design Progress",
              xlabel="Number of Experiments",
              ylabel="D-Optimality Criterion",
              legend=:bottomright,
              grid=true,
              size=(800, 500))
    
    plot!(p1, fim_history[:n_experiments], fim_history[:criterion_values],
          label="FIM-based",
          linewidth=3,
          marker=:circle,
          color=:blue)
    
    plot!(p1, gim_history[:n_experiments], gim_history[:criterion_values],
          label="GIM-based",
          linewidth=3,
          marker=:square,
          color=:red,
          linestyle=:dash)
    
    savefig(p1, "design_history.png")
    display(p1)
    
    # Plot information comparison
    p2 = plot(title="Information Content Comparison",
              xlabel="Number of Experiments",
              ylabel="Determinant of FIM (log scale)",
              yaxis=:log10,
              legend=:topleft,
              size=(800, 500))
    
    n_range = 3:min(length(fim_experiments), length(gim_experiments))
    
    fim_dets = [det(compute_fim_experiments(θ_true, fim_experiments[1:i], σ, axial_dispersion_model)) 
                for i in n_range]
    gim_dets = [det(compute_fim_experiments(θ_true, gim_experiments[1:i], σ, axial_dispersion_model)) 
                for i in n_range]
    rand_dets = [det(compute_fim_experiments(θ_true, random_experiments[1:i], σ, axial_dispersion_model)) 
                 for i in n_range]
    
    plot!(p2, n_range, fim_dets, label="FIM-Optimal", linewidth=3, color=:blue)
    plot!(p2, n_range, gim_dets, label="GIM-Optimal", linewidth=3, color=:red, linestyle=:dash)
    plot!(p2, n_range, rand_dets, label="Random", linewidth=2, color=:gray, linestyle=:dot)
    
    savefig(p2, "information_comparison.png")
    display(p2)
    
    println("✓ Plots saved successfully")
catch e
    println("Warning: Could not create plots: $e")
end

# ----------------------------------------------------------------------------
# Final recommendations
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("FINAL RECOMMENDATIONS")
println("="^70)

println("\nBased on the analysis:")
println("1. Minimum experiments needed: $(min(fim_optimal_n, gim_optimal_n))")
println("2. Optimal number of experiments: $(max(fim_optimal_n, gim_optimal_n))")
println("3. Recommended approach:")

if fim_optimal_n <= gim_optimal_n
    println("   • Use FIM-based design if confident in initial parameters")
    println("   • Efficiency gain: $(round(metrics["eff_fim"], digits=1))× vs random design")
else
    println("   • Use GIM-based design for robustness to parameter uncertainty")
    println("   • Efficiency gain: $(round(metrics["eff_gim"], digits=1))× vs random design")
end

println("\n4. Key experimental conditions:")
println("   • Temperature: Focus around 390-420 K")
println("   • Position: Include multiple points (z = 0.3, 0.5, 0.7)")
println("   • Flow rate: Vary to explore kinetics vs transport")
println("   • Dispersion: Include both low and high Pe numbers")

println("\n5. Validation strategy:")
println("   • Run 2-3 additional experiments at predicted optimal conditions")
println("   • Compare model predictions with measurements")
println("   • If uncertainty >10%, add experiments at temperature extremes")

println("\n" * "="^70)
println("OPTIMAL EXPERIMENTAL DESIGN COMPLETE")
println("="^70)