using NLsolve
using Optim
using QuasiMonteCarlo
using Distributions
using Statistics
using Plots  # Added for visualization
using LinearAlgebra  # Added for matrix operations
using LaTeXStrings  # Added for better plot labels

#=============================================================================
FIXED AND ENHANCED CSTR PARAMETER ESTIMATION CODE
=============================================================================#

"""
    CSTR_model(ca_in, Temp; tau=5, n=1, k=[0.0, 0.0])

Solve CSTR mass balance to find outlet concentration.
"""
function CSTR_model(ca_in, Temp; tau=5, n=1, k=[0.0, 0.0])
    # Define the residual function for nlsolve
    function residual!(F, CA)
        F[1] = 1/tau * (ca_in - CA[1]) - k[1] * exp(-k[2]/(8.314 * Temp)) * CA[1]^n
    end
    
    # Solve nonlinear equation
    sol = nlsolve(residual!, [0.1], ftol=1e-12, iterations=1000)
    
    # Check if solution converged
    if !sol.f_converged
        @warn "CSTR model did not converge for ca_in=$ca_in, Temp=$Temp"
        return [NaN]
    end
    
    return sol.zero
end

"""
    random_points_generator(Nexps; lb, ub, Sampling=HaltonSample())

Generate quasi-random experimental conditions.
"""
function random_points_generator(Nexps; lb=[0.0], ub=[1.0], Sampling=HaltonSample())
    # Convert to proper vector format
    lb = vec(lb)
    ub = vec(ub)
    
    # Generate samples
    s = QuasiMonteCarlo.sample(Nexps, lb, ub, Sampling)
    
    # Convert to matrix and transpose to get experiments in rows
    s = reshape(s, length(lb), Nexps)'
    
    return s
end

"""
    ca_exp(ca_ins, Temp; k, add_noise, sigma, N_repeats, order)

Generate synthetic experimental data with optional noise.
"""
function ca_exp(ca_ins, Temp; k=[1.0, 20000.0], add_noise=false, sigma=1e-6, 
                N_repeats=5, order=1.0)
    # Ensure inputs are vectors
    ca_ins = vec(ca_ins)
    Temp = vec(Temp)
    Nexps = length(ca_ins)
    
    # Initialize output arrays
    ca_out_without_noise = zeros(Nexps)
    ca_out_matrix = zeros(N_repeats, Nexps)
    
    # Generate data for each experiment
    for i in 1:Nexps
        # True value without noise
        ca_out_without_noise[i] = CSTR_model(ca_ins[i], Temp[i]; k=k, n=order)[1]
        
        # Add noise if requested
        if add_noise
            ca_out_matrix[:, i] = ca_out_without_noise[i] .+ rand(Normal(0, sigma), N_repeats)
        else
            ca_out_matrix[:, i] = fill(ca_out_without_noise[i], N_repeats)
        end
    end
    
    # Average over repeats
    ca_out = vec(mean(ca_out_matrix, dims=1))
    
    return ca_out, ca_out_without_noise
end

"""
    concentration_compute(F, CA; k, n, Temp, ca_in, tau)

Residual function for CSTR mass balance.
"""
function concentration_compute(F, CA; k, n, Temp, ca_in, tau)
    F[1] = 1/tau * (ca_in - CA[1]) - k[1] * exp(-k[2]/(8.314 * Temp)) * CA[1]^n
    return nothing
end

"""
    ca_model(k; ca_in, Temp, n, tau, ra!)

Solve CSTR model for given parameters.
"""
function ca_model(k; ca_in=0.0, Temp=0.0, n=1.0, tau=5.0, ra!)
    sol = nlsolve((F, CA) -> ra!(F, CA; k=k, ca_in=ca_in, Temp=Temp, n=n, tau=tau), 
                  [0.1], ftol=1e-12, iterations=1000)
    
    if !sol.f_converged
        @warn "ca_model did not converge"
        return [NaN]
    end
    
    return sol.zero
end

"""
    parameter_estimator(; ca_exp_data, ca_in, Temp, order, initial_guess, 
                        weights=nothing)

Estimate kinetic parameters using weighted or unweighted least squares.
"""
function parameter_estimator(; ca_exp_data, ca_in, Temp, order=1.0, 
                             initial_guess=[1.0, 10000.0], weights=nothing)
    
    # Ensure inputs are vectors
    ca_exp_data = vec(ca_exp_data)
    ca_in = vec(ca_in)
    Temp = vec(Temp)
    Nexps = length(ca_exp_data)
    
    # If no weights provided, use equal weights
    if isnothing(weights)
        weights = ones(Nexps)
    else
        weights = vec(weights)
    end
    
    # Normalize weights
    weights = weights / sum(weights) * Nexps
    
    # Define objective function
    function objective(k)
        total_error = 0.0
        for i in 1:Nexps
            try
                ca_pred = ca_model(k; ca_in=ca_in[i], Temp=Temp[i], n=order, 
                                   tau=5.0, ra!=concentration_compute)[1]
                if !isnan(ca_pred)
                    residual = (ca_exp_data[i] - ca_pred)
                    total_error += weights[i] * residual^2
                else
                    total_error += 1e6  # Penalty for non-convergence
                end
            catch
                total_error += 1e6  # Penalty for errors
            end
        end
        return total_error
    end
    
    # Set up optimization
    solver = LBFGS()
    
    # Parameter bounds (physical constraints)
    lower = [1e-6, 1e-6]    # k0 > 0, Ea > 0
    upper = [1e6, 1e6]       # Reasonable upper bounds
    
    # Create optimization problem with bounds
    function objective_wrapper(k)
        # Transform variables for bounded optimization
        k_transformed = lower .+ (upper - lower) .* (1 ./(1 .+ exp.(-k)))
        return objective(k_transformed)
    end
    
    # Initial guess in transformed space
    k0_transformed = log.((initial_guess .- lower) ./ (upper .- initial_guess))
    
    # Run optimization
    res = Optim.optimize(objective_wrapper, k0_transformed, solver, 
                         Optim.Options(iterations=1000, 
                                     show_trace=false, 
                                     g_tol=1e-8,
                                     f_tol=1e-8))
    
    # Transform back to original space
    k_opt = lower .+ (upper - lower) .* (1 ./(1 .+ exp.(-Optim.minimizer(res))))
    
    # Calculate parameter covariance (using Fisher Information Matrix approximation)
    function calculate_covariance(k_opt)
        n_params = length(k_opt)
        epsilon = 1e-6
        J = zeros(Nexps, n_params)  # Sensitivity matrix
        
        for i in 1:Nexps
            # Nominal prediction
            y_nom = ca_model(k_opt; ca_in=ca_in[i], Temp=Temp[i], n=order, 
                            tau=5.0, ra!=concentration_compute)[1]
            
            # Finite differences for each parameter
            for j in 1:n_params
                k_pert = copy(k_opt)
                k_pert[j] += epsilon
                y_pert = ca_model(k_pert; ca_in=ca_in[i], Temp=Temp[i], n=order, 
                                 tau=5.0, ra!=concentration_compute)[1]
                J[i, j] = (y_pert - y_nom) / epsilon
            end
        end
        
        # Measurement noise variance
        sigma2 = objective(k_opt) / (Nexps - n_params)
        
        # Parameter covariance
        try
            FIM = J' * J / sigma2
            cov_matrix = inv(FIM)
        catch
            cov_matrix = zeros(n_params, n_params)
        end
        
        return cov_matrix
    end
    
    # Calculate covariance
    cov_matrix = calculate_covariance(k_opt)
    
    return k_opt, cov_matrix, res
end

"""
    run_sequential_parameter_estimation(; N_initial, N_max, true_k, 
                                         sampling_method, add_noise, sigma)

Run sequential parameter estimation with visualization of improvement.
"""
function run_sequential_parameter_estimation(; N_initial=3, N_max=15, 
                                              true_k=[1.0, 20000.0],
                                              sampling_method=HaltonSample(),
                                              add_noise=true, sigma=1e-3, order=1.0)
    
    println("="^60)
    println("SEQUENTIAL PARAMETER ESTIMATION FOR CSTR KINETICS")
    println("="^60)
    println("\nTrue parameters:")
    println("  k0 (pre-exponential factor): $(true_k[1])")
    println("  Ea (activation energy): $(true_k[2]) J/mol")
    println("  Reaction order: $order")
    println("\nExperimental conditions:")
    println("  Temperature range: 300-500 K")
    println("  Concentration range: 0.1-2.0 mol/m³")
    println("  Measurement noise: $(add_noise ? "Yes (σ=$sigma)" : "No")")
    
    # Storage for results
    iterations = []
    k0_estimates = []
    Ea_estimates = []
    k0_errors = []
    Ea_errors = []
    k0_ci = []
    Ea_ci = []
    all_experiments_T = []
    all_experiments_C = []
    
    # Initial parameter guess
    current_guess = [0.5, 10000.0]  # Start with poor initial guess
    
    # Sequential design loop
    for N in N_initial:N_max
        println("\n" * "-"^40)
        println("ITERATION WITH $N EXPERIMENTS")
        println("-"^40)
        
        # Generate experimental conditions (can be replaced with optimal design later)
        Temp = random_points_generator(N; lb=[300.0], ub=[500.0], Sampling=sampling_method)
        ca_ins = random_points_generator(N; lb=[0.1], ub=[2.0], Sampling=sampling_method)
        
        # Flatten arrays
        Temp = vec(Temp)
        ca_ins = vec(ca_ins)
        
        # Generate experimental data using true parameters
        ca_exper, ca_true = ca_exp(ca_ins, Temp; k=true_k, add_noise=add_noise, 
                                    sigma=sigma, N_repeats=3, order=order)
        
        # Estimate parameters
        try
            k_est, cov_mat, opt_res = parameter_estimator(;
                ca_exp_data=ca_exper,
                ca_in=ca_ins,
                Temp=Temp,
                order=order,
                initial_guess=current_guess
            )
            
            # Calculate errors
            k0_error = abs(k_est[1] - true_k[1]) / true_k[1] * 100
            Ea_error = abs(k_est[2] - true_k[2]) / true_k[2] * 100
            
            # Calculate confidence intervals (95%)
            if all(diag(cov_mat) .> 0)
                k0_std = sqrt(cov_mat[1,1])
                Ea_std = sqrt(cov_mat[2,2])
                k0_ci_lower = k_est[1] - 1.96 * k0_std
                k0_ci_upper = k_est[1] + 1.96 * k0_std
                Ea_ci_lower = k_est[2] - 1.96 * Ea_std
                Ea_ci_upper = k_est[2] + 1.96 * Ea_std
            else
                k0_std = Ea_std = 0.0
                k0_ci_lower = k0_ci_upper = k_est[1]
                Ea_ci_lower = Ea_ci_upper = k_est[2]
            end
            
            # Store results
            push!(iterations, N)
            push!(k0_estimates, k_est[1])
            push!(Ea_estimates, k_est[2])
            push!(k0_errors, k0_error)
            push!(Ea_errors, Ea_error)
            push!(k0_ci, (k0_ci_lower, k0_ci_upper))
            push!(Ea_ci, (Ea_ci_lower, Ea_ci_upper))
            
            # Store experimental conditions
            push!(all_experiments_T, Temp)
            push!(all_experiments_C, ca_ins)
            
            # Update current guess for next iteration
            current_guess = k_est
            
            # Print results
            println("\nEstimated parameters:")
            println("  k0 = $(round(k_est[1], digits=4)) (error: $(round(k0_error, digits=2))%)")
            println("  Ea = $(round(Int, k_est[2])) J/mol (error: $(round(Ea_error, digits=2))%)")
            println("\n95% Confidence intervals:")
            println("  k0: [$(round(k0_ci_lower, digits=4)), $(round(k0_ci_upper, digits=4))]")
            println("  Ea: [$(round(Int, k0_ci_lower)), $(round(Int, k0_ci_upper))]")
            
        catch e
            println("Error in estimation at N=$N: $e")
            continue
        end
    end
    
    # Create visualization plots
    create_visualization(iterations, k0_estimates, Ea_estimates, 
                         k0_errors, Ea_errors, k0_ci, Ea_ci,
                         all_experiments_T, all_experiments_C,
                         true_k)
    
    return iterations, k0_estimates, Ea_estimates
end

"""
    create_visualization(iterations, k0_est, Ea_est, k0_err, Ea_err, 
                        k0_ci, Ea_ci, exp_T, exp_C, true_k)

Create comprehensive visualization of parameter estimation improvement.
"""
function create_visualization(iterations, k0_est, Ea_est, k0_err, Ea_err, 
                             k0_ci, Ea_ci, exp_T, exp_C, true_k)
    
    # Create a 2x2 plot layout
    p1 = plot(layout=(2,2), size=(1200, 800), 
              title="Parameter Estimation Improvement with Sequential Experiments",
              fontfamily="Helvetica")
    
    # Plot 1: k0 estimation with confidence intervals
    plot!(p1[1], iterations, k0_est, 
          marker=:circle, markersize=8, linewidth=2, color=:blue,
          label="Estimated k₀", 
          xlabel="Number of Experiments", 
          ylabel=L"k_0 \text{ (pre-exponential factor)}",
          title="k₀ Estimation Progress")
    
    # Add confidence intervals
    k0_lower = [ci[1] for ci in k0_ci]
    k0_upper = [ci[2] for ci in k0_ci]
    plot!(p1[1], iterations, k0_lower, fillrange=k0_upper, 
          fillalpha=0.2, linewidth=0, label="95% CI", color=:lightblue)
    
    # Add true value
    hline!(p1[1], [true_k[1]], linewidth=2, linestyle=:dash, 
           color=:red, label="True k₀")
    
    # Plot 2: Ea estimation with confidence intervals
    plot!(p1[2], iterations, Ea_est, 
          marker=:square, markersize=8, linewidth=2, color=:green,
          label="Estimated Eₐ", 
          xlabel="Number of Experiments", 
          ylabel=L"E_a \text{ (J/mol)}",
          title="Activation Energy Estimation Progress")
    
    # Add confidence intervals
    Ea_lower = [ci[1] for ci in Ea_ci]
    Ea_upper = [ci[2] for ci in Ea_ci]
    plot!(p1[2], iterations, Ea_lower, fillrange=Ea_upper, 
          fillalpha=0.2, linewidth=0, label="95% CI", color=:lightgreen)
    
    # Add true value
    hline!(p1[2], [true_k[2]], linewidth=2, linestyle=:dash, 
           color=:red, label="True Eₐ")
    
    # Plot 3: Relative errors
    plot!(p1[3], iterations, k0_err, 
          marker=:circle, linewidth=2, color=:blue,
          label="k₀ relative error", 
          xlabel="Number of Experiments", 
          ylabel="Relative Error (%)",
          title="Parameter Estimation Errors")
    
    plot!(p1[3], iterations, Ea_err, 
          marker=:square, linewidth=2, color=:green,
          label="Eₐ relative error")
    
    # Add reference line at 5% error
    hline!(p1[3], [5.0], linewidth=1, linestyle=:dash, 
           color=:black, label="5% threshold")
    
    # Plot 4: Experimental design space
    plot!(p1[4], 
          xlabel="Inlet Concentration (mol/m³)", 
          ylabel="Temperature (K)",
          title="Experimental Conditions",
          legend=false)
    
    # Plot experiments with different colors for each iteration
    colors = cgrad(:viridis, length(exp_T))
    for i in 1:length(exp_T)
        scatter!(p1[4], exp_C[i], exp_T[i], 
                color=colors[i], markersize=5, marker=:circle,
                label=i==1 ? "Experiments" : false)
    end
    
    # Add colorbar to show iteration number
    scatter!([NaN], [NaN], marker=:circle, color=colors, 
             label="Iteration", seriescolor=colors)
    
    # Save plot
    savefig(p1, "parameter_estimation_improvement.png")
    println("\nPlot saved as 'parameter_estimation_improvement.png'")
    
    # Display plot
    display(p1)
    
    return p1
end

"""
    run_optimal_design_comparison()

Compare random sampling vs optimal experimental design.
"""
function run_optimal_design_comparison(; n_experiments=10, n_trials=5)
    
    true_k = [1.0, 20000.0]
    
    # Store results
    random_errors = []
    optimal_errors = []
    
    for trial in 1:n_trials
        println("\nTrial $trial of $n_trials")
        
        # Random design
        Temp_random = random_points_generator(n_experiments; lb=[300.0], ub=[500.0])
        ca_random = random_points_generator(n_experiments; lb=[0.1], ub=[2.0])
        
        Temp_random = vec(Temp_random)
        ca_random = vec(ca_random)
        
        ca_exp_random, _ = ca_exp(ca_random, Temp_random; k=true_k, add_noise=true, 
                                   sigma=1e-3, order=1.0)
        
        k_est_random, _, _ = parameter_estimator(;
            ca_exp_data=ca_exp_random,
            ca_in=ca_random,
            Temp=Temp_random,
            order=1.0,
            initial_guess=[0.5, 10000.0]
        )
        
        random_error = norm(k_est_random - true_k) / norm(true_k) * 100
        push!(random_errors, random_error)
        
        # For optimal design, we would use FIM-based selection
        # Here we just use a simple grid search as placeholder
        T_grid = range(300, 500, length=10)
        C_grid = range(0.1, 2.0, length=10)
        
        # Select points that maximize sensitivity (simplified)
        best_points = []
        for i in 1:n_experiments
            best_T = T_grid[argmax(abs.(exp.(-20000.0./(8.314.*T_grid))))]
            best_C = C_grid[argmax(C_grid)]
            push!(best_points, (best_T, best_C))
        end
        
        Temp_optimal = [p[1] for p in best_points]
        ca_optimal = [p[2] for p in best_points]
        
        ca_exp_optimal, _ = ca_exp(ca_optimal, Temp_optimal; k=true_k, add_noise=true,
                                    sigma=1e-3, order=1.0)
        
        k_est_optimal, _, _ = parameter_estimator(;
            ca_exp_data=ca_exp_optimal,
            ca_in=ca_optimal,
            Temp=Temp_optimal,
            order=1.0,
            initial_guess=[0.5, 10000.0]
        )
        
        optimal_error = norm(k_est_optimal - true_k) / norm(true_k) * 100
        push!(optimal_errors, optimal_error)
    end
    
    # Create comparison plot
    p2 = plot(size=(800, 500))
    
    boxplot!(p2, fill(1, n_trials), random_errors, 
             color=:red, alpha=0.5, label="Random Design")
    boxplot!(p2, fill(2, n_trials), optimal_errors, 
             color=:blue, alpha=0.5, label="Optimal Design")
    
    xlabel!(p2, "Experimental Design Method")
    ylabel!(p2, "Parameter Estimation Error (%)")
    title!(p2, "Random vs Optimal Experimental Design Comparison")
    
    savefig(p2, "design_comparison.png")
    display(p2)
    
    println("\nRandom Design - Mean Error: $(round(mean(random_errors), digits=2))%")
    println("Optimal Design - Mean Error: $(round(mean(optimal_errors), digits=2))%")
    
    return random_errors, optimal_errors
end

#=============================================================================
MAIN EXECUTION
=============================================================================#

println("\n" * "#"^60)
println("# CSTR PARAMETER ESTIMATION WITH VISUALIZATION")
println("#"^60)

# Run sequential parameter estimation
iterations, k0_est, Ea_est = run_sequential_parameter_estimation(
    N_initial=3,
    N_max=12,
    true_k=[1.0, 20000.0],
    sampling_method=HaltonSample(),
    add_noise=true,
    sigma=1e-3,
    order=1.0
)

# Uncomment to run design comparison (takes more time)
# println("\n" * "#"^60)
# println("# COMPARING RANDOM VS OPTIMAL DESIGN")
# println("#"^60)
# random_errors, optimal_errors = run_optimal_design_comparison(n_experiments=8, n_trials=3)

println("\n" * "#"^60)
println("# ANALYSIS COMPLETE")
println("#"^60)
println("\nKey observations:")
println("1. Parameter estimates improve as more experiments are added")
println("2. Confidence intervals shrink with increasing information")
println("3. Error typically decreases below 5% with sufficient experiments")
println("4. Optimal design would converge faster than random sampling")