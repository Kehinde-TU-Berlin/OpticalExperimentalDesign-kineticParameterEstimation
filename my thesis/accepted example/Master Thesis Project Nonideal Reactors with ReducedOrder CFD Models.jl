#= 
OPTIMAL EXPERIMENTAL DESIGN FOR KINETIC PARAMETER ESTIMATION
Using FIM (Fisher Information Matrix) and GIM (General Information Matrix)
Author: [Your Name]
Date: 2024

This code implements sequential optimal experimental design for estimating
kinetic parameters in non-ideal reactors using a reduced-order CFD model.
=#

# PACKAGE INSTALLATION AND SETUP
# First, let's install and load all necessary packages
using Pkg
using Pkg
# Ensure all necessary tools are installed
Pkg.add(["Plots", "LinearAlgebra", "Distributions", "Random", "Measures", "Optim"])
Pkg.add(["DifferentialEquations"])
Pkg.gc()
Pkg.add("OrdinaryDiffEq")
versioninfo()
using Pkg
Pkg.activate("ThesisEnv")
Pkg.update()
#= 
Installing required packages:
- DifferentialEquations: For solving ODE/PDE systems
- Optim: For parameter estimation and optimization
- LinearAlgebra: For matrix operations
- Plots: For visualization
- Statistics: For statistical computations
- Distributions: For probability distributions
=#

# Check and install packages if not already installed
packages = ["DifferentialEquations", "Optim", "LinearAlgebra", 
            "Plots", "Statistics", "Distributions", "ForwardDiff"]

for pkg in packages
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

# Load all packages
using DifferentialEquations  # For solving kinetic models
using Optim                  # For parameter estimation
using LinearAlgebra          # For matrix operations
using Plots                   # For visualization
using Statistics              # For statistical calculations
using Distributions           # For confidence intervals
using ForwardDiff             # For automatic differentiation

#=============================================================================
PART 1: MODEL DEFINITION
=============================================================================#

#= 
We define a reduced-order model for a non-ideal reactor. 
The model incorporates:
- Axial dispersion (characterized by Peclet number)
- Reaction kinetics (power-law model)
- Temperature and concentration dependence

Parameters to estimate: 
- k0 (pre-exponential factor)
- Ea (activation energy)
- n (reaction order)
=#

"""
    reactor_model!(du, u, p, z)

Reduced-order model for non-ideal reactor with axial dispersion.
This is a 1D steady-state model along the reactor length.

Arguments:
- du: Derivatives vector (output)
- u: State variables [C, T] concentration and temperature
- p: Model parameters [k0, Ea, n, Pe, L, R, rho_cp]
- z: Spatial coordinate (0 to L)

The model equation:
d²C/dz² = Pe * (dC/dz) + Pe * r(C,T) * (L²/D)
where r(C,T) is the reaction rate
"""
function reactor_model!(du, u, p, z)
    # Unpack state variables
    C, T = u  # Concentration and temperature at position z
    
    # Unpack parameters
    k0, Ea, n, Pe, L, R, rho_cp = p
    
    #= 
    Reaction rate using Arrhenius equation:
    r = k0 * exp(-Ea/(R*T)) * C^n
    =#
    r_rate = k0 * exp(-Ea/(R*T)) * C^n
    
    #= 
    Dimensionless equations for axial dispersion model:
    First derivative: dC/dz
    Second derivative: d²C/dz² = Pe * (dC/dz - r_rate)
    =#
    du[1] = C  # Store C for second derivative calculation
    du[2] = Pe * (C - r_rate)  # Simplified model for demonstration
    
    return nothing
end

"""
    solve_reactor(p, C0, T0, z_span)

Solve the reactor model for given parameters and inlet conditions.

Arguments:
- p: Parameter vector [k0, Ea, n, Pe, L, R, rho_cp]
- C0: Inlet concentration
- T0: Inlet temperature
- z_span: Spatial domain [0, L]

Returns:
- Solution object containing concentration and temperature profiles
"""
function solve_reactor(p, C0, T0, z_span)
    # Initial condition [C, T]
    u0 = [C0, T0]
    
    # Define the ODE problem
    prob = ODEProblem(reactor_model!, u0, z_span, p)
    
    # Solve the ODE
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    
    return sol
end

#=============================================================================
PART 2: FISHER INFORMATION MATRIX (FIM) CALCULATION
=============================================================================#

"""
    calculate_fim(parameters, experimental_conditions, measured_outputs)

Calculate the Fisher Information Matrix for given experimental conditions.

The FIM is defined as: FIM = J^T * Σ^(-1) * J
where:
- J is the sensitivity matrix (∂y/∂θ)
- Σ is the measurement error covariance matrix

Arguments:
- parameters: Current parameter estimates [k0, Ea, n]
- exp_conditions: Matrix of experimental conditions [T, C0] for each experiment
- measured_outputs: Measured outputs at reactor outlet

Returns:
- FIM: Fisher Information Matrix
- parameter_covariance: Covariance matrix of parameter estimates
"""
function calculate_fim(parameters, exp_conditions, measured_outputs)
    # Extract current parameter values
    k0, Ea, n = parameters
    Pe = 100.0  # Fixed Peclet number (constant)
    L = 1.0     # Reactor length (constant)
    R = 8.314   # Gas constant (fixed)
    rho_cp = 1000.0  # Heat capacity (fixed)
    
    # Full parameter vector including fixed parameters
    full_params = [k0, Ea, n, Pe, L, R, rho_cp]
    
    # Number of experiments
    n_exp = size(exp_conditions, 1)
    
    #= 
    Calculate sensitivity matrix using finite differences
    We perturb each parameter slightly to compute ∂y/∂θ
    =#
    n_params = length(parameters)
    n_outputs = 2  # We measure both C and T at outlet
    
    # Initialize sensitivity matrix (n_outputs * n_exp rows, n_params columns)
    S = zeros(n_exp * n_outputs, n_params)
    
    # Perturbation size for finite differences
    ϵ = 1e-6
    
    # Loop through each experiment
    for i in 1:n_exp
        # Get experimental conditions for this run
        T_exp = exp_conditions[i, 1]  # Temperature
        C0_exp = exp_conditions[i, 2]  # Inlet concentration
        
        # Solve model with nominal parameters
        sol_nominal = solve_reactor(full_params, C0_exp, T_exp, (0.0, L))
        y_nominal = [sol_nominal(L)[1], sol_nominal(L)[2]]  # C and T at outlet
        
        # Calculate sensitivities for each parameter
        for j in 1:n_params
            # Perturb parameter j
            params_perturbed = copy(full_params)
            params_perturbed[j] += ϵ
            
            # Solve with perturbed parameters
            sol_perturbed = solve_reactor(params_perturbed, C0_exp, T_exp, (0.0, L))
            y_perturbed = [sol_perturbed(L)[1], sol_perturbed(L)[2]]
            
            # Finite difference approximation of sensitivity
            sensitivity = (y_perturbed - y_nominal) / ϵ
            
            # Store in sensitivity matrix
            row_start = (i-1) * n_outputs + 1
            S[row_start:row_start+n_outputs-1, j] = sensitivity
        end
    end
    
    #= 
    Measurement error covariance matrix
    Assume independent measurements with variance based on measurement noise
    =#
    σ_C = 0.01 * mean(abs.(measured_outputs[:, 1]))  # 1% error in concentration
    σ_T = 0.01 * mean(abs.(measured_outputs[:, 2]))  # 1% error in temperature
    
    Σ_inv = diagm(repeat([1/σ_C^2, 1/σ_T^2], n_exp))
    
    # Calculate Fisher Information Matrix
    FIM = S' * Σ_inv * S
    
    # Parameter covariance matrix (inverse of FIM)
    try
        param_covariance = inv(FIM)
    catch
        # If FIM is singular, add small regularization
        param_covariance = inv(FIM + 1e-6 * I)
    end
    
    return FIM, param_covariance
end

#=============================================================================
PART 3: GENERAL INFORMATION MATRIX (GIM) AND EXPERIMENTAL DESIGN
=============================================================================#

"""
    calculate_gim(parameters, exp_conditions, measured_outputs, prior_info)

Calculate the General Information Matrix which combines prior knowledge
with experimental data.

GIM = FIM + Prior_information

Arguments:
- parameters: Current parameter estimates
- exp_conditions: Experimental conditions matrix
- measured_outputs: Measured outputs
- prior_info: Prior covariance matrix (from previous experiments)

Returns:
- GIM: General Information Matrix
"""
function calculate_gim(parameters, exp_conditions, measured_outputs, prior_info)
    # Calculate FIM for current experimental design
    FIM, _ = calculate_fim(parameters, exp_conditions, measured_outputs)
    
    #= 
    GIM combines Fisher information with prior knowledge
    Prior information matrix is inverse of prior covariance
    =#
    if isnothing(prior_info)
        # No prior information, use only FIM
        GIM = FIM
        prior_inv = zeros(size(FIM))
    else
        # Add prior precision matrix to FIM
        prior_precision = inv(prior_info)
        GIM = FIM + prior_precision
        prior_inv = prior_info
    end
    
    return GIM, prior_inv
end

"""
    design_criterion(FIM, criterion_type)

Calculate optimal design criterion based on FIM.

Available criteria:
- D-optimal: Maximize determinant of FIM (minimize volume of confidence ellipsoid)
- A-optimal: Minimize trace of inverse FIM (minimize average variance)
- E-optimal: Maximize minimum eigenvalue (minimize maximum variance)

Arguments:
- FIM: Fisher Information Matrix
- criterion_type: String specifying criterion ("D", "A", or "E")

Returns:
- criterion_value: Value of the design criterion
"""
function design_criterion(FIM, criterion_type)
    if criterion_type == "D"
        # D-optimal: maximize |FIM|
        return det(FIM)
    elseif criterion_type == "A"
        # A-optimal: minimize trace(FIM^(-1))
        return -trace(inv(FIM))  # Negative for maximization
    elseif criterion_type == "E"
        # E-optimal: maximize minimum eigenvalue
        return minimum(eigvals(FIM))
    else
        error("Unknown criterion type. Use 'D', 'A', or 'E'")
    end
end

"""
    optimize_experimental_design(current_params, exp_domain, n_candidates, criterion)

Find optimal experimental conditions for next experiment.

Arguments:
- current_params: Current parameter estimates
- exp_domain: Domain boundaries [T_min, T_max; C_min, C_max]
- n_candidates: Number of candidate points to evaluate
- criterion: Design criterion to use

Returns:
- optimal_conditions: Best experimental conditions [T, C0]
- criterion_value: Value of criterion at optimal point
"""
function optimize_experimental_design(current_params, exp_domain, n_candidates, criterion)
    # Generate candidate experimental conditions
    T_range = range(exp_domain[1, 1], exp_domain[1, 2], length=Int(sqrt(n_candidates)))
    C_range = range(exp_domain[2, 1], exp_domain[2, 2], length=Int(sqrt(n_candidates)))
    
    # Initialize storage for criterion values
    criterion_values = zeros(length(T_range), length(C_range))
    
    #= 
    For each candidate condition, calculate the FIM and design criterion
    We assume no measurements yet for these candidate experiments
    =#
    for (i, T) in enumerate(T_range)
        for (j, C0) in enumerate(C_range)
            # Create temporary experimental condition
            temp_conditions = [T C0]
            
            # Dummy measured outputs (will be replaced by model predictions)
            dummy_outputs = zeros(1, 2)
            
            # Calculate FIM for this candidate
            FIM, _ = calculate_fim(current_params, temp_conditions, dummy_outputs)
            
            # Calculate design criterion
            criterion_values[i, j] = design_criterion(FIM, criterion)
        end
    end
    
    # Find optimal condition
    opt_idx = argmax(criterion_values[:])
    i_opt, j_opt = ind2sub((length(T_range), length(C_range)), opt_idx)
    
    optimal_conditions = [T_range[i_opt], C_range[j_opt]]
    
    return optimal_conditions, criterion_values[i_opt, j_opt]
end

#=============================================================================
PART 4: PARAMETER ESTIMATION AND SEQUENTIAL DESIGN
=============================================================================#

"""
    estimate_parameters(exp_conditions, measurements, initial_guess)

Estimate kinetic parameters using nonlinear least squares.

Arguments:
- exp_conditions: Matrix of experimental conditions [T, C0] for each experiment
- measurements: Matrix of measured outputs [C_out, T_out] for each experiment
- initial_guess: Initial guess for parameters [k0, Ea, n]

Returns:
- estimated_params: Optimized parameter values
- parameter_covariance: Covariance matrix of estimates
- optimization_summary: Summary of optimization results
"""
function estimate_parameters(exp_conditions, measurements, initial_guess)
    # Fixed parameters
    Pe = 100.0
    L = 1.0
    R = 8.314
    rho_cp = 1000.0
    
    # Number of experiments
    n_exp = size(exp_conditions, 1)
    
    #= 
    Define objective function for optimization
    Minimize sum of squared residuals between model and measurements
    =#
    function objective(params)
        k0, Ea, n = params
        full_params = [k0, Ea, n, Pe, L, R, rho_cp]
        
        residuals = []
        for i in 1:n_exp
            T_exp = exp_conditions[i, 1]
            C0_exp = exp_conditions[i, 2]
            
            # Solve model
            sol = solve_reactor(full_params, C0_exp, T_exp, (0.0, L))
            
            # Model predictions at outlet
            C_pred = sol(L)[1]
            T_pred = sol(L)[2]
            
            # Calculate residuals
            push!(residuals, (C_pred - measurements[i, 1]) / measurements[i, 1])
            push!(residuals, (T_pred - measurements[i, 2]) / measurements[i, 2])
        end
        
        return sum(residuals.^2)
    end
    
    # Set up optimization problem
    opt = optimize(objective, initial_guess, BFGS(), 
                   Optim.Options(iterations=1000, show_trace=false))
    
    # Extract results
    estimated_params = Optim.minimizer(opt)
    
    # Calculate parameter covariance using FIM at optimum
    FIM_final, param_cov = calculate_fim(estimated_params, exp_conditions, measurements)
    
    return estimated_params, param_cov, opt
end

"""
    sequential_experimental_design(initial_params, exp_domain, n_experiments_max, 
                                   stopping_criterion, design_criterion)

Perform sequential optimal experimental design.

This function implements the iterative process:
1. Start with initial parameter guess
2. Design optimal next experiment using FIM/GIM
3. Perform experiment (simulated here)
4. Update parameter estimates
5. Repeat until stopping criterion met

Arguments:
- initial_params: Initial parameter guess [k0, Ea, n]
- exp_domain: Experimental domain [T_min T_max; C_min C_max]
- n_experiments_max: Maximum number of experiments
- stopping_criterion: Function that determines when to stop
- design_criterion: Type of optimal design criterion

Returns:
- results: Dictionary containing all results
"""
function sequential_experimental_design(initial_params, exp_domain, 
                                        n_experiments_max, stopping_criterion,
                                        design_criterion)
    
    # Initialize storage for results
    results = Dict(
        "parameters_history" => [initial_params],
        "covariance_history" => [],
        "experimental_conditions" => [],
        "measurements" => [],
        "fim_det_history" => [],
        "confidence_intervals" => [],
        "iteration" => 0
    )
    
    # Current parameters and experimental history
    current_params = copy(initial_params)
    exp_conditions = Float64[]
    measurements = Float64[]
    prior_info = nothing
    
    #= 
    True parameters for simulation (in reality, these are unknown)
    We use these to generate synthetic experimental data
    =#
    true_params = [1.0, 50000.0, 1.5]  # k0, Ea, n
    
    # Fixed parameters for simulation
    Pe = 100.0
    L = 1.0
    R = 8.314
    rho_cp = 1000.0
    full_true_params = [true_params; Pe; L; R; rho_cp]
    
    println("\n" * "="^60)
    println("STARTING SEQUENTIAL OPTIMAL EXPERIMENTAL DESIGN")
    println("="^60)
    println("\nTrue parameters (unknown to estimator):")
    println("  k0 = $(true_params[1])")
    println("  Ea = $(true_params[2]) J/mol")
    println("  n  = $(true_params[3])")
    println("\nExperimental domain:")
    println("  Temperature: $(exp_domain[1,1]) - $(exp_domain[1,2]) K")
    println("  Concentration: $(exp_domain[2,1]) - $(exp_domain[2,2]) mol/m³")
    
    # Sequential design loop
    for iter in 1:n_experiments_max
        println("\n" * "-"^40)
        println("ITERATION $iter")
        println("-"^40)
        
        #= 
        STEP 1: Design optimal experiment
        Use FIM/GIM to find conditions maximizing information
        =#
        if iter == 1
            # First experiment: Use D-optimal design without prior
            n_candidates = 100
            opt_conditions, criterion_val = optimize_experimental_design(
                current_params, exp_domain, n_candidates, design_criterion)
        else
            # Subsequent experiments: Use GIM with prior information
            # Here we use the same optimization but could modify for GIM
            opt_conditions, criterion_val = optimize_experimental_design(
                current_params, exp_domain, n_candidates, design_criterion)
        end
        
        println("\nOptimal experimental conditions:")
        println("  Temperature: $(opt_conditions[1]) K")
        println("  Concentration: $(opt_conditions[2]) mol/m³")
        println("  Design criterion value: $criterion_val")
        
        #= 
        STEP 2: Perform experiment (simulated)
        Add small random noise to true model output
        =#
        T_exp, C0_exp = opt_conditions
        
        # Solve true model
        sol_true = solve_reactor(full_true_params, C0_exp, T_exp, (0.0, L))
        C_true = sol_true(L)[1]
        T_true = sol_true(L)[2]
        
        # Add measurement noise (1% relative noise)
        C_meas = C_true * (1 + 0.01 * randn())
        T_meas = T_true * (1 + 0.0001 * randn())  # Very small temp noise
        
        println("\nExperimental measurements:")
        println("  True outlet concentration: $C_true mol/m³")
        println("  Measured concentration: $C_meas mol/m³")
        println("  True outlet temperature: $T_true K")
        println("  Measured temperature: $T_meas K")
        
        # Store experimental results
        if isempty(exp_conditions)
            exp_conditions = [opt_conditions']
            measurements = [C_meas T_meas]
        else
            exp_conditions = vcat(exp_conditions, opt_conditions')
            measurements = vcat(measurements, [C_meas T_meas])
        end
        
        #= 
        STEP 3: Update parameter estimates
        Use all available data to re-estimate parameters
        =#
        if iter > 1  # Need at least 2 experiments for estimation
            estimated_params, param_cov, opt_result = estimate_parameters(
                exp_conditions, measurements, current_params)
            
            # Update current parameters
            current_params = estimated_params
            
            println("\nUpdated parameter estimates:")
            println("  k0 = $(estimated_params[1]) (true: $(true_params[1]))")
            println("  Ea = $(estimated_params[2]) J/mol (true: $(true_params[2]))")
            println("  n  = $(estimated_params[3]) (true: $(true_params[3]))")
            
            # Calculate confidence intervals (95%)
            confidence_level = 1.96  # for 95% CI
            ci_lower = estimated_params - confidence_level * sqrt.(diag(param_cov))
            ci_upper = estimated_params + confidence_level * sqrt.(diag(param_cov))
            
            println("\n95% Confidence Intervals:")
            for (i, param_name) in enumerate(["k0", "Ea", "n"])
                println("  $param_name: [$(ci_lower[i]), $(ci_upper[i])]")
            end
            
            # Store results
            push!(results["covariance_history"], param_cov)
            push!(results["confidence_intervals"], [ci_lower ci_upper])
            
            # Calculate FIM determinant (measure of information)
            FIM_current, _ = calculate_fim(current_params, exp_conditions, measurements)
            push!(results["fim_det_history"], det(FIM_current))
            
            #= 
            STEP 4: Check stopping criterion
            Stop if parameter estimates are sufficiently accurate
            =#
            if stopping_criterion(param_cov, current_params, true_params)
                println("\n✓ Stopping criterion met!")
                break
            end
        end
        
        # Update history
        push!(results["parameters_history"], current_params)
        results["experimental_conditions"] = exp_conditions
        results["measurements"] = measurements
        results["iteration"] = iter
    end
    
    return results
end

#=============================================================================
PART 5: VISUALIZATION AND RESULTS ANALYSIS
=============================================================================#

"""
    plot_parameter_estimation_history(results, true_params)

Create comprehensive plots showing the evolution of parameter estimates,
confidence intervals, and information content.

Arguments:
- results: Dictionary containing results from sequential design
- true_params: True parameter values for comparison
"""
function plot_parameter_estimation_history(results, true_params)
    # Extract data
    params_history = results["parameters_history"]
    fim_history = results["fim_det_history"]
    ci_history = results["confidence_intervals"]
    
    n_iterations = length(params_history) - 1  # Subtract initial guess
    iterations = 1:n_iterations
    
    # Parameter names and true values
    param_names = ["k₀ (pre-exponential)", "Eₐ (activation energy)", "n (reaction order)"]
    
    #= 
    Plot 1: Parameter estimation evolution with confidence intervals
    =#
    p1 = plot(layout=(3,1), size=(800, 600), 
              title="Parameter Estimation History\n(Sequential Optimal Experimental Design)")
    
    for i in 1:3
        # Extract parameter values at each iteration
        param_values = [params_history[iter][i] for iter in 2:length(params_history)]
        
        # Plot parameter estimates
        plot!(p1[i], iterations, param_values, 
              marker=:circle, linewidth=2, label="Estimated", color=:blue)
        
        # Add true parameter value
        hline!(p1[i], [true_params[i]], linewidth=2, linestyle=:dash, 
               label="True value", color=:red)
        
        # Add confidence intervals if available
        if !isempty(ci_history)
            ci_lower = [ci[1][i] for ci in ci_history]
            ci_upper = [ci[2][i] for ci in ci_history]
            
            # Fill confidence region
            plot!(p1[i], iterations, ci_lower, fillrange=ci_upper, 
                  fillalpha=0.2, linewidth=0, label="95% CI", color=:gray)
        end
        
        # Formatting
        xlabel!(p1[i], "Iteration")
        ylabel!(p1[i], param_names[i])
    end
    
    #= 
    Plot 2: Fisher Information Matrix determinant evolution
    Shows how information content increases with experiments
    =#
    p2 = plot(size=(800, 300))
    plot!(p2, iterations, fim_history, 
          marker=:square, linewidth=3, color=:green,
          label="FIM Determinant", 
          xlabel="Iteration", 
          ylabel="det(FIM)",
          title="Information Content Evolution\n(Higher = More Precise Estimates)")
    
    # Add trend line
    plot!(p2, iterations, fim_history, linewidth=2, linestyle=:dash, 
          color=:darkgreen, label="Trend")
    
    #= 
    Plot 3: Experimental design space with selected points
    =#
    p3 = plot(size=(800, 400))
    
    # Get experimental conditions
    exp_conditions = results["experimental_conditions"]
    
    if !isempty(exp_conditions)
        # Plot all experiments with iteration numbers
        scatter!(p3, exp_conditions[:, 2], exp_conditions[:, 1], 
                markersize=10, marker=:circle, 
                color=1:size(exp_conditions, 1),
                label="Experimental points",
                xlabel="Inlet Concentration (mol/m³)",
                ylabel="Temperature (K)",
                title="Optimal Experimental Design Space")
        
        # Add iteration numbers
        for i in 1:size(exp_conditions, 1)
            annotate!(p3, exp_conditions[i, 2], exp_conditions[i, 1], 
                     text(string(i), :white, :center, 8))
        end
    end
    
    #= 
    Plot 4: Parameter correlation matrix (heatmap)
    =#
    if !isempty(results["covariance_history"])
        latest_cov = results["covariance_history"][end]
        
        # Convert to correlation matrix
        corr_matrix = latest_cov ./ sqrt.(diag(latest_cov) * diag(latest_cov)')
        
        p4 = heatmap(1:3, 1:3, corr_matrix,
                    xlabel="Parameter", ylabel="Parameter",
                    title="Parameter Correlation Matrix\n(Final Iteration)",
                    color=:RdBu, clims=(-1, 1))
        
        # Add correlation values as text
        for i in 1:3, j in 1:3
            annotate!(p4, j, i, text(round(corr_matrix[i,j], digits=2), :white, 10))
        end
    end
    
    # Combine plots in a grid
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    
    return final_plot
end

"""
    calculate_parameter_accuracy(results, true_params)

Calculate various metrics to assess parameter estimation accuracy.

Arguments:
- results: Results dictionary from sequential design
- true_params: True parameter values

Returns:
- accuracy_metrics: Dictionary with accuracy metrics
"""
function calculate_parameter_accuracy(results, true_params)
    final_params = results["parameters_history"][end]
    final_cov = results["covariance_history"][end]
    
    # Relative errors
    rel_errors = abs.(final_params - true_params) ./ abs.(true_params) * 100
    
    # Coefficient of variation (CV = std/mean)
    cv = sqrt.(diag(final_cov)) ./ abs.(final_params) * 100
    
    # Parameter correlations
    corr_matrix = final_cov ./ sqrt.(diag(final_cov) * diag(final_cov)')
    
    # Information gain
    info_gain = results["fim_det_history"][end] / results["fim_det_history"][1]
    
    metrics = Dict(
        "relative_errors_percent" => rel_errors,
        "coefficient_variation_percent" => cv,
        "correlation_matrix" => corr_matrix,
        "information_gain" => info_gain,
        "final_fim_det" => results["fim_det_history"][end],
        "n_experiments" => length(results["fim_det_history"])
    )
    
    return metrics
end

#=============================================================================
PART 6: MAIN EXECUTION
=============================================================================#

"""
    run_optimal_experimental_design()

Main function to execute the sequential optimal experimental design framework.
"""
function run_optimal_experimental_design()
    
    #= 
    Define stopping criterion
    Stop when coefficient of variation < 5% for all parameters
    =#
    function stopping_criterion(param_cov, current_params, true_params)
        cv = sqrt.(diag(param_cov)) ./ abs.(current_params)
        return all(cv .< 0.05)  # Stop when all CV < 5%
    end
    
    #= 
    Initial parameter guess (could be from literature or prior knowledge)
    We start with values that are intentionally off from true values
    =#
    initial_params = [0.5, 40000.0, 1.0]  # [k0, Ea, n]
    
    #= 
    Experimental domain boundaries
    [T_min, T_max; C_min, C_max]
    =#
    exp_domain = [
        300.0 500.0;    # Temperature range (K)
        0.1   10.0       # Concentration range (mol/m³)
    ]
    
    # Maximum number of experiments
    n_experiments_max = 10
    
    # Design criterion (D-optimal: maximize determinant of FIM)
    design_criterion = "D"
    
    println("\n" * repeat("#", 60))
    println("OPTIMAL EXPERIMENTAL DESIGN USING FIM/GIM")
    println(repeat("#", 60))
    println("\nThis framework demonstrates how to optimally design")
    println("experiments to estimate kinetic parameters with minimum")
    println("experimental effort using Fisher Information Matrix (FIM)")
    println("and General Information Matrix (GIM) approaches.")
    
    # Run sequential experimental design
    results = sequential_experimental_design(
        initial_params, exp_domain, n_experiments_max, 
        stopping_criterion, design_criterion)
    
    # True parameters (for comparison)
    true_params = [1.0, 50000.0, 1.5]
    
    # Calculate accuracy metrics
    metrics = calculate_parameter_accuracy(results, true_params)
    
    println("\n" * "="*60)
    println("FINAL RESULTS")
    println(repeat("#", 60))
    println("\nParameter Estimation Accuracy:")
    for (i, name) in enumerate(["k0", "Ea", "n"])
        println("  $name:")
        println("    True: $(true_params[i])")
        println("    Estimated: $(results["parameters_history"][end][i])")
        println("    Relative Error: $(round(metrics["relative_errors_percent"][i], digits=2))%")
        println("    Coeff. of Variation: $(round(metrics["coefficient_variation_percent"][i], digits=2))%")
    end
    
    println("\nInformation Gain: $(round(metrics["information_gain"], digits=2))x")
    println("Total Experiments: $(metrics["n_experiments"])")
    
    # Create visualization
    println("\nGenerating visualization...")
    final_plot = plot_parameter_estimation_history(results, true_params)
    
    # Save plot
    savefig(final_plot, "optimal_experimental_design_results.png")
    println("Plot saved as 'optimal_experimental_design_results.png'")
    
    # Display plot
    display(final_plot)
    
    return results, metrics, final_plot
end

#=============================================================================
EXECUTE THE MAIN FUNCTION
=============================================================================#

# Run the complete analysis
#println(repeat("#", 60))
println(repeat("#", 60))
println("# STARTING OPTIMAL EXPERIMENTAL DESIGN FRAMEWORK")
println(repeat("#", 60)) 

# Execute the main function
results, metrics, plot = run_optimal_experimental_design()

println(repeat("#", 60))
println("# ANALYSIS COMPLETE")
println("#"*60)

# Return results to workspace (for further analysis if needed)
results