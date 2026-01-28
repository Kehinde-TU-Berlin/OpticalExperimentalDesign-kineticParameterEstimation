using QuasiMonteCarlo
using Optim
using LinearAlgebra

function run_parameter_estimation(; Nexps=5, Sampling=HaltonSample(), add_noise=true, sigma=1e-3, order=1)
    # Generate optimal experimental conditions using Halton sampling
    Temp = generate_optimal_temperatures(Nexps; lb=200.0, ub=400.0, Sampling=Sampling)
    ca_ins = generate_optimal_concentrations(Nexps; lb=0.1, ub=1.0, Sampling=Sampling)
    
    # Generate experimental data (assuming ca_exp is your data generator)
    ca_exper = ca_exp(ca_ins, Temp; Nexps=Nexps, k=[1.0, 20000], add_noise=add_noise, sigma=sigma, order=order)
    
    # Run parameter estimation
    result = parameter_estimator(; Nexps=Nexps, ca_exp=ca_exper, ca_in=ca_ins, Temp=Temp, 
                                order=order, initial_guess=[0.0, 0.0])
    
    return result, (Temp, ca_ins, ca_exper)
end

function generate_optimal_temperatures(Nexps; lb=200.0, ub=400.0, Sampling=HaltonSample())
    """
    Generate optimally spaced temperature points using quasi-random sampling
    """
    lb_vec = [lb]
    ub_vec = [ub]
    
    samples = QuasiMonteCarlo.sample(Nexps, lb_vec, ub_vec, Sampling)
    return vec(samples)  # Convert to 1D vector
end

function generate_optimal_concentrations(Nexps; lb=0.1, ub=1.0, Sampling=HaltonSample())
    """
    Generate optimally spaced concentration points using quasi-random sampling
    """
    lb_vec = [lb]
    ub_vec = [ub]
    
    samples = QuasiMonteCarlo.sample(Nexps, lb_vec, ub_vec, Sampling)
    return vec(samples)  # Convert to 1D vector
end

# Alternative: Generate both parameters together for better space-filling
function generate_optimal_conditions(Nexps; T_lb=200.0, T_ub=400.0, ca_lb=0.1, ca_ub=1.0, Sampling=HaltonSample())
    """
    Generate optimal (Temperature, Concentration) pairs together for better experimental design
    """
    lb = [T_lb, ca_lb]
    ub = [T_ub, ca_ub]
    
    samples = QuasiMonteCarlo.sample(Nexps, lb, ub, Sampling)
    Temps = samples[1, :]
    ca_ins = samples[2, :]
    
    return Temps, ca_ins
end

# Enhanced version with better experimental design
function run_parameter_estimation_enhanced(; Nexps=5, Sampling=HaltonSample(), add_noise=true, 
                                         sigma=1e-3, order=1, joint_sampling=true)
    """
    Enhanced parameter estimation with better experimental design options
    """
    
    if joint_sampling
        # Sample Temperature and Concentration together for optimal space-filling
        Temp, ca_ins = generate_optimal_conditions(Nexps; T_lb=200.0, T_ub=400.0, 
                                                 ca_lb=0.1, ca_ub=1.0, Sampling=Sampling)
    else
        # Sample independently (your original approach)
        Temp = generate_optimal_temperatures(Nexps; lb=200.0, ub=400.0, Sampling=Sampling)
        ca_ins = generate_optimal_concentrations(Nexps; lb=0.1, ub=1.0, Sampling=Sampling)
    end
    
    # Generate experimental data
    true_params = [1.0, 20000]  # Pre-exponential factor, Activation energy
    ca_exper = ca_exp(ca_ins, Temp; Nexps=Nexps, k=true_params, add_noise=add_noise, 
                     sigma=sigma, order=order)
    
    # Better initial guess based on physical knowledge
    initial_guess = [0.5, 15000]  # More reasonable starting point
    
    # Run parameter estimation
    result = parameter_estimator(; Nexps=Nexps, ca_exp=ca_exper, ca_in=ca_ins, 
                                Temp=Temp, order=order, initial_guess=initial_guess)
    
    return result, (Temp, ca_ins, ca_exper, true_params)
end

# Analysis function to evaluate estimation quality
function analyze_estimation_results(result, true_params, experimental_data)
    """
    Analyze the quality of parameter estimation
    """
    estimated_params = result.minimizer
    Temp, ca_ins, ca_exper = experimental_data
    
    println("Parameter Estimation Results:")
    println("True parameters: ", true_params)
    println("Estimated parameters: ", estimated_params)
    println("Relative errors: ", abs.(estimated_params - true_params) ./ abs.(true_params) .* 100, "%")
    
    # Calculate RMSE
    predictions = [CSTR_model(ca_ins[i], Temp[i]; k=estimated_params) for i in 1:length(ca_ins)]
    rmse = sqrt(mean((predictions .- ca_exper).^2))
    println("RMSE: ", rmse)
    
    return rmse, predictions
end

# Monte Carlo version for uncertainty analysis
function monte_carlo_parameter_estimation(; N_runs=100, Nexps=5, sigma=1e-3, order=1)
    """
    Run multiple parameter estimations to assess uncertainty
    """
    all_estimates = Vector{Vector{Float64}}(undef, N_runs)
    all_rmse = Vector{Float64}(undef, N_runs)
    
    for i in 1:N_runs
        result, data = run_parameter_estimation_enhanced(; Nexps=Nexps, sigma=sigma, order=order)
        all_estimates[i] = result.minimizer
        rmse, _ = analyze_estimation_results(result, data[4], data[1:3])
        all_rmse[i] = rmse
    end
    
    # Calculate statistics
    estimates_matrix = reduce(hcat, all_estimates)'
    mean_estimates = mean(estimates_matrix, dims=1)
    std_estimates = std(estimates_matrix, dims=1)
    
    println("\nMonte Carlo Analysis (", N_runs, " runs):")
    println("Mean estimates: ", vec(mean_estimates))
    println("Std. dev.: ", vec(std_estimates))
    println("Coefficient of variation: ", vec(std_estimates) ./ vec(mean_estimates) .* 100, "%")
    
    return all_estimates, all_rmse
end

# Usage examples
function demo_parameter_estimation()
    println("=== Basic Parameter Estimation ===")
    result1, data1 = run_parameter_estimation(Nexps=10, sigma=1e-3)
    
    println("\n=== Enhanced Parameter Estimation ===")
    result2, data2 = run_parameter_estimation_enhanced(Nexps=10, joint_sampling=true)
    rmse, predictions = analyze_estimation_results(result2, data2[4], data2[1:3])
    
    println("\n=== Monte Carlo Uncertainty Analysis ===")
    estimates, rmses = monte_carlo_parameter_estimation(N_runs=50, Nexps=10)
    
    return result2, data2
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_parameter_estimation()
end