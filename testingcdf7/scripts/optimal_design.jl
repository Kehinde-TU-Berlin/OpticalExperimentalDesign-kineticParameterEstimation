# optimal_design.jl - New file to be placed in the same directory
module OptimalExperimentalDesign

using LinearAlgebra
using Statistics
using Distributions
using QuasiMonteCarlo
using Optim
using ForwardDiff
using Printf
using Plots
using DataFrames
using CSV
using Dates

# Import from your existing modules
using ..Example_Inverse_Problem_Validation_2D
using inverse_problem_RBS_CFD_het_cat

# Structure to hold Fisher Information Matrix
struct FisherInformationMatrix
    FIM::Matrix{Float64}
    eigenvalues::Vector{Float64}
    condition_number::Float64
    determinant::Float64
    trace::Float64
end

# Structure for Global Information Matrix
struct GlobalInformationMatrix
    GIM::Matrix{Float64}
    weights::Vector{Float64}
    individual_FIMs::Vector{FisherInformationMatrix}
    D_optimal_criterion::Float64
    A_optimal_criterion::Float64
    E_optimal_criterion::Float64
end

# Structure for experiment design
struct ExperimentDesign
    temperature::Float64
    inlet_compositions::Vector{Float64}
    expected_information_gain::Float64
    predicted_parameter_variance::Vector{Float64}
end

# Structure for design history
struct DesignHistory
    iteration::Int
    experiments::Vector{ExperimentDesign}
    parameter_estimates::Vector{Vector{Float64}}
    parameter_uncertainties::Vector{Vector{Float64}}
    fim_condition_number::Float64
    gim_condition_number::Float64
    prediction_error::Float64
end

"""
    compute_sensitivity_matrix(parameters, param_indices, experiment_conditions)

Compute sensitivity matrix S = ∂y/∂θ using finite differences
"""
function compute_sensitivity_matrix(parameters::Vector{Float64}, param_indices::Vector{Int}, 
                                   experiment_conditions::Dict, St::Matrix{Float64}, 
                                   nspec::Int, nref::Int=2500)
    
    n_params = length(parameters)
    n_outputs = nspec  # Number of output measurements
    
    # Base simulation
    base_sol = main(
        nref=nref,
        T=experiment_conditions["temperature"],
        inlet_MFs=experiment_conditions["inlet_compositions"],
        St=St,
        k0=parameters,
        ratio=0.1,
        RBS=false
    )
    
    base_output = youts(base_sol, Nspec=nspec)
    
    # Compute sensitivities using central differences
    S = zeros(n_outputs, n_params)
    δ = 1e-6  # Finite difference step
    
    for i in 1:n_params
        params_plus = copy(parameters)
        params_minus = copy(parameters)
        params_plus[i] += δ
        params_minus[i] -= δ
        
        sol_plus = main(
            nref=nref,
            T=experiment_conditions["temperature"],
            inlet_MFs=experiment_conditions["inlet_compositions"],
            St=St,
            k0=params_plus,
            ratio=0.1,
            RBS=false
        )
        
        sol_minus = main(
            nref=nref,
            T=experiment_conditions["temperature"],
            inlet_MFs=experiment_conditions["inlet_compositions"],
            St=St,
            k0=params_minus,
            ratio=0.1,
            RBS=false
        )
        
        output_plus = youts(sol_plus, Nspec=nspec)
        output_minus = youts(sol_minus, Nspec=nspec)
        
        S[:, i] = (output_plus - output_minus) / (2δ)
    end
    
    return S, base_output
end

"""
    compute_fisher_information_matrix(S, noise_variance)

Compute Fisher Information Matrix: FIM = S^T * Σ^{-1} * S
"""
function compute_fisher_information_matrix(S::Matrix{Float64}, noise_variance::Float64)
    n_outputs = size(S, 1)
    Σ_inv = (1.0 / noise_variance) * I(n_outputs)
    FIM = S' * Σ_inv * S
    
    # Add small regularization to ensure positive definiteness
    FIM += 1e-8 * I(size(FIM, 1))
    
    eigenvals = eigvals(FIM)
    cond_num = maximum(eigenvals) / minimum(eigenvals)
    det_val = det(FIM)
    trace_val = tr(FIM)
    
    return FisherInformationMatrix(FIM, real.(eigenvals), cond_num, det_val, trace_val)
end

"""
    compute_global_information_matrix(experiment_FIMs, weights)

Compute Global Information Matrix from multiple experiments
"""
function compute_global_information_matrix(experiment_FIMs::Vector{FisherInformationMatrix}, 
                                          weights::Vector{Float64}=Float64[])
    
    n_experiments = length(experiment_FIMs)
    n_params = size(experiment_FIMs[1].FIM, 1)
    
    # If no weights provided, use equal weights
    if isempty(weights)
        weights = fill(1.0/n_experiments, n_experiments)
    else
        weights = weights / sum(weights)
    end
    
    # Compute weighted sum of FIMs
    GIM = zeros(n_params, n_params)
    for (idx, fim) in enumerate(experiment_FIMs)
        GIM += weights[idx] * fim.FIM
    end
    
    # Compute criteria
    eigenvals = eigvals(GIM)
    D_opt = det(GIM)^(1/n_params)
    A_opt = tr(inv(GIM + 1e-8 * I))
    E_opt = minimum(eigenvals)
    
    return GlobalInformationMatrix(GIM, weights, experiment_FIMs, D_opt, A_opt, E_opt)
end

"""
    design_next_experiment(current_parameters, parameter_uncertainties, 
                          design_space, St, nspec)

Use D-optimal design to select next experimental conditions
"""
function design_next_experiment(current_parameters::Vector{Float64},
                               current_covariance::Matrix{Float64},
                               design_space::Dict,
                               St::Matrix{Float64},
                               nspec::Int,
                               n_candidates::Int=50)
    
    best_criterion = -Inf
    best_experiment = nothing
    best_information_gain = 0.0
    
    # Generate candidate points in design space
    candidates = []
    for _ in 1:n_candidates
        T = design_space["temperature_range"][1] + rand() * (design_space["temperature_range"][2] - design_space["temperature_range"][1])
        
        # Generate inlet compositions (must sum to 1)
        inlet_comps = zeros(nspec)
        for i in 1:nspec-1
            inlet_comps[i] = design_space["inlet_range"][i][1] + rand() * (design_space["inlet_range"][i][2] - design_space["inlet_range"][i][1])
        end
        inlet_comps[nspec] = 1.0 - sum(inlet_comps[1:nspec-1])
        
        # Ensure valid composition
        if all(inlet_comps .>= 0) && abs(sum(inlet_comps) - 1.0) < 1e-6
            push!(candidates, (T=T, inlet=inlet_comps))
        end
    end
    
    # Evaluate each candidate
    for candidate in candidates
        exp_conditions = Dict(
            "temperature" => candidate.T,
            "inlet_compositions" => candidate.inlet
        )
        
        # Compute sensitivity and FIM for this candidate
        try
            S, _ = compute_sensitivity_matrix(current_parameters, 1:length(current_parameters), 
                                             exp_conditions, St, nspec)
            noise_variance = 1e-6  # Measurement noise variance
            FIM_candidate = compute_fisher_information_matrix(S, noise_variance)
            
            # D-optimal criterion: maximize determinant of (current_FIM + new_FIM)
            combined_FIM = current_covariance + FIM_candidate.FIM
            criterion = det(combined_FIM)^(1/length(current_parameters))
            
            # Information gain
            info_gain = 0.5 * log(det(combined_FIM) / det(current_covariance + 1e-8*I))
            
            if criterion > best_criterion
                best_criterion = criterion
                best_experiment = ExperimentDesign(
                    candidate.T,
                    candidate.inlet,
                    info_gain,
                    sqrt.(diag(inv(combined_FIM + 1e-8*I)))
                )
                best_information_gain = info_gain
            end
        catch e
            @warn "Failed to evaluate candidate: $e"
            continue
        end
    end
    
    return best_experiment
end

"""
    run_optimal_design_workflow(initial_parameters, parameter_bounds, 
                               St, nspec, max_iterations)

Main workflow for optimal experimental design
"""
function run_optimal_design_workflow(initial_parameters::Vector{Float64},
                                    parameter_bounds::Vector{Tuple{Float64,Float64}},
                                    St::Matrix{Float64},
                                    nspec::Int;
                                    max_iterations::Int=10,
                                    initial_experiments::Int=3,
                                    noise_level::Float64=1e-6)
    
    # Initialize history
    history = DesignHistory[]
    current_parameters = copy(initial_parameters)
    current_covariance = 1e-2 * I(length(initial_parameters))  # Initial uncertainty
    all_experiments = ExperimentDesign[]
    
    # Define design space
    design_space = Dict(
        "temperature_range" => (300.0, 600.0),
        "inlet_range" => [[(0.1, 0.5)] for _ in 1:nspec-1]
    )
    
    # Store results for plotting
    parameter_history = [copy(current_parameters)]
    uncertainty_history = [sqrt.(diag(current_covariance))]
    fim_condition_history = []
    gim_condition_history = []
    prediction_error_history = []
    
    # Initial random experiments to get started
    @info "Running $initial_experiments initial experiments..."
    initial_FIMs = []
    
    for i in 1:initial_experiments
        # Random experiment conditions
        T = design_space["temperature_range"][1] + rand() * (design_space["temperature_range"][2] - design_space["temperature_range"][1])
        inlet = zeros(nspec)
        for j in 1:nspec-1
            inlet[j] = design_space["inlet_range"][j][1] + rand() * (design_space["inlet_range"][j][2] - design_space["inlet_range"][j][1])
        end
        inlet[nspec] = 1.0 - sum(inlet[1:nspec-1])
        
        exp_conditions = Dict("temperature" => T, "inlet_compositions" => inlet)
        S, output = compute_sensitivity_matrix(current_parameters, 1:length(current_parameters), 
                                              exp_conditions, St, nspec)
        FIM = compute_fisher_information_matrix(S, noise_level)
        push!(initial_FIMs, FIM)
        push!(all_experiments, ExperimentDesign(T, inlet, 0.0, sqrt.(diag(inv(FIM.FIM)))))
    end
    
    # Compute initial GIM
    GIM = compute_global_information_matrix(initial_FIMs)
    current_covariance = inv(GIM.GIM + 1e-8 * I)
    
    push!(fim_condition_history, mean([f.condition_number for f in initial_FIMs]))
    push!(gim_condition_history, GIM.condition_number)
    
    # Main iterative design loop
    for iteration in 1:max_iterations
        @info "Optimal Design Iteration $iteration/$max_iterations"
        
        # Design next experiment based on current knowledge
        next_exp = design_next_experiment(current_parameters, current_covariance,
                                         design_space, St, nspec)
        
        if next_exp === nothing
            @warn "No valid experiment found, stopping"
            break
        end
        
        @info "Selected experiment: T=$(next_exp.temperature)K, Inlet=$(round.(next_exp.inlet_compositions, digits=3))"
        @info "Expected information gain: $(next_exp.expected_information_gain)"
        
        # Perform the experiment (simulate with true parameters)
        exp_conditions = Dict(
            "temperature" => next_exp.temperature,
            "inlet_compositions" => next_exp.inlet_compositions
        )
        
        # Simulate experiment with some noise
        true_sol = main(
            nref=2500,
            T=next_exp.temperature,
            inlet_MFs=next_exp.inlet_compositions,
            St=St,
            k0=current_parameters,  # Use current best estimate
            ratio=0.1,
            RBS=false
        )
        
        true_output = youts(true_sol, Nspec=nspec)
        
        # Add measurement noise
        noisy_output = true_output + rand(Normal(0, noise_level), nspec)
        
        # Compute FIM for this experiment
        S, _ = compute_sensitivity_matrix(current_parameters, 1:length(current_parameters),
                                         exp_conditions, St, nspec)
        new_FIM = compute_fisher_information_matrix(S, noise_level)
        
        # Update global information matrix
        push!(initial_FIMs, new_FIM)
        GIM = compute_global_information_matrix(initial_FIMs)
        
        # Update parameter covariance
        current_covariance = inv(GIM.GIM + 1e-8 * I)
        
        # Estimate parameters using all available data
        @info "Re-estimating parameters..."
        current_parameters = estimate_parameters(all_experiments, initial_FIMs, 
                                                 current_parameters, St, nspec, noise_level)
        
        # Store history
        push!(all_experiments, next_exp)
        push!(parameter_history, copy(current_parameters))
        push!(uncertainty_history, sqrt.(diag(current_covariance)))
        push!(fim_condition_history, mean([f.condition_number for f in initial_FIMs]))
        push!(gim_condition_history, GIM.condition_number)
        
        # Compute prediction error
        pred_error = compute_prediction_error(current_parameters, St, nspec)
        push!(prediction_error_history, pred_error)
        
        # Create history record
        push!(history, DesignHistory(
            iteration,
            copy(all_experiments),
            copy(parameter_history),
            copy(uncertainty_history),
            gim_condition_history[end],
            fim_condition_history[end],
            pred_error
        ))
        
        # Check convergence
        if iteration > 2 && maximum(abs.(parameter_history[end] - parameter_history[end-1])) < 1e-4
            @info "Parameters converged!"
            break
        end
    end
    
    return history, all_experiments, parameter_history, uncertainty_history
end

"""
    estimate_parameters(experiments, FIMs, initial_guess, St, nspec, noise_level)

Estimate parameters using weighted least squares with FIM-based weights
"""
function estimate_parameters(experiments::Vector{ExperimentDesign},
                           FIMs::Vector{FisherInformationMatrix},
                           initial_guess::Vector{Float64},
                           St::Matrix{Float64},
                           nspec::Int,
                           noise_level::Float64)
    
    # Use weights based on FIM determinants (more informative experiments get higher weight)
    weights = [log(fim.determinant + 1) for fim in FIMs]
    weights ./= sum(weights)
    
    # Objective function for parameter estimation
    function objective(params)
        residual = 0.0
        for (idx, exp) in enumerate(experiments)
            # Simulate at current parameters
            sol = main(
                nref=2500,
                T=exp.temperature,
                inlet_MFs=exp.inlet_compositions,
                St=St,
                k0=params,
                ratio=0.1,
                RBS=false
            )
            output = youts(sol, Nspec=nspec)
            
            # Weighted residual
            residual += weights[idx] * sum(output .^ 2)
        end
        return residual
    end
    
    # Optimize parameters
    result = optimize(objective, initial_guess, LBFGS(), 
                     Optim.Options(iterations=100, show_trace=false))
    
    return result.minimizer
end

"""
    compute_prediction_error(parameters, St, nspec)

Compute prediction error for validation
"""
function compute_prediction_error(parameters::Vector{Float64}, St::Matrix{Float64}, nspec::Int)
    # Test on a validation condition
    T_val = 450.0
    inlet_val = [0.3, 0.3, 0.4]
    
    sol = main(
        nref=2500,
        T=T_val,
        inlet_MFs=inlet_val,
        St=St,
        k0=parameters,
        ratio=0.1,
        RBS=false
    )
    
    output = youts(sol, Nspec=nspec)
    return norm(output)
end

"""
    plot_design_results(history, parameter_history, uncertainty_history, 
                       fim_history, gim_history, error_history)

Create comprehensive plots showing improvement
"""
function plot_design_results(history::Vector{DesignHistory},
                            parameter_history::Vector{Vector{Float64}},
                            uncertainty_history::Vector{Vector{Float64}},
                            fim_history::Vector{Float64},
                            gim_history::Vector{Float64},
                            error_history::Vector{Float64},
                            true_parameters::Vector{Float64}=Float64[])
    
    n_iterations = length(parameter_history)
    n_params = length(parameter_history[1])
    
    # Create plots
    plots = []
    
    # Plot 1: Parameter convergence
    p1 = plot(title="Parameter Convergence", xlabel="Iteration", ylabel="Parameter Value")
    for i in 1:n_params
        plot!(p1, 1:n_iterations, [p[i] for p in parameter_history], 
              label="θ$i", marker=:circle, linewidth=2)
        if !isempty(true_parameters)
            hline!(p1, [true_parameters[i]], linestyle=:dash, label="True θ$i", alpha=0.5)
        end
    end
    push!(plots, p1)
    
    # Plot 2: Parameter uncertainty reduction
    p2 = plot(title="Parameter Uncertainty Reduction", xlabel="Iteration", ylabel="Standard Deviation")
    for i in 1:n_params
        plot!(p2, 1:n_iterations, [u[i] for u in uncertainty_history], 
              label="σ_θ$i", marker=:square, linewidth=2)
    end
    push!(plots, p2)
    
    # Plot 3: FIM and GIM condition numbers
    p3 = plot(title="Information Matrix Condition Numbers", xlabel="Iteration", ylabel="Condition Number")
    plot!(p3, 1:length(fim_history), fim_history, label="Mean FIM Condition", marker=:diamond, linewidth=2)
    plot!(p3, 1:length(gim_history), gim_history, label="GIM Condition", marker=:star, linewidth=2)
    plot!(p3, yscale=:log10)
    push!(plots, p3)
    
    # Plot 4: Prediction error reduction
    p4 = plot(title="Prediction Error Reduction", xlabel="Iteration", ylabel="Prediction Error (norm)")
    plot!(p4, 1:length(error_history), error_history, label="Validation Error", 
          marker=:circle, linewidth=2, color=:red)
    push!(plots, p4)
    
    # Plot 5: D-optimal criterion improvement
    p5 = plot(title="D-Optimal Criterion", xlabel="Iteration", ylabel="Det(GIM)^(1/p)")
    d_opt_history = [h.gim_condition_number^(-1/n_params) for h in history]
    plot!(p5, 1:length(d_opt_history), d_opt_history, label="D-optimality", 
          marker=:circle, linewidth=2, color=:green)
    push!(plots, p5)
    
    # Plot 6: Experimental conditions over iterations
    p6 = plot(title="Selected Experimental Conditions", xlabel="Iteration", ylabel="Temperature (K)")
    temp_history = [exp.temperature for exp in history[1].experiments]
    plot!(p6, 1:length(temp_history), temp_history, label="Temperature", marker=:circle, linewidth=2)
    
    # Add inlet composition as second y-axis
    p6_2 = twinx(p6)
    inlet1_history = [exp.inlet_compositions[1] for exp in history[1].experiments]
    plot!(p6_2, 1:length(inlet1_history), inlet1_history, label="Inlet Species 1", 
          marker=:square, linewidth=2, color=:orange, linestyle=:dash)
    ylabel!(p6_2, "Inlet Composition")
    push!(plots, p6)
    
    # Plot 7: Information gain per experiment
    p7 = plot(title="Information Gain per Experiment", xlabel="Experiment Number", ylabel="Expected Information Gain")
    info_gain_history = [exp.expected_information_gain for exp in history[1].experiments[3:end]]
    bar!(p7, 1:length(info_gain_history), info_gain_history, label="Information Gain", color=:purple)
    push!(plots, p7)
    
    # Plot 8: Parameter correlation matrix evolution
    p8 = plot(title="Parameter Correlation Matrix (Final)", 
              xlabel="Parameter Index", ylabel="Parameter Index")
    # This would show the correlation matrix from the final GIM
    push!(plots, p8)
    
    # Combine plots into a single figure
    final_plot = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(1200, 1600))
    
    return final_plot
end

"""
    save_design_results(history, filename_prefix)

Save results to CSV files
"""
function save_design_results(history::Vector{DesignHistory}, filename_prefix::String)
    # Save parameter history
    param_df = DataFrame(
        Iteration=1:length(history),
        [Pair("θ$i", [h.parameter_estimates[end][i] for h in history]) for i in 1:length(history[1].parameter_estimates[1])]...
    )
    CSV.write("$(filename_prefix)_parameters.csv", param_df)
    
    # Save uncertainty history
    unc_df = DataFrame(
        Iteration=1:length(history),
        [Pair("σ_θ$i", [h.parameter_uncertainties[end][i] for h in history]) for i in 1:length(history[1].parameter_uncertainties[1])]...
    )
    CSV.write("$(filename_prefix)_uncertainties.csv", unc_df)
    
    # Save FIM/GIM history
    info_df = DataFrame(
        Iteration=1:length(history),
        FIM_Condition=[h.fim_condition_number for h in history],
        GIM_Condition=[h.gim_condition_number for h in history],
        Prediction_Error=[h.prediction_error for h in history]
    )
    CSV.write("$(filename_prefix)_information.csv", info_df)
    
    # Save experiment conditions
    exp_df = DataFrame(
        Experiment=1:length(history[1].experiments),
        Temperature=[exp.temperature for exp in history[1].experiments],
        Inlet_Species1=[exp.inlet_compositions[1] for exp in history[1].experiments],
        Inlet_Species2=[exp.inlet_compositions[2] for exp in history[1].experiments],
        Inlet_Species3=[exp.inlet_compositions[3] for exp in history[1].experiments],
        Information_Gain=[exp.expected_information_gain for exp in history[1].experiments]
    )
    CSV.write("$(filename_prefix)_experiments.csv", exp_df)
    
    @info "Results saved to $(filename_prefix)_*.csv"
end

"""
    main_optimal_design()

Main function to run the complete optimal design workflow
"""
function main_optimal_design()
    # Setup problem
    St = [-2 -1 2]
    nspec = size(St, 2)
    true_parameters = [4000.0, 4000.0]
    initial_guess = [1000.0, 1000.0]
    
    parameter_bounds = [(100.0, 10000.0), (100.0, 10000.0)]
    
    @info "Starting Optimal Experimental Design for Catalytic Reactor"
    @info "True parameters: $true_parameters"
    @info "Initial guess: $initial_guess"
    
    # Run optimal design
    history, experiments, param_history, unc_history = run_optimal_design_workflow(
        initial_guess,
        parameter_bounds,
        St,
        nspec,
        max_iterations=8,
        initial_experiments=3,
        noise_level=1e-6
    )
    
    # Create plots
    @info "Generating plots..."
    fim_history = [h.fim_condition_number for h in history]
    gim_history = [h.gim_condition_number for h in history]
    error_history = [h.prediction_error for h in history]
    
    plot = plot_design_results(history, param_history, unc_history, 
                               fim_history, gim_history, error_history,
                               true_parameters)
    
    # Save plot
    savefig(plot, "optimal_design_results.png")
    @info "Plot saved to optimal_design_results.png"
    
    # Save results
    save_design_results(history, "optimal_design")
    
    # Print final results
    @info "\n=== FINAL RESULTS ==="
    @info "Final parameter estimates: $(round.(param_history[end], digits=2))"
    @info "Final uncertainties: $(round.(unc_history[end], digits=2))"
    @info "True parameters: $true_parameters"
    @info "Final GIM condition number: $(gim_history[end])"
    @info "Final prediction error: $(error_history[end])"
    
    # Calculate improvement
    initial_error = error_history[1]
    final_error = error_history[end]
    improvement = (initial_error - final_error) / initial_error * 100
    @info "Prediction error improvement: $(round(improvement, digits=2))%"
    
    return history, experiments, param_history, unc_history, plot
end

end # module
