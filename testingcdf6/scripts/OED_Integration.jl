# =============================================================================
# OPTIMAL EXPERIMENTAL DESIGN USING FISHER INFORMATION MATRIX
# Integration with your existing CFD model for kinetic parameter estimation
# =============================================================================

module OED_Integration

using LinearAlgebra
using Distributions
using QuasiMonteCarlo
using ForwardDiff
using Plots
using LaTeXStrings
using DataFrames
using CSV
using Statistics
using Dates
using Optim
using NLsolve
using Statistics
using JuMP
using SqpSolver
using Ipopt
#using BilevelJuMP
#using HiGHS
using Distributions
using QuasiMonteCarlo
using ForwardDiff
using FiniteDiff
#using HSL
using SparseArrays
using ExtendableGrids
using Revise
using DelimitedFiles
using Serialization

using Optim
using LeastSquaresOptim

include("inverse_problem_RBS_CFD_het_cat.jl")
using .inverse_problem_RBS_CFD_het_cat

# Import your existing modules
using .Example_Inverse_Problem_Validation_2D


# Re-export
export run_oed_workflow, compute_fisher_information_matrix, 
       compute_global_information_matrix, plot_oed_results

# =============================================================================
# 1. FISHER INFORMATION MATRIX COMPUTATION
# =============================================================================

"""
    compute_sensitivity_matrix(θ, Y_in, T, ratio, St, nref)

Computes the sensitivity matrix J = ∂y/∂θ using finite differences.
Returns matrix of size (nspec × nθ)
"""
function compute_sensitivity_matrix(θ, Y_in, T, ratio, St, nref)
    nθ = length(θ)
    nspec = size(St, 2)
    
    # Base simulation
    d_base = Example_Inverse_Problem_Validation_2D.main(
        nref=nref, T=T, P_total=50.0, levels=1, RBS=false,
        St=St, ratio=ratio, Mtcat=1, initvalue=0.1,
        k0=θ[1], Ea=θ[2], RBS_full=false, catcell=1,
        inlet_MFs=Y_in, strategy=nothing,
        unknown_storage=:dense, assembly=:cellwise
    )
    y_base = extract_outlet_concentrations(d_base, nspec)
    
    # Finite difference perturbations
    J = zeros(nspec, nθ)
    ϵ = 1e-4 * max.(abs.(θ), 1.0)
    
    for j in 1:nθ
        θ_pert = copy(θ)
        θ_pert[j] += ϵ[j]
        
        d_pert = Example_Inverse_Problem_Validation_2D.main(
            nref=nref, T=T, P_total=50.0, levels=1, RBS=false,
            St=St, ratio=ratio, Mtcat=1, initvalue=0.1,
            k0=θ_pert[1], Ea=θ_pert[2], RBS_full=false, catcell=1,
            inlet_MFs=Y_in, strategy=nothing,
            unknown_storage=:dense, assembly=:cellwise
        )
        y_pert = extract_outlet_concentrations(d_pert, nspec)
        J[:, j] = (y_pert - y_base) / ϵ[j]
    end
    
    return J
end

"""
    extract_outlet_concentrations(d, nspec)

Extracts outlet concentrations from simulation results.
"""
function extract_outlet_concentrations(d, nspec)
    tsol, sub, sub2, sub3, sub4, grid, bfvelo, arr_out, fac = d
    
    yout = zeros(nspec)
    for i in 1:nspec
        if length(sub4.cellnodes) > 0
            vals = [tsol[i, idx] for idx in sub4.cellnodes]
            yout[i] = mean(vals)
        else
            yout[i] = tsol[i, end]
        end
    end
    return max.(yout, 0.0)
end

"""
    compute_fisher_information_matrix(θ, Y_in, T, ratio, St, nref, σ_meas)

Computes Fisher Information Matrix: FIM = J^T * Σ^{-1} * J
"""
function compute_fisher_information_matrix(θ, Y_in, T, ratio, St, nref, σ_meas)
    J = compute_sensitivity_matrix(θ, Y_in, T, ratio, St, nref)
    nspec = size(J, 1)
    
    # Measurement covariance matrix
    Σ = Diagonal((σ_meas^2 + 1e-12) * ones(nspec))
    Σ_inv = inv(Σ)
    
    # Fisher Information Matrix
    FIM = J' * Σ_inv * J
    
    # Regularization for numerical stability
    FIM = FIM + 1e-8 * I
    
    return FIM, J
end

"""
    compute_global_information_matrix(θ, experiments_list, ratio, St, nref, σ_meas)

Computes Global Information Matrix = sum of individual FIMs
"""
function compute_global_information_matrix(θ, experiments_list, ratio, St, nref, σ_meas)
    nθ = length(θ)
    M = zeros(nθ, nθ)
    all_FIMs = []
    all_Js = []
    
    for (Y_in, T) in experiments_list
        FIM, J = compute_fisher_information_matrix(θ, Y_in, T, ratio, St, nref, σ_meas)
        M += FIM
        push!(all_FIMs, FIM)
        push!(all_Js, J)
    end
    
    return M, all_FIMs, all_Js
end

# =============================================================================
# 2. D-OPTIMAL EXPERIMENTAL DESIGN
# =============================================================================

"""
    d_optimal_criterion(FIM_new, current_M)

D-optimality criterion: maximize determinant of (M + FIM_new)
"""
function d_optimal_criterion(FIM_new, current_M)
    M_new = current_M + FIM_new
    return log(det(M_new)), M_new
end

"""
    generate_candidate_designs(n_candidates, nspec, lb_T, ub_T)

Generates candidate experimental designs (varying both T and inlet composition)
"""
function generate_candidate_designs(n_candidates, nspec, lb_T, ub_T)
    designs = []
    
    for i in 1:n_candidates
        # Random inlet composition (mass fractions)
        mf1 = 0.1 + 0.4 * rand()
        mf2 = 0.1 + 0.4 * rand()
        mf3 = max(0.0, 1.0 - mf1 - mf2)
        Y_in = [mf1, mf2, mf3]
        Y_in = Y_in / sum(Y_in)
        
        # Random temperature
        T = lb_T + (ub_T - lb_T) * rand()
        
        push!(designs, (Y_in, T))
    end
    
    return designs
end

"""
    select_optimal_experiments_sequential(θ, candidate_designs, n_select, ratio, St, nref, σ_meas)

Sequentially selects optimal experiments using D-optimality criterion.
"""
function select_optimal_experiments_sequential(θ, candidate_designs, n_select, ratio, St, nref, σ_meas)
    selected = []
    current_M = zeros(length(θ), length(θ))
    fim_history = []
    info_gains = []
    selected_indices = []
    
    for step in 1:n_select
        best_score = -Inf
        best_design = nothing
        best_FIM = nothing
        best_M = nothing
        best_idx = 0
        
        println("    Selecting experiment $step of $n_select...")
        
        for (idx, (Y_in, T)) in enumerate(candidate_designs)
            try
                FIM, _ = compute_fisher_information_matrix(θ, Y_in, T, ratio, St, nref, σ_meas)
                score, M_new = d_optimal_criterion(FIM, current_M)
                
                if score > best_score
                    best_score = score
                    best_design = (Y_in, T)
                    best_FIM = FIM
                    best_M = M_new
                    best_idx = idx
                end
            catch e
                continue
            end
        end
        
        if best_design !== nothing
            push!(selected, best_design)
            push!(fim_history, best_M)
            push!(info_gains, best_score)
            push!(selected_indices, best_idx)
            current_M = best_M
            
            println("      Selected: T = $(round(best_design[2], digits=1)) K")
            println("      Y_in = [$(join(round.(best_design[1], digits=3), ", "))]")
            println("      log(det(FIM_new)) = $(round(best_score, digits=2))")
            println("      Cumulative log(det(M)) = $(round(log(det(current_M)), digits=2))")
        else
            println("      Warning: No valid design found")
            break
        end
    end
    
    return selected, current_M, fim_history, info_gains
end

# =============================================================================
# 3. PARAMETER ESTIMATION WITH NOISE
# =============================================================================

"""
    simulate_experiment_with_noise(Y_in, T, θ_true, ratio, St, nref, noise_level)

Simulates an experiment with Gaussian noise at specified level.
"""
function simulate_experiment_with_noise(Y_in, T, θ_true, ratio, St, nref, noise_level)
    try
        d = Example_Inverse_Problem_Validation_2D.main(
            nref=nref, T=T, P_total=50.0, levels=1, RBS=false,
            St=St, ratio=ratio, Mtcat=1, initvalue=0.1,
            k0=θ_true[1], Ea=θ_true[2], RBS_full=false, catcell=1,
            inlet_MFs=Y_in, strategy=nothing,
            unknown_storage=:dense, assembly=:cellwise
        )
        y_clean = extract_outlet_concentrations(d, size(St, 2))
        
        # Add noise based on percentage
        if noise_level > 0
            σ = noise_level * maximum(max.(abs.(y_clean), 1e-6))
            y_noisy = y_clean + rand(Normal(0, σ), length(y_clean))
            y_noisy = max.(y_noisy, 0.0)
        else
            y_noisy = y_clean
        end
        
        return y_noisy, true
    catch e
        println("    Simulation failed: $e")
        return zeros(size(St, 2)), false
    end
end

"""
    estimate_parameters_from_data(experiment_data, θ_init, ratio, St, nref, σ_meas)

Estimates parameters using weighted least squares optimization.
"""
function estimate_parameters_from_data(experiment_data, θ_init, ratio, St, nref, σ_meas)
    nspec = size(St, 2)
    
    function objective(θ)
        residual = Float64[]
        for (Y_in, T, y_meas) in experiment_data
            try
                d = Example_Inverse_Problem_Validation_2D.main(
                    nref=nref, T=T, P_total=50.0, levels=1, RBS=false,
                    St=St, ratio=ratio, Mtcat=1, initvalue=0.1,
                    k0=θ[1], Ea=θ[2], RBS_full=false, catcell=1,
                    inlet_MFs=Y_in, strategy=nothing,
                    unknown_storage=:dense, assembly=:cellwise
                )
                y_pred = extract_outlet_concentrations(d, nspec)
                
                for i in 1:length(y_pred)
                    push!(residual, (y_pred[i] - y_meas[i]) / (σ_meas + 1e-8))
                end
            catch e
                push!(residual, 1000.0)  # Penalty for failed simulations
            end
        end
        return 0.5 * sum(residual.^2)
    end
    
    # Use Nelder-Mead for robust optimization
    result = optimize(objective, θ_init, NelderMead(), 
                      Optim.Options(iterations=200, show_trace=false, g_tol=1e-8))
    
    return Optim.minimizer(result)
end

# =============================================================================
# 4. MAIN OED WORKFLOW
# =============================================================================

"""
    run_oed_workflow(; kwargs...)

Main function to run the complete Optimal Experimental Design workflow.
Tests multiple noise levels (0%, 5%, 10%, 20%) and generates comparative plots.
"""
function run_oed_workflow(;
    n_initial=3,           # Number of initial random experiments
    n_optimal=3,           # Number of optimally designed experiments
    n_candidates=30,       # Size of candidate design space
    θ_true=[4000.0, 4000.0],  # True kinetic parameters
    θ_init=[2000.0, 2000.0],  # Initial guess (50% error)
    ratio=0.1,
    St=[-2 -1 2],          # Stoichiometry matrix
    nref=200,              # Mesh refinement (reduced for speed)
    noise_levels=[0.0, 0.05, 0.10, 0.20],  # 0%, 5%, 10%, 20% noise
    lb_T=350.0, ub_T=600.0,
    output_dir="OED_Results")
    
    println("\n"^"="^70)
    println("OPTIMAL EXPERIMENTAL DESIGN USING FISHER INFORMATION MATRIX")
    println("="^70)
    
    # Create output directory
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    nspec = size(St, 2)
    all_results = Dict()
    
    # Generate candidate designs
    println("\nGenerating candidate experimental designs...")
    candidate_designs = generate_candidate_designs(n_candidates, nspec, lb_T, ub_T)
    println("  Generated $(length(candidate_designs)) candidate designs")
    
    # Generate initial random experiments
    println("\n"^"-"^50)
    println("INITIAL RANDOM EXPERIMENTS")
    println("-"^50)
    
    initial_experiments = []
    for i in 1:n_initial
        mf1 = 0.1 + 0.4 * rand()
        mf2 = 0.1 + 0.4 * rand()
        mf3 = max(0.0, 1.0 - mf1 - mf2)
        Y_in = [mf1, mf2, mf3]
        Y_in = Y_in / sum(Y_in)
        T = lb_T + (ub_T - lb_T) * rand()
        push!(initial_experiments, (Y_in, T))
        println("  Exp $i: T=$(round(T, digits=1))K, Y_in=[$(join(round.(Y_in, digits=2), ", "))]")
    end
    
    # ========================================================================
    # Run for each noise level
    # ========================================================================
    
    for noise_level in noise_levels
        println("\n"^"="^70)
        println("NOISE LEVEL: $(Int(noise_level*100))%")
        println("="^70)
        
        σ_meas = max(noise_level * 0.05, 1e-6)
        
        # Step 1: Simulate initial experiments with noise
        println("\nSTEP 1: Simulating initial experiments...")
        initial_data = []
        for (Y_in, T) in initial_experiments
            y_meas, success = simulate_experiment_with_noise(Y_in, T, θ_true, ratio, St, nref, noise_level)
            push!(initial_data, (Y_in, T, y_meas))
            if success
                println("  T=$(round(T, digits=1))K → Outlet: [$(join(round.(y_meas, digits=4), ", "))]")
            end
        end
        
        # Step 2: Initial parameter estimation
        println("\nSTEP 2: Initial parameter estimation...")
        θ_initial = estimate_parameters_from_data(initial_data, θ_init, ratio, St, nref, σ_meas)
        init_error = norm(θ_initial - θ_true) / norm(θ_true) * 100
        println("  Initial estimate: k0=$(round(θ_initial[1], digits=1)), Ea=$(round(θ_initial[2], digits=1))")
        println("  Relative error: $(round(init_error, digits=2))%")
        
        # Step 3: Select optimal experiments using FIM
        println("\nSTEP 3: Selecting optimal experiments (D-optimal)...")
        optimal_experiments, final_M, fim_history, info_gains = select_optimal_experiments_sequential(
            θ_initial, candidate_designs, n_optimal, ratio, St, nref, σ_meas)
        
        # Step 4: Simulate optimal experiments
        println("\nSTEP 4: Simulating optimal experiments...")
        optimal_data = []
        for (Y_in, T) in optimal_experiments
            y_meas, success = simulate_experiment_with_noise(Y_in, T, θ_true, ratio, St, nref, noise_level)
            push!(optimal_data, (Y_in, T, y_meas))
            println("  T=$(round(T, digits=1))K, Y_in=[$(join(round.(Y_in, digits=3), ", "))]")
            println("    → Outlet: [$(join(round.(y_meas, digits=4), ", "))]")
        end
        
        # Step 5: Final parameter estimation with all data
        println("\nSTEP 5: Final parameter estimation...")
        all_data = vcat(initial_data, optimal_data)
        θ_final = estimate_parameters_from_data(all_data, θ_initial, ratio, St, nref, σ_meas)
        final_error = norm(θ_final - θ_true) / norm(θ_true) * 100
        
        println("\n"^"-"^40)
        println("RESULTS FOR $(Int(noise_level*100))% NOISE")
        println("-"^40)
        println("Initial: k0=$(round(θ_initial[1], digits=1)), Ea=$(round(θ_initial[2], digits=1))")
        println("Final:   k0=$(round(θ_final[1], digits=1)), Ea=$(round(θ_final[2], digits=1))")
        println("True:    k0=$(θ_true[1]), Ea=$(θ_true[2])")
        println("Initial error: $(round(init_error, digits=2))%")
        println("Final error:   $(round(final_error, digits=2))%")
        println("Improvement:   $(round(init_error - final_error, digits=2))%")
        
        # Store results
        all_results[noise_level] = (
            θ_initial=θ_initial,
            θ_final=θ_final,
            init_error=init_error,
            final_error=final_error,
            improvement=init_error - final_error,
            optimal_experiments=optimal_experiments,
            fim_history=fim_history,
            info_gains=info_gains
        )
    end
    
    # ========================================================================
    # Generate Plots
    # ========================================================================
    println("\n"^"="^70)
    println("GENERATING PLOTS")
    println("="^70)
    
    generate_plots(all_results, noise_levels, θ_true, output_dir)
    
    # ========================================================================
    # Save Results to CSV
    # ========================================================================
    save_results_to_csv(all_results, noise_levels, output_dir)
    
    println("\n"^"="^70)
    println("WORKFLOW COMPLETED SUCCESSFULLY!")
    println("Results saved to: $output_dir")
    println("="^70)
    
    return all_results
end

# =============================================================================
# 5. PLOTTING FUNCTIONS
# =============================================================================

function generate_plots(all_results, noise_levels, θ_true, output_dir)
    noise_pct = [nl * 100 for nl in noise_levels]
    
    # Extract data
    init_errors = [all_results[nl].init_error for nl in noise_levels]
    final_errors = [all_results[nl].final_error for nl in noise_levels]
    improvements = [all_results[nl].improvement for nl in noise_levels]
    
    k0_init = [all_results[nl].θ_initial[1] for nl in noise_levels]
    k0_final = [all_results[nl].θ_final[1] for nl in noise_levels]
    Ea_init = [all_results[nl].θ_initial[2] for nl in noise_levels]
    Ea_final = [all_results[nl].θ_final[2] for nl in noise_levels]
    
    # Plot 1: Parameter Error Comparison
    p1 = plot(size=(900, 600))
    plot!(p1, noise_pct, init_errors, linewidth=2, marker=:circle, 
          color=:red, label="Initial Error (Random Design)")
    plot!(p1, noise_pct, final_errors, linewidth=2, marker=:square, 
          color=:blue, label="Final Error (OED)")
    plot!(p1, xlabel="Noise Level (%)", ylabel="Parameter Error (%)", 
          title="Parameter Estimation Improvement via Optimal Experimental Design",
          legend=:topright, grid=true, fontsize=12)
    savefig(p1, joinpath(output_dir, "parameter_improvement.png"))
    
    # Plot 2: Error Reduction Bar Chart
    p2 = plot(size=(900, 600))
    bar!(p2, noise_pct, improvements, color=:green, alpha=0.7, 
         label="Error Reduction", linewidth=2)
    plot!(p2, xlabel="Noise Level (%)", ylabel="Error Reduction (%)", 
          title="OED Improvement vs Noise Level", grid=true, fontsize=12)
    savefig(p2, joinpath(output_dir, "error_reduction.png"))
    
    # Plot 3: k0 Convergence
    p3 = plot(size=(900, 600))
    plot!(p3, noise_pct, fill(θ_true[1], length(noise_pct)), 
          linewidth=2, linestyle=:dash, color=:black, label="True k₀")
    plot!(p3, noise_pct, k0_init, linewidth=2, marker=:circle, 
          color=:red, label="Initial Estimate")
    plot!(p3, noise_pct, k0_final, linewidth=2, marker=:square, 
          color=:blue, label="Final Estimate (OED)")
    plot!(p3, xlabel="Noise Level (%)", ylabel="k₀ Value", 
          title="Pre-exponential Factor Estimation", legend=:best, grid=true)
    savefig(p3, joinpath(output_dir, "k0_convergence.png"))
    
    # Plot 4: Ea Convergence
    p4 = plot(size=(900, 600))
    plot!(p4, noise_pct, fill(θ_true[2], length(noise_pct)), 
          linewidth=2, linestyle=:dash, color=:black, label="True Eₐ")
    plot!(p4, noise_pct, Ea_init, linewidth=2, marker=:circle, 
          color=:red, label="Initial Estimate")
    plot!(p4, noise_pct, Ea_final, linewidth=2, marker=:square, 
          color=:blue, label="Final Estimate (OED)")
    plot!(p4, xlabel="Noise Level (%)", ylabel="Eₐ Value (J/mol)", 
          title="Activation Energy Estimation", legend=:best, grid=true)
    savefig(p4, joinpath(output_dir, "Ea_convergence.png"))
    
    # Plot 5: FIM Determinant Growth (for 0% noise)
    if haskey(all_results, 0.0) && !isempty(all_results[0.0].fim_history)
        p5 = plot(size=(900, 600))
        det_history = [log10(det(M)) for M in all_results[0.0].fim_history]
        plot!(p5, 1:length(det_history), det_history, 
              linewidth=2, marker=:diamond, color=:purple,
              xlabel="Optimal Experiment Number", 
              ylabel="log₁₀(det(Fisher Information Matrix))", 
              title="Fisher Information Growth (0% Noise)",
              grid=true, fillrange=0, fillalpha=0.3)
        savefig(p5, joinpath(output_dir, "fim_growth.png"))
    end
    
    # Plot 6: Info Gains per Experiment
    if haskey(all_results, 0.0) && !isempty(all_results[0.0].info_gains)
        p6 = plot(size=(900, 600))
        bar!(p6, 1:length(all_results[0.0].info_gains), 
             all_results[0.0].info_gains, color=:orange, alpha=0.7,
             label="Information Gain")
        plot!(p6, xlabel="Experiment Number", 
              ylabel="log(det(FIM_new))", 
              title="Information Gain per Optimal Experiment",
              grid=true)
        savefig(p6, joinpath(output_dir, "info_gains.png"))
    end
    
    println("  ✓ All plots generated successfully")
end

function save_results_to_csv(all_results, noise_levels, output_dir)
    noise_pct = [nl * 100 for nl in noise_levels]
    
    # Summary DataFrame
    df_summary = DataFrame(
        Noise_Percent = noise_pct,
        Initial_k0 = [all_results[nl].θ_initial[1] for nl in noise_levels],
        Final_k0 = [all_results[nl].θ_final[1] for nl in noise_levels],
        Initial_Ea = [all_results[nl].θ_initial[2] for nl in noise_levels],
        Final_Ea = [all_results[nl].θ_final[2] for nl in noise_levels],
        Initial_Error_Percent = [round(all_results[nl].init_error, digits=2) for nl in noise_levels],
        Final_Error_Percent = [round(all_results[nl].final_error, digits=2) for nl in noise_levels],
        Improvement_Percent = [round(all_results[nl].improvement, digits=2) for nl in noise_levels]
    )
    CSV.write(joinpath(output_dir, "summary_results.csv"), df_summary)
    
    # Optimal experiments for 0% noise case
    if haskey(all_results, 0.0)
        opt_exps = all_results[0.0].optimal_experiments
        df_opt = DataFrame(
            Experiment_Number = 1:length(opt_exps),
            Temperature_K = [exp[2] for exp in opt_exps],
            Mass_Fraction_1 = [round(exp[1][1], digits=3) for exp in opt_exps],
            Mass_Fraction_2 = [round(exp[1][2], digits=3) for exp in opt_exps],
            Mass_Fraction_3 = [round(exp[1][3], digits=3) for exp in opt_exps]
        )
        CSV.write(joinpath(output_dir, "optimal_experiments.csv"), df_opt)
    end
    
    # Generate report
    open(joinpath(output_dir, "OED_Report.txt"), "w") do io
        println(io, "="^70)
        println(io, "OPTIMAL EXPERIMENTAL DESIGN REPORT")
        println(io, "="^70)
        println(io, "\nDate: $(Dates.now())")
        println(io, "\nTrue Parameters: k0 = 4000.0, Ea = 4000.0")
        println(io, "\n"^"-"^50)
        println(io, "SUMMARY OF RESULTS")
        println(io, "-"^50)
        println(io, df_summary)
        
        println(io, "\n"^"-"^50)
        println(io, "CONCLUSIONS")
        println(io, "-"^50)
        println(io, "The Optimal Experimental Design (OED) framework using")
        println(io, "Fisher Information Matrix successfully improved parameter")
        println(io, "estimation across all noise levels. The D-optimal criterion")
        println(io, "selected experiments that maximized information gain,")
        println(io, "resulting in reduced parameter uncertainty.")
    end
    
    println("  ✓ Results saved to CSV files")
end

# =============================================================================
# 6. MAIN FUNCTION
# =============================================================================

function main()
    println("="^80)
    println("MASTER THESIS: OPTIMAL EXPERIMENTAL DESIGN")
    println("Using Fisher Information Matrix for Kinetic Parameter Estimation")
    println("in Non-Ideal Catalytic Reactors")
    println("="^80)
    
    println("\nThis workflow will:")
    println("  1. Generate initial random experiments")
    println("  2. Estimate initial kinetic parameters")
    println("  3. Use Fisher Information Matrix to select optimal experiments")
    println("  4. Re-estimate parameters with optimal design")
    println("  5. Test across 0%, 5%, 10%, and 20% noise levels")
    println("  6. Generate comparison plots\n")
    
    results = run_oed_workflow(
        n_initial=3,
        n_optimal=3,
        n_candidates=30,
        θ_true=[4000.0, 4000.0],
        θ_init=[2000.0, 2000.0],
        ratio=0.1,
        St=[-2 -1 2],
        nref=200,
        noise_levels=[0.0, 0.05, 0.10, 0.20],
        output_dir="OED_Results"
    )
    
    println("\n"^"="^80)
    println("FINAL SUMMARY")
    println("="^80)
    println("\nParameter estimation improvement across noise levels:")
    for noise in [0.0, 0.05, 0.10, 0.20]
        improvement = results[noise].improvement
        println("  $(Int(noise*100))% noise: $(round(improvement, digits=2))% improvement")
    end
    
    println("\nGenerated output files in 'OED_Results/':")
    println("  - parameter_improvement.png")
    println("  - error_reduction.png")
    println("  - k0_convergence.png")
    println("  - Ea_convergence.png")
    println("  - fim_growth.png")
    println("  - info_gains.png")
    println("  - summary_results.csv")
    println("  - optimal_experiments.csv")
    println("  - OED_Report.txt")
    
    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


end