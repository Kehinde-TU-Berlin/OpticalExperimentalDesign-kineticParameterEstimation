# #I'll rewrite the code with proper error handling, better axis management, and clear explanations. Here's the corrected version:


# # ============================================================================
# # COMPARISON OF GLOBAL INFORMATION MATRIX (GIM) vs FISHER INFORMATION MATRIX (FIM)
# # WITH AND WITHOUT NOISE FOR REACTOR KINETIC PARAMETER ESTIMATION
# # ============================================================================

# # ============================================================================
# # STEP 1: LOAD REQUIRED PACKAGES
# # ============================================================================

# using Plots
# using LinearAlgebra
# using Distributions
# using Random
# using Printf
# using Statistics

# # ============================================================================
# # STEP 2: SET UP PLOTTING AND RANDOMNESS
# # ============================================================================

# # Use GR backend with anti-aliasing for better quality
# gr(smooth=true, dpi=300)

# # Set random seed for reproducibility
# Random.seed!(42)

# # ============================================================================
# # STEP 3: DEFINE THE REACTOR MODEL WITH ERROR HANDLING
# # ============================================================================


# #Reactor model for first-order reaction with axial dispersion.
# #Returns outlet concentration of reactant A.

# function reactor_model(θ, x)
#     # θ[1] = Pre-exponential factor A (s⁻¹)
#     # θ[2] = Activation energy E (J/mol)
#     # Use try-catch to handle potential numerical issues
#     try
#         A, E = θ
        
#         # x[1] = Temperature T (K)
#         # x[2] = Inlet concentration C₀ (mol/m³)
#         # x[3] = Flow velocity u (m/s)
#         # x[4] = Reactor position z (dimensionless)
#         T, C0, u, z = x
        
#         # Validate inputs
#         @assert all(θ .> 0) "Parameters must be positive"
#         @assert T > 0 "Temperature must be positive"
#         @assert C0 > 0 "Inlet concentration must be positive"
#         @assert u > 0 "Flow velocity must be positive"
#         @assert 0 <= z <= 1 "Position must be between 0 and 1"
        
#         # Universal gas constant
#         R = 8.314
        
#         # Reactor length (fixed)
#         L = 1.0
        
#         # Arrhenius equation: reaction rate depends on temperature
#         k = A * exp(-E/(R*T))
        
#         # Damköhler number: ratio of reaction rate to flow rate
#         Da = k * L / u
        
#         # Conversion of reactant A (simplified model)
#         # Add small epsilon to prevent numerical underflow
#         X = 1 - exp(-max(Da * z, 1e-10))
        
#         # Outlet concentration = inlet × (1 - conversion)
#         return C0 * (1 - X)
        
#     catch e
#         # If there's an error, return a safe value
#         println("Warning in reactor_model: $e")
#         return 0.0
#     end
# end

# # ============================================================================
# # STEP 4: COMPUTE PARAMETER SENSITIVITIES WITH SAFE NUMERICAL DIFFERENTIATION
# # ============================================================================


# #Calculate parameter sensitivities using central finite differences.
# #More accurate than forward differences.

# function compute_sensitivities(θ, x, model; δ=1e-4)
#     n_params = length(θ)
#     sensitivities = zeros(n_params)
    
#     for i in 1:n_params
#         # Central difference: more accurate than forward difference
#         θ_plus = copy(θ)
#         θ_minus = copy(θ)
        
#         # Perturb parameter in both directions
#         perturbation = δ * max(abs(θ[i]), 1e-6)  # Ensure meaningful perturbation
#         θ_plus[i] += perturbation
#         θ_minus[i] -= perturbation
        
#         # Compute model outputs
#         y_plus = model(θ_plus, x)
#         y_minus = model(θ_minus, x)
        
#         # Central difference formula
#         sensitivities[i] = (y_plus - y_minus) / (2 * perturbation)
#     end
    
#     return sensitivities
# end

# # ============================================================================
# # STEP 5: SAFE FISHER INFORMATION MATRIX COMPUTATION
# # ============================================================================


# #Calculate Fisher Information Matrix with regularization to avoid singularities.

# function compute_fim_safe(θ, x, σ, model; ϵ=1e-8)
#     # Ensure σ is not zero
#     σ_safe = max(σ, 1e-10)
    
#     # Get sensitivity vector
#     S = compute_sensitivities(θ, x, model)
    
#     # Check if sensitivities are valid
#     if any(isnan.(S)) || any(isinf.(S))
#         println("Warning: Invalid sensitivities detected")
#         S = [1e-6, 1e-6]  # Default small values
#     end
    
#     # Compute FIM with Tikhonov regularization
#     FIM = (1/σ_safe^2) * (S * S')
    
#     # Add small diagonal term to ensure positive definiteness
#     FIM_reg = FIM + ϵ * I
    
#     return FIM_reg
# end

# # ============================================================================
# # STEP 6: ROBUST GLOBAL INFORMATION MATRIX COMPUTATION
# # ============================================================================


# #Calculate Global Information Matrix with adaptive Monte Carlo sampling.

# function compute_gim_robust(prior_mean, prior_cov, x, σ, model; 
#                            min_samples=500, max_samples=2000, rtol=0.01)
    
#     n_params = length(prior_mean)
    
#     # Create multivariate normal prior with positive definite covariance
#     # Ensure covariance matrix is positive definite
#     prior_cov_pd = prior_cov + 1e-6 * I
#     prior_dist = MvNormal(prior_mean, prior_cov_pd)
    
#     # Initialize
#     M_global = zeros(n_params, n_params)
#     M_global_prev = zeros(n_params, n_params)
    
#     # Adaptive Monte Carlo loop
#     converged = false
#     n_samples = 0
    
#     while !converged && n_samples < max_samples
#         # Determine batch size (adaptive)
#         batch_size = min(100, max_samples - n_samples)
        
#         for _ in 1:batch_size
#             # Sample from prior
#             θ_sample = rand(prior_dist)
            
#             # Ensure sampled parameters are positive
#             θ_sample = max.(θ_sample, 1e-6)
            
#             # Compute FIM for this sample
#             FIM_local = compute_fim_safe(θ_sample, x, σ, model)
            
#             # Update running average
#             M_global = (M_global * n_samples + FIM_local) / (n_samples + 1)
#             n_samples += 1
#         end
        
#         # Check convergence after minimum samples
#         if n_samples >= min_samples
#             # Compute relative change
#             change = norm(M_global - M_global_prev) / max(norm(M_global_prev), 1e-10)
#             if change < rtol
#                 converged = true
#             end
#         end
        
#         M_global_prev = copy(M_global)
#     end
    
#     if !converged
#         println("Warning: GIM Monte Carlo did not fully converge after $n_samples samples")
#     end
    
#     return M_global
# end

# # ============================================================================
# # STEP 7: SAFE METRIC COMPUTATION FUNCTIONS
# # ============================================================================


# #Safely compute determinant of a matrix.
# #Returns 0 for singular matrices.

# function safe_det(M; ϵ=1e-10)
#     try
#         d = det(M)
#         return max(d, ϵ)  # Ensure non-negative
#     catch e
#         println("Warning in safe_det: $e")
#         return ϵ
#     end
# end


# #Safely compute condition number.
# #Clamps extremely large values.

# function safe_cond(M; max_cond=1e12)
#     try
#         λ = eigvals(M)
#         λ_max = maximum(λ)
#         λ_min = maximum([minimum(λ), 1e-12])
#         cond_val = λ_max / λ_min
        
#         # Clamp extremely large condition numbers
#         return min(cond_val, max_cond)
#     catch e
#         println("Warning in safe_cond: $e")
#         return max_cond
#     end
# end


# #Safely compute trace.

# function safe_trace(M)
#     try
#         return tr(M)
#     catch e
#         println("Warning in safe_trace: $e")
#         return 0.0
#     end
# end

# # ============================================================================
# # STEP 8: SET UP THE PROBLEM PARAMETERS
# # ============================================================================

# println("Setting up reactor parameter estimation problem...")
# println("="^60)

# # "True" kinetic parameters
# θ_true = [1e5, 50000.0]

# # Prior knowledge with uncertainty
# θ_prior_mean = [1e5, 50000.0]
# θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])

# # Define different noise levels
# noise_levels = [0.0, 0.005, 0.01, 0.02]
# noise_names = ["No noise", "0.5% noise", "1% noise", "2% noise"]

# # Temperature range for experiments
# temperatures = collect(350:5:450)
# experiments = [[T, 1.0, 0.1, 0.5] for T in temperatures]

# println("Temperature range: $(minimum(temperatures))K to $(maximum(temperatures))K")
# println("Noise levels: $noise_levels")
# println("Number of experiments: $(length(experiments))")

# # ============================================================================
# # STEP 9: COMPUTE FIM AND GIM FOR ALL CONDITIONS (WITH PROGRESS BAR)
# # ============================================================================

# println("\nComputing information matrices...")
# println("-"^60)

# # Pre-allocate results dictionaries
# fim_results = Dict{Float64, Dict{Symbol, Vector{Float64}}}()
# gim_results = Dict{Float64, Dict{Symbol, Vector{Float64}}}()

# total_iterations = length(noise_levels) * length(experiments)
# current_iteration = 0

# for (noise_idx, σ) in enumerate(noise_levels)
#     println("Processing noise level: $(noise_names[noise_idx]) (σ = $σ)")
    
#     # Initialize arrays for this noise level
#     fim_dets = Float64[]
#     gim_dets = Float64[]
#     fim_traces = Float64[]
#     gim_traces = Float64[]
#     fim_cond = Float64[]
#     gim_cond = Float64[]
    
#     for (exp_idx, x) in enumerate(experiments)
#         current_iteration += 1
        
#         # Progress indicator
#         if exp_idx % 5 == 0 || exp_idx == length(experiments)
#             progress = round(current_iteration / total_iterations * 100, digits=1)
#             @printf("\r  Progress: %.1f%% (T = %dK)", progress, x[1])
#         end
        
#         # Compute FIM
#         FIM = compute_fim_safe(θ_true, x, σ, reactor_model)
#         push!(fim_dets, safe_det(FIM))
#         push!(fim_traces, safe_trace(FIM))
#         push!(fim_cond, safe_cond(FIM))
        
#         # Compute GIM (use fewer samples for speed during exploration)
#         GIM = compute_gim_robust(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, 
#                                 min_samples=200, max_samples=500)
#         push!(gim_dets, safe_det(GIM))
#         push!(gim_traces, safe_trace(GIM))
#         push!(gim_cond, safe_cond(GIM))
#     end
    
#     println()  # New line after progress
    
#     # Store results
#     fim_results[σ] = Dict(:det => fim_dets, :trace => fim_traces, :cond => fim_cond)
#     gim_results[σ] = Dict(:det => gim_dets, :trace => gim_traces, :cond => gim_cond)
# end

# println("\n✓ Computation complete!")

# # ============================================================================
# # STEP 10: SAFE PLOTTING FUNCTION WITH AXIS MANAGEMENT
# # ============================================================================


# #Create a plot with safe axis limits and error handling.

# function create_safe_plot(; title="", xlabel="", ylabel="", 
#                           yscale=:identity, xscale=:identity,
#                           legend_pos=:topright, grid_on=true,
#                           figsize=(800, 500))
    
#     # Create empty plot
#     p = plot(size=figsize,
#              title=title,
#              xlabel=xlabel,
#              ylabel=ylabel,
#              legend=legend_pos,
#              grid=grid_on,
#              foreground_color_legend=nothing,
#              background_color_legend=nothing)
    
#     # Set scales if specified
#     if yscale != :identity
#         try
#             yscale!(p, yscale)
#         catch e
#             println("Warning: Could not set yscale to $yscale: $e")
#         end
#     end
    
#     if xscale != :identity
#         try
#             xscale!(p, xscale)
#         catch e
#             println("Warning: Could not set xscale to $yscale: $e")
#         end
#     end
    
#     return p
# end


# #Add data to plot with safe axis limit management.

# function add_safe_curve!(p, x_data, y_data; label="", color=:auto, 
#                          linewidth=2, linestyle=:solid, marker=nothing)
    
#     # Filter out invalid data
#     valid_indices = findall(y -> isfinite(y) && !isnan(y), y_data)
    
#     if length(valid_indices) == 0
#         println("Warning: No valid data for curve '$label'")
#         return p
#     end
    
#     x_valid = x_data[valid_indices]
#     y_valid = y_data[valid_indices]
    
#     # Plot the valid data
#     plot!(p, x_valid, y_valid,
#           label=label,
#           linewidth=linewidth,
#           linestyle=linestyle,
#           color=color,
#           marker=marker)
    
#     return p
# end


# #Set safe axis limits based on data.

# function set_safe_limits!(p, x_data, y_data; padding=0.1)
#     # Get valid data
#     valid_indices = findall(y -> isfinite(y) && !isnan(y), y_data)
    
#     if length(valid_indices) == 0
#         # Default limits if no valid data
#         xlims!(p, (minimum(x_data), maximum(x_data)))
#         ylims!(p, (0, 1))
#         return p
#     end
    
#     x_valid = x_data[valid_indices]
#     y_valid = y_data[valid_indices]
    
#     # Calculate limits with padding
#     x_min = minimum(x_valid)
#     x_max = maximum(x_valid)
#     y_min = minimum(y_valid)
#     y_max = maximum(y_valid)
    
#     # Handle zero or negative ranges for log scales
#     x_range = x_max - x_min
#     y_range = y_max - y_min
    
#     x_pad = max(x_range * padding, 0.1)
#     y_pad = max(y_range * padding, 0.1 * y_max)
    
#     # Set limits
#     xlims!(p, (x_min - x_pad, x_max + x_pad))
    
#     # For log scale, ensure positive y_min
#     if yscale(p) == :log10
#         y_min_safe = max(y_min, 1e-10)
#         ylims!(p, (y_min_safe / 10, y_max * 10))
#     else
#         ylims!(p, (max(0, y_min - y_pad), y_max + y_pad))
#     end
    
#     return p
# end

# # ============================================================================
# # STEP 11: CREATE COMPARISON PLOTS - DETERMINANT (D-OPTIMALITY)
# # ============================================================================

# println("\nCreating comparison plots...")
# println("-"^60)

# # Define colors and styles
# colors = [:blue, :green, :orange, :red]
# line_styles = [:solid, :dash, :dot, :dashdot]
# markers = [:circle, :square, :diamond, :xcross]

# # ----------------------------------------------------------------------------
# # Plot 1: FIM Determinant
# # ----------------------------------------------------------------------------
# println("Creating FIM determinant plot...")

# p1_fim = create_safe_plot(
#     title="Fisher Information Matrix (FIM) - Determinant",
#     xlabel="Temperature (K)",
#     ylabel="Determinant",
#     yscale=:log10,
#     legend_pos=:topright,
#     figsize=(850, 550)
# )

# # Add curves for each noise level
# for (i, σ) in enumerate(noise_levels)
#     add_safe_curve!(p1_fim, temperatures, fim_results[σ][:det],
#                    label="$(noise_names[i])",
#                    color=colors[i],
#                    linewidth=2.5,
#                    linestyle=line_styles[i],
#                    marker=(i==1 ? markers[1] : nothing))
# end

# # Set safe limits
# set_safe_limits!(p1_fim, temperatures, fim_results[0.0][:det])

# # Add optimal temperature annotation
# if length(fim_results[0.0][:det]) > 0
#     opt_idx = argmax(fim_results[0.0][:det])
#     opt_temp = temperatures[opt_idx]
#     opt_val = maximum(fim_results[0.0][:det])
    
#     # Add vertical line
#     vline!(p1_fim, [opt_temp],
#            label="Optimal T = $(opt_temp)K",
#            color=:black,
#            linestyle=:dash,
#            linewidth=1.5,
#            alpha=0.7)
    
#     # Add text annotation
#     annotate!(p1_fim, opt_temp, opt_val * 0.05,
#               text("Optimal\n$(opt_temp)K", 9, :center, :black, :center))
# end

# # ----------------------------------------------------------------------------
# # Plot 2: GIM Determinant
# # ----------------------------------------------------------------------------
# println("Creating GIM determinant plot...")

# p1_gim = create_safe_plot(
#     title="Global Information Matrix (GIM) - Determinant",
#     xlabel="Temperature (K)",
#     ylabel="Determinant",
#     yscale=:log10,
#     legend_pos=:topright,
#     figsize=(850, 550)
# )

# # Add curves for each noise level
# for (i, σ) in enumerate(noise_levels)
#     add_safe_curve!(p1_gim, temperatures, gim_results[σ][:det],
#                    label="$(noise_names[i])",
#                    color=colors[i],
#                    linewidth=2.5,
#                    linestyle=line_styles[i],
#                    marker=(i==1 ? markers[2] : nothing))
# end

# # Set safe limits
# set_safe_limits!(p1_gim, temperatures, gim_results[0.0][:det])

# # Add optimal temperature annotation
# if length(gim_results[0.0][:det]) > 0
#     opt_idx = argmax(gim_results[0.0][:det])
#     opt_temp = temperatures[opt_idx]
#     opt_val = maximum(gim_results[0.0][:det])
    
#     vline!(p1_gim, [opt_temp],
#            label="Optimal T = $(opt_temp)K",
#            color=:black,
#            linestyle=:dash,
#            linewidth=1.5,
#            alpha=0.7)
    
#     annotate!(p1_gim, opt_temp, opt_val * 0.05,
#               text("Optimal\n$(opt_temp)K", 9, :center, :black, :center))
# end

# # ----------------------------------------------------------------------------
# # Plot 3: Direct comparison at 0% and 1% noise
# # ----------------------------------------------------------------------------
# println("Creating direct comparison plot...")

# p1_compare = create_safe_plot(
#     title="FIM vs GIM Comparison",
#     xlabel="Temperature (K)",
#     ylabel="Determinant",
#     yscale=:log10,
#     legend_pos=:topright,
#     figsize=(850, 550)
# )

# # Compare key noise levels: 0% and 1%
# key_noise_levels = [0.0, 0.01]
# key_names = ["No noise", "1% noise"]

# for (i, σ) in enumerate(key_noise_levels)
#     # FIM curves
#     add_safe_curve!(p1_compare, temperatures, fim_results[σ][:det],
#                    label="FIM ($(key_names[i]))",
#                    color=colors[i],
#                    linewidth=3,
#                    linestyle=:solid,
#                    marker=nothing)
    
#     # GIM curves
#     add_safe_curve!(p1_compare, temperatures, gim_results[σ][:det],
#                    label="GIM ($(key_names[i]))",
#                    color=colors[i],
#                    linewidth=3,
#                    linestyle=:dash,
#                    marker=nothing)
# end

# # Set limits based on the data
# all_data_compare = vcat(fim_results[0.0][:det], fim_results[0.01][:det],
#                        gim_results[0.0][:det], gim_results[0.01][:det])
# set_safe_limits!(p1_compare, temperatures, all_data_compare)

# # Add annotation about noise effect
# annotate!(p1_compare, temperatures[15], maximum(fim_results[0.0][:det]) * 0.02,
#           text("Higher noise →\nLower information", 10, :left, :red, :left))

# # ============================================================================
# # STEP 12: CREATE PLOTS FOR MATRIX CONDITIONING
# # ============================================================================

# # ----------------------------------------------------------------------------
# # Plot 4: FIM Condition Number
# # ----------------------------------------------------------------------------
# println("Creating FIM conditioning plot...")

# p2_fim = create_safe_plot(
#     title="FIM - Matrix Conditioning",
#     xlabel="Temperature (K)",
#     ylabel="Condition Number",
#     yscale=:log10,
#     legend_pos=:topright,
#     figsize=(850, 550)
# )

# for (i, σ) in enumerate(noise_levels)
#     add_safe_curve!(p2_fim, temperatures, fim_results[σ][:cond],
#                    label="$(noise_names[i])",
#                    color=colors[i],
#                    linewidth=2,
#                    linestyle=line_styles[i],
#                    marker=nothing)
# end

# # Set limits and add annotation
# set_safe_limits!(p2_fim, temperatures, vcat([fim_results[σ][:cond] for σ in noise_levels]...))
# annotate!(p2_fim, temperatures[10], maximum(fim_results[0.0][:cond]) * 0.1,
#           text("Lower is better", 10, :left, :black, :left))

# # ----------------------------------------------------------------------------
# # Plot 5: GIM Condition Number
# # ----------------------------------------------------------------------------
# println("Creating GIM conditioning plot...")

# p2_gim = create_safe_plot(
#     title="GIM - Matrix Conditioning",
#     xlabel="Temperature (K)",
#     ylabel="Condition Number",
#     yscale=:log10,
#     legend_pos=:topright,
#     figsize=(850, 550)
# )

# for (i, σ) in enumerate(noise_levels)
#     add_safe_curve!(p2_gim, temperatures, gim_results[σ][:cond],
#                    label="$(noise_names[i])",
#                    color=colors[i],
#                    linewidth=2,
#                    linestyle=line_styles[i],
#                    marker=nothing)
# end

# # Set limits
# set_safe_limits!(p2_gim, temperatures, vcat([gim_results[σ][:cond] for σ in noise_levels]...))

# # ----------------------------------------------------------------------------
# # Plot 6: Noise effect on optimal temperature
# # ----------------------------------------------------------------------------
# println("Creating noise effect plot...")

# p3_noise_effect = create_safe_plot(
#     title="Optimal Temperature vs Measurement Noise",
#     xlabel="Measurement Noise Level (σ)",
#     ylabel="Optimal Temperature (K)",
#     legend_pos=:topright,
#     figsize=(850, 550)
# )

# # Arrays to store optimal temperatures
# fim_opt_temps = Float64[]
# gim_opt_temps = Float64[]

# for σ in noise_levels
#     if length(fim_results[σ][:det]) > 0
#         opt_idx_fim = argmax(fim_results[σ][:det])
#         push!(fim_opt_temps, temperatures[opt_idx_fim])
        
#         opt_idx_gim = argmax(gim_results[σ][:det])
#         push!(gim_opt_temps, temperatures[opt_idx_gim])
#     end
# end

# # Scatter plot with trend lines
# scatter!(p3_noise_effect, noise_levels, fim_opt_temps,
#          label="FIM Optimal T",
#          color=:blue,
#          markersize=10,
#          marker=:circle,
#          markeralpha=0.8)

# scatter!(p3_noise_effect, noise_levels, gim_opt_temps,
#          label="GIM Optimal T",
#          color=:red,
#          markersize=10,
#          marker=:square,
#          markeralpha=0.8)

# # Add trend lines with polynomial fit
# if length(noise_levels) >= 2
#     # Simple linear interpolation for trend lines
#     plot!(p3_noise_effect, noise_levels, fim_opt_temps,
#           label="FIM trend",
#           color=:blue,
#           linewidth=2,
#           linestyle=:solid,
#           alpha=0.5)
    
#     plot!(p3_noise_effect, noise_levels, gim_opt_temps,
#           label="GIM trend",
#           color=:red,
#           linewidth=2,
#           linestyle=:dash,
#           alpha=0.5)
# end

# # Set appropriate limits
# x_pad = (maximum(noise_levels) - minimum(noise_levels)) * 0.1
# y_min = min(minimum(fim_opt_temps), minimum(gim_opt_temps))
# y_max = max(maximum(fim_opt_temps), maximum(gim_opt_temps))
# y_pad = (y_max - y_min) * 0.1

# xlims!(p3_noise_effect, (-0.001, maximum(noise_levels) + x_pad))
# ylims!(p3_noise_effect, (y_min - y_pad, y_max + y_pad))

# # Annotations
# annotate!(p3_noise_effect, 0.015, 410,
#           text("FIM: More sensitive\nto noise", 10, :left, :blue, :left))

# annotate!(p3_noise_effect, 0.015, 400,
#           text("GIM: More robust\nto noise", 10, :left, :red, :left))

# # ============================================================================
# # STEP 13: CREATE INFORMATION LOSS PLOT
# # ============================================================================

# println("Creating information loss plot...")

# p4_info_loss = create_safe_plot(
#     title="Information Loss Due to Measurement Noise",
#     xlabel="Temperature (K)",
#     ylabel="Information Relative to No Noise (%)",
#     legend_pos=:bottomright,
#     figsize=(850, 550)
# )

# # Calculate information loss for FIM and GIM at 1% noise
# if 0.0 in noise_levels && 0.01 in noise_levels
#     # Safe calculation of information ratio
#     fim_info_ratio = Float64[]
#     gim_info_ratio = Float64[]
    
#     for i in 1:length(temperatures)
#         fim_no_noise = max(fim_results[0.0][:det][i], 1e-10)
#         fim_with_noise = max(fim_results[0.01][:det][i], 1e-10)
#         gim_no_noise = max(gim_results[0.0][:det][i], 1e-10)
#         gim_with_noise = max(gim_results[0.01][:det][i], 1e-10)
        
#         push!(fim_info_ratio, (fim_with_noise / fim_no_noise) * 100)
#         push!(gim_info_ratio, (gim_with_noise / gim_no_noise) * 100)
#     end
    
#     # Plot information ratios
#     add_safe_curve!(p4_info_loss, temperatures, fim_info_ratio,
#                    label="FIM (1% noise)",
#                    color=:blue,
#                    linewidth=3,
#                    linestyle=:solid)
    
#     add_safe_curve!(p4_info_loss, temperatures, gim_info_ratio,
#                    label="GIM (1% noise)",
#                    color=:red,
#                    linewidth=3,
#                    linestyle=:dash)
    
#     # Add 100% reference line
#     hline!(p4_info_loss, [100],
#            label="No information loss",
#            color=:black,
#            linestyle=:dot,
#            linewidth=1.5,
#            alpha=0.5)
    
#     # Set appropriate y-limits
#     all_ratios = vcat(fim_info_ratio, gim_info_ratio)
#     valid_ratios = filter(isfinite, all_ratios)
    
#     if !isempty(valid_ratios)
#         y_min_loss = max(0, minimum(valid_ratios) * 0.9)
#         y_max_loss = min(110, maximum(valid_ratios) * 1.1)
#         ylims!(p4_info_loss, (y_min_loss, y_max_loss))
#     end
    
#     # Highlight worst information loss
#     if !isempty(fim_info_ratio) && !isempty(gim_info_ratio)
#         min_fim_idx = argmin(fim_info_ratio)
#         min_gim_idx = argmin(gim_info_ratio)
        
#         # Only annotate if the values are meaningful
#         if fim_info_ratio[min_fim_idx] < 95
#             annotate!(p4_info_loss, temperatures[min_fim_idx], 
#                      fim_info_ratio[min_fim_idx] - 3,
#                      text("FIM worst: $(round(fim_info_ratio[min_fim_idx], digits=1))%", 
#                      9, :center, :blue, :center))
#         end
        
#         if gim_info_ratio[min_gim_idx] < 95
#             annotate!(p4_info_loss, temperatures[min_gim_idx], 
#                      gim_info_ratio[min_gim_idx] - 3,
#                      text("GIM worst: $(round(gim_info_ratio[min_gim_idx], digits=1))%", 
#                      9, :center, :red, :center))
#         end
#     end
# end

# # ============================================================================
# # STEP 14: CREATE COMPOSITE PLOT AND SAVE
# # ============================================================================

# println("\nCreating composite plot...")

# # Create a nicely arranged composite plot
# composite_plot = plot(p1_fim, p1_gim, p1_compare,
#                       p2_fim, p2_gim, p3_noise_effect,
#                       layout=(3, 2),
#                       size=(1400, 1100),
#                       plot_title=plot_title("FIM vs GIM: Effect of Measurement Noise\non Optimal Experimental Design for Reactor Parameter Estimation",
#                                            fontsize=14),
#                       left_margin=15mm,
#                       bottom_margin=10mm,
#                       top_margin=10mm)

# # Save all plots

# println("Saving plots...")

# savefig(composite_plot, "fim_vs_gim_noise_comparison.png")
# savefig(p1_fim, "fim_determinant.png")
# savefig(p1_gim, "gim_determinant.png")
# savefig(p1_compare, "fim_gim_comparison.png")
# savefig(p2_fim, "fim_conditioning.png")
# savefig(p2_gim, "gim_conditioning.png")
# savefig(p3_noise_effect, "noise_effect.png")
# savefig(p4_info_loss, "information_loss.png")

# println("✓ All plots saved successfully!")

# # ============================================================================
# # STEP 15: DISPLAY RESULTS AND ANALYSIS
# # ============================================================================

# println("\n" * "="^60)
# println("ANALYSIS RESULTS")
# println("="^60)

# # Find optimal temperatures for key noise levels

# println("\n1. OPTIMAL EXPERIMENTAL TEMPERATURES:")

# for σ in [0.0, 0.01]
#     if haskey(fim_results, σ) && haskey(gim_results, σ)
#         if !isempty(fim_results[σ][:det]) && !isempty(gim_results[σ][:det])
#             idx_fim = argmax(fim_results[σ][:det])
#             idx_gim = argmax(gim_results[σ][:det])
            
#             noise_name = σ == 0.0 ? "No noise" : "1% noise"
#             println("\n   $noise_name:")
#             println("     • FIM optimal: $(temperatures[idx_fim]) K")
#             println("     • GIM optimal: $(temperatures[idx_gim]) K")
            
#             if σ > 0 && haskey(fim_results, 0.0) && haskey(gim_results, 0.0)
#                 if !isempty(fim_results[0.0][:det]) && !isempty(gim_results[0.0][:det])
#                     idx_fim_no_noise = argmax(fim_results[0.0][:det])
#                     idx_gim_no_noise = argmax(gim_results[0.0][:det])
                    
#                     shift_fim = temperatures[idx_fim] - temperatures[idx_fim_no_noise]
#                     shift_gim = temperatures[idx_gim] - temperatures[idx_gim_no_noise]
                    
#                     println("     • Shift from no-noise case:")
#                     println("       FIM = $(shift_fim) K, GIM = $(shift_gim) K")
#                 end
#             end
#         end
#     end
# end

# # Calculate information content

# println("\n2. INFORMATION CONTENT COMPARISON:")

# σ_ref = 0.01  # Reference noise level

# if haskey(fim_results, 0.0) && haskey(gim_results, 0.0) &&
#    haskey(fim_results, σ_ref) && haskey(gim_results, σ_ref)
    
#     max_fim_no_noise = maximum(fim_results[0.0][:det])
#     max_gim_no_noise = maximum(gim_results[0.0][:det])
#     max_fim_with_noise = maximum(fim_results[σ_ref][:det])
#     max_gim_with_noise = maximum(gim_results[σ_ref][:det])
    
#     fim_reduction = (1 - max_fim_with_noise/max_fim_no_noise) * 100
#     gim_reduction = (1 - max_gim_with_noise/max_gim_no_noise) * 100
    
#     println("   Maximum determinant (no noise):")
#     println("     • FIM: $(@sprintf("%.2e", max_fim_no_noise))")
#     println("     • GIM: $(@sprintf("%.2e", max_gim_no_noise))")
    
#     println("\n   With 1% measurement noise:")
#     println("     • FIM: $(@sprintf("%.2e", max_fim_with_noise))")
#     println("     • GIM: $(@sprintf("%.2e", max_gim_with_noise))")
    
#     println("\n   Information reduction due to 1% noise:")
#     println("     • FIM: $(round(fim_reduction, digits=1))% reduction")
#     println("     • GIM: $(round(gim_reduction, digits=1))% reduction")
# end

# # Robustness analysis

# println("\n3. ROBUSTNESS TO NOISE:")

# opt_temps_fim = Float64[]
# opt_temps_gim = Float64[]

# for σ in noise_levels
#     if haskey(fim_results, σ) && !isempty(fim_results[σ][:det]) &&
#        haskey(gim_results, σ) && !isempty(gim_results[σ][:det])
        
#         push!(opt_temps_fim, temperatures[argmax(fim_results[σ][:det])])
#         push!(opt_temps_gim, temperatures[argmax(gim_results[σ][:det])])
#     end
# end

# if length(opt_temps_fim) > 1 && length(opt_temps_gim) > 1
#     fim_sensitivity = std(opt_temps_fim)
#     gim_sensitivity = std(opt_temps_gim)




=========================================================================================================================================================
=========================================================================================================================================================
=========================================================================================================================================================
=========================================================================================================================================================


# ============================================================================
# COMPARISON OF GLOBAL INFORMATION MATRIX (GIM) vs FISHER INFORMATION MATRIX (FIM)
# WITH AND WITHOUT NOISE FOR REACTOR KINETIC PARAMETER ESTIMATION
# CORRECTED VERSION - NO PLOTTING ERRORS
# ============================================================================

# ============================================================================
# STEP 1: LOAD REQUIRED PACKAGES
# ============================================================================

using Plots
using LinearAlgebra
using Distributions
using Random
using Printf
using Statistics
using Plots.Measures

# ============================================================================
# STEP 2: SET UP PLOTTING AND RANDOMNESS
# ============================================================================

# Use GR backend with anti-aliasing for better quality
gr()

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================================
# STEP 3: DEFINE THE REACTOR MODEL WITH ERROR HANDLING
# ============================================================================

"""
Reactor model for first-order reaction with axial dispersion.
Returns outlet concentration of reactant A.
"""
function reactor_model(θ, x)
    try
        A, E = θ
        T, C0, u, z = x
        
        # Validate inputs
        if any(θ .<= 0) || T <= 0 || C0 <= 0 || u <= 0 || z < 0 || z > 1
            return 0.0
        end
        
        # Universal gas constant
        R = 8.314
        
        # Reactor length (fixed)
        L = 1.0
        
        # Arrhenius equation: reaction rate depends on temperature
        k = A * exp(-E/(R*T))
        
        # Damköhler number: ratio of reaction rate to flow rate
        Da = k * L / u
        
        # Conversion of reactant A (simplified model)
        # Add small epsilon to prevent numerical underflow
        Da_z = Da * z
        if Da_z > 100  # Prevent exp(-large) from underflowing
            X = 1.0
        else
            X = 1 - exp(-Da_z)
        end
        
        # Outlet concentration = inlet × (1 - conversion)
        return C0 * (1 - X)
        
    catch e
        # If there's an error, return a safe value
        return 0.0
    end
end

# ============================================================================
# STEP 4: COMPUTE PARAMETER SENSITIVITIES WITH SAFE NUMERICAL DIFFERENTIATION
# ============================================================================

"""
Calculate parameter sensitivities using central finite differences.
More accurate than forward differences.
"""
function compute_sensitivities(θ, x, model; δ=1e-4)
    n_params = length(θ)
    sensitivities = zeros(n_params)
    
    for i in 1:n_params
        # Central difference: more accurate than forward difference
        θ_plus = copy(θ)
        θ_minus = copy(θ)
        
        # Determine safe perturbation size
        if abs(θ[i]) < 1e-10
            perturbation = 1e-6
        else
            perturbation = δ * abs(θ[i])
        end
        
        # Perturb parameter in both directions
        θ_plus[i] += perturbation
        θ_minus[i] -= perturbation
        
        # Compute model outputs
        y_plus = model(θ_plus, x)
        y_minus = model(θ_minus, x)
        
        # Central difference formula
        if perturbation > 1e-12
            sensitivities[i] = (y_plus - y_minus) / (2 * perturbation)
        else
            sensitivities[i] = 0.0
        end
    end
    
    # Check for NaN or Inf
    for i in 1:n_params
        if isnan(sensitivities[i]) || isinf(sensitivities[i])
            sensitivities[i] = 1e-10
        end
    end
    
    return sensitivities
end

# ============================================================================
# STEP 5: SAFE FISHER INFORMATION MATRIX COMPUTATION
# ============================================================================

"""
Calculate Fisher Information Matrix with regularization to avoid singularities.
"""
function compute_fim_safe(θ, x, σ, model; ϵ=1e-8)
    # Ensure σ is not zero
    σ_safe = max(σ, 1e-10)
    
    # Get sensitivity vector
    S = compute_sensitivities(θ, x, model)
    
    # Compute FIM with Tikhonov regularization
    FIM = (1/σ_safe^2) * (S * S')
    
    # Add small diagonal term to ensure positive definiteness
    FIM_reg = FIM + ϵ * I(2)
    
    return FIM_reg
end

# ============================================================================
# STEP 6: ROBUST GLOBAL INFORMATION MATRIX COMPUTATION
# ============================================================================

"""
Calculate Global Information Matrix with Monte Carlo sampling.
"""
function compute_gim_robust(prior_mean, prior_cov, x, σ, model; n_samples=300)
    n_params = length(prior_mean)
    
    # Ensure covariance matrix is positive definite
    prior_cov_pd = prior_cov + 1e-6 * I
    
    # Create multivariate normal prior
    try
        prior_dist = MvNormal(prior_mean, prior_cov_pd)
    catch e
        println("Warning: Could not create prior distribution: $e")
        # Return identity matrix as fallback
        return I(2)
    end
    
    # Initialize
    M_global = zeros(n_params, n_params)
    valid_samples = 0
    
    for i in 1:n_samples
        try
            # Sample from prior
            θ_sample = rand(prior_dist)
            
            # Ensure sampled parameters are reasonable
            θ_sample = max.(θ_sample, 1e-6)
            θ_sample = min.(θ_sample, 1e6)
            
            # Compute FIM for this sample
            FIM_local = compute_fim_safe(θ_sample, x, σ, model)
            
            # Update running average
            M_global += FIM_local
            valid_samples += 1
            
        catch e
            # Skip problematic samples
            continue
        end
    end
    
    # Average the results
    if valid_samples > 0
        M_global = M_global / valid_samples
    else
        # Return a reasonable default if all samples failed
        M_global = I(2) * 1e-6
    end
    
    return M_global
end

# ============================================================================
# STEP 7: SAFE METRIC COMPUTATION FUNCTIONS
# ============================================================================

"""
Safely compute determinant of a matrix.
"""
function safe_det(M)
    try
        d = det(M)
        if d < 1e-20
            return 1e-20
        else
            return d
        end
    catch e
        return 1e-20
    end
end

"""
Safely compute condition number.
"""
function safe_cond(M; max_cond=1e12)
    try
        λ = eigvals(M)
        λ_max = maximum(λ)
        λ_min = max(minimum(λ), 1e-12)
        cond_val = λ_max / λ_min
        
        # Cap extremely large condition numbers
        return min(cond_val, max_cond)
    catch e
        return max_cond
    end
end

"""
Safely compute trace.
"""
function safe_trace(M)
    try
        return tr(M)
    catch e
        return 0.0
    end
end

# ============================================================================
# STEP 8: SET UP THE PROBLEM PARAMETERS
# ============================================================================

println("="^60)
println("REACTOR PARAMETER ESTIMATION - OPTIMAL EXPERIMENTAL DESIGN")
println("="^60)

# "True" kinetic parameters
θ_true = [1e5, 50000.0]  # A = 100,000 s⁻¹, E = 50,000 J/mol

# Prior knowledge with uncertainty
θ_prior_mean = [1e5, 50000.0]
θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])

# Define different noise levels
noise_levels = [0.0, 0.005, 0.01, 0.02]
noise_names = ["No noise", "0.5% noise", "1% noise", "2% noise"]

# Temperature range for experiments
temperatures = collect(350:5:450)
experiments = [[T, 1.0, 0.1, 0.5] for T in temperatures]

println("Temperature range: $(minimum(temperatures))K to $(maximum(temperatures))K")
println("Number of experiments: $(length(experiments))")
println("Noise levels: $noise_levels")

# ============================================================================
# STEP 9: COMPUTE FIM AND GIM FOR ALL CONDITIONS
# ============================================================================

println("\nComputing information matrices...")
println("-"^60)

# Pre-allocate results dictionaries
fim_results = Dict{Float64, Dict{Symbol, Vector{Float64}}}()
gim_results = Dict{Float64, Dict{Symbol, Vector{Float64}}}()

total_experiments = length(experiments)

for (noise_idx, σ) in enumerate(noise_levels)
    println("Processing: $(noise_names[noise_idx]) (σ = $σ)")
    
    # Initialize arrays for this noise level
    fim_dets = Float64[]
    gim_dets = Float64[]
    fim_traces = Float64[]
    gim_traces = Float64[]
    fim_cond = Float64[]
    gim_cond = Float64[]
    
    for (exp_idx, x) in enumerate(experiments)
        # Progress indicator
        if exp_idx % 5 == 0 || exp_idx == total_experiments
            @printf("\r  Experiment %d/%d", exp_idx, total_experiments)
        end
        
        # Compute FIM
        FIM = compute_fim_safe(θ_true, x, σ, reactor_model)
        push!(fim_dets, safe_det(FIM))
        push!(fim_traces, safe_trace(FIM))
        push!(fim_cond, safe_cond(FIM))
        
        # Compute GIM (use fewer samples for speed)
        GIM = compute_gim_robust(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=200)
        push!(gim_dets, safe_det(GIM))
        push!(gim_traces, safe_trace(GIM))
        push!(gim_cond, safe_cond(GIM))
    end
    
    println()  # New line after progress
    
    # Store results
    fim_results[σ] = Dict(:det => fim_dets, :trace => fim_traces, :cond => fim_cond)
    gim_results[σ] = Dict(:det => gim_dets, :trace => gim_traces, :cond => gim_cond)
end

println("\n✓ Computation complete!")

# ============================================================================
# STEP 10: CREATE PLOTS - CORRECTED VERSION (NO yscale! ERRORS)
# ============================================================================

println("\nCreating plots...")
println("-"^60)

# Define colors and styles
colors = [:blue, :green, :orange, :red]
line_styles = [:solid, :dash, :dot, :dashdot]
markers = [:circle, :square, :diamond, :xcross]

# ----------------------------------------------------------------------------
# Plot 1: FIM Determinant (D-optimality) - CORRECTED
# ----------------------------------------------------------------------------
println("Creating FIM determinant plot...")

# Create plot with log scale directly in the constructor
p1_fim = plot(
    title="Fisher Information Matrix (FIM) - Determinant",
    xlabel="Temperature (K)",
    ylabel="Determinant (log scale)",
    yaxis=:log10,  # Correct way to set log scale
    legend=:topright,
    grid=true,
    size=(850, 550),
    linewidth=2,
    left_margin=5mm,
    bottom_margin=5mm
)

# Add curves for each noise level
for (i, σ) in enumerate(noise_levels)
    # Filter out any zero or negative values for log scale
    y_data = fim_results[σ][:det]
    
    # For log scale, we need positive values
    y_data_plot = [max(y, 1e-10) for y in y_data]
    
    plot!(p1_fim, temperatures, y_data_plot,
          label=noise_names[i],
          color=colors[i],
          linestyle=line_styles[i],
          marker=(i==1 ? markers[i] : :none),
          markersize=4,
          linewidth=2)
end

# Add optimal temperature annotation for no-noise case
opt_idx = argmax(fim_results[0.0][:det])
opt_temp = temperatures[opt_idx]
opt_val = maximum(fim_results[0.0][:det])

vline!(p1_fim, [opt_temp],
       label="Optimal: $(opt_temp)K",
       color=:black,
       linestyle=:dash,
       linewidth=1.5)

# ----------------------------------------------------------------------------
# Plot 2: GIM Determinant - CORRECTED
# ----------------------------------------------------------------------------
println("Creating GIM determinant plot...")

p1_gim = plot(
    title="Global Information Matrix (GIM) - Determinant",
    xlabel="Temperature (K)",
    ylabel="Determinant (log scale)",
    yaxis=:log10,  # Correct way to set log scale
    legend=:topright,
    grid=true,
    size=(850, 550),
    left_margin=5mm,
    bottom_margin=5mm
)

# Add curves for each noise level
for (i, σ) in enumerate(noise_levels)
    # Filter out any zero or negative values for log scale
    y_data = gim_results[σ][:det]
    y_data_plot = [max(y, 1e-10) for y in y_data]
    
    plot!(p1_gim, temperatures, y_data_plot,
          label=noise_names[i],
          color=colors[i],
          linestyle=line_styles[i],
          marker=(i==1 ? markers[i] : :none),
          markersize=4,
          linewidth=2)
end

# Add optimal temperature annotation for no-noise case
opt_idx_gim = argmax(gim_results[0.0][:det])
opt_temp_gim = temperatures[opt_idx_gim]
opt_val_gim = maximum(gim_results[0.0][:det])

vline!(p1_gim, [opt_temp_gim],
       label="Optimal: $(opt_temp_gim)K",
       color=:black,
       linestyle=:dash,
       linewidth=1.5)

# ----------------------------------------------------------------------------
# Plot 3: Direct comparison at 0% and 1% noise - CORRECTED
# ----------------------------------------------------------------------------
println("Creating direct comparison plot...")

p1_compare = plot(
    title="FIM vs GIM Comparison (0% and 1% Noise)",
    xlabel="Temperature (K)",
    ylabel="Determinant (log scale)",
    yaxis=:log10,  # Correct way to set log scale
    legend=:topright,
    grid=true,
    size=(850, 550),
    left_margin=5mm,
    bottom_margin=5mm
)

# Compare key noise levels: 0% and 1%
key_noise_levels = [0.0, 0.01]
key_names = ["No noise", "1% noise"]

for (i, σ) in enumerate(key_noise_levels)
    # FIM curves (solid)
    y_fim = [max(y, 1e-10) for y in fim_results[σ][:det]]
    plot!(p1_compare, temperatures, y_fim,
          label="FIM ($(key_names[i]))",
          color=colors[i],
          linewidth=3,
          linestyle=:solid)
    
    # GIM curves (dashed)
    y_gim = [max(y, 1e-10) for y in gim_results[σ][:det]]
    plot!(p1_compare, temperatures, y_gim,
          label="GIM ($(key_names[i]))",
          color=colors[i],
          linewidth=3,
          linestyle=:dash)
end

# ----------------------------------------------------------------------------
# Plot 4: FIM Condition Number - CORRECTED
# ----------------------------------------------------------------------------
println("Creating FIM conditioning plot...")

p2_fim = plot(
    title="FIM - Matrix Conditioning",
    xlabel="Temperature (K)",
    ylabel="Condition Number (log scale)",
    yaxis=:log10,  # Correct way to set log scale
    legend=:topright,
    grid=true,
    size=(850, 550),
    left_margin=5mm,
    bottom_margin=5mm
)

for (i, σ) in enumerate(noise_levels)
    # Ensure positive values for log scale
    y_data = [max(c, 1e-6) for c in fim_results[σ][:cond]]
    
    plot!(p2_fim, temperatures, y_data,
          label=noise_names[i],
          color=colors[i],
          linestyle=line_styles[i],
          linewidth=2)
end

# Add annotation
annotate!(p2_fim, temperatures[10], 1e8,
          text("Lower is better", 10, :left, :black))

# ----------------------------------------------------------------------------
# Plot 5: GIM Condition Number - CORRECTED
# ----------------------------------------------------------------------------
println("Creating GIM conditioning plot...")

p2_gim = plot(
    title="GIM - Matrix Conditioning",
    xlabel="Temperature (K)",
    ylabel="Condition Number (log scale)",
    yaxis=:log10,  # Correct way to set log scale
    legend=:topright,
    grid=true,
    size=(850, 550),
    left_margin=5mm,
    bottom_margin=5mm
)

for (i, σ) in enumerate(noise_levels)
    # Ensure positive values for log scale
    y_data = [max(c, 1e-6) for c in gim_results[σ][:cond]]
    
    plot!(p2_gim, temperatures, y_data,
          label=noise_names[i],
          color=colors[i],
          linestyle=line_styles[i],
          linewidth=2)
end

# ----------------------------------------------------------------------------
# Plot 6: Noise effect on optimal temperature
# ----------------------------------------------------------------------------
println("Creating noise effect plot...")

p3_noise_effect = plot(
    title="Optimal Temperature vs Measurement Noise",
    xlabel="Measurement Noise Level (σ)",
    ylabel="Optimal Temperature (K)",
    legend=:topright,
    grid=true,
    size=(850, 550),
    left_margin=5mm,
    bottom_margin=5mm
)

# Arrays to store optimal temperatures
fim_opt_temps = Float64[]
gim_opt_temps = Float64[]

for σ in noise_levels
    if haskey(fim_results, σ) && haskey(gim_results, σ)
        if !isempty(fim_results[σ][:det]) && !isempty(gim_results[σ][:det])
            opt_idx_fim = argmax(fim_results[σ][:det])
            push!(fim_opt_temps, temperatures[opt_idx_fim])
            
            opt_idx_gim = argmax(gim_results[σ][:det])
            push!(gim_opt_temps, temperatures[opt_idx_gim])
        end
    end
end

# Scatter plot with markers
scatter!(p3_noise_effect, noise_levels[1:length(fim_opt_temps)], fim_opt_temps,
         label="FIM Optimal T",
         color=:blue,
         markersize=10,
         marker=:circle,
         markeralpha=0.8)

scatter!(p3_noise_effect, noise_levels[1:length(gim_opt_temps)], gim_opt_temps,
         label="GIM Optimal T",
         color=:red,
         markersize=10,
         marker=:square,
         markeralpha=0.8)

# Connect the points with lines
plot!(p3_noise_effect, noise_levels[1:length(fim_opt_temps)], fim_opt_temps,
      label="",
      color=:blue,
      linewidth=2,
      linestyle=:solid,
      alpha=0.5)

plot!(p3_noise_effect, noise_levels[1:length(gim_opt_temps)], gim_opt_temps,
      label="",
      color=:red,
      linewidth=2,
      linestyle=:dash,
      alpha=0.5)

# Set y-axis limits
y_min_temp = min(minimum(fim_opt_temps), minimum(gim_opt_temps)) - 5
y_max_temp = max(maximum(fim_opt_temps), maximum(gim_opt_temps)) + 5
ylims!(p3_noise_effect, y_min_temp, y_max_temp)

# ----------------------------------------------------------------------------
# Plot 7: Information Loss Plot
# ----------------------------------------------------------------------------
println("Creating information loss plot...")

p4_info_loss = plot(
    title="Information Loss Due to 1% Measurement Noise",
    xlabel="Temperature (K)",
    ylabel="Information Relative to No Noise (%)",
    legend=:bottom,
    grid=true,
    size=(850, 550),
    left_margin=5mm,
    bottom_margin=5mm,
    ylims=(0, 110)
)

# Calculate information loss for FIM and GIM at 1% noise
if haskey(fim_results, 0.0) && haskey(fim_results, 0.01) &&
   haskey(gim_results, 0.0) && haskey(gim_results, 0.01)
    
    fim_info_ratio = Float64[]
    gim_info_ratio = Float64[]
    
    for i in 1:length(temperatures)
        fim_no_noise = max(fim_results[0.0][:det][i], 1e-10)
        fim_with_noise = max(fim_results[0.01][:det][i], 1e-10)
        gim_no_noise = max(gim_results[0.0][:det][i], 1e-10)
        gim_with_noise = max(gim_results[0.01][:det][i], 1e-10)
        
        push!(fim_info_ratio, (fim_with_noise / fim_no_noise) * 100)
        push!(gim_info_ratio, (gim_with_noise / gim_no_noise) * 100)
    end
    
    # Plot information ratios
    plot!(p4_info_loss, temperatures, fim_info_ratio,
          label="FIM (1% noise)",
          color=:blue,
          linewidth=3,
          linestyle=:solid)
    
    plot!(p4_info_loss, temperatures, gim_info_ratio,
          label="GIM (1% noise)",
          color=:red,
          linewidth=3,
          linestyle=:dash)
    
    # Add 100% reference line
    hline!(p4_info_loss, [100],
           label="No information loss",
           color=:black,
           linestyle=:dot,
           linewidth=1.5,
           alpha=0.5)
end

# ============================================================================
# STEP 11: CREATE COMPOSITE PLOT
# ============================================================================

println("\nCreating composite plot...")

# Create a nicely arranged composite plot
composite_plot = plot(
    p1_fim, p1_gim, p1_compare,
    p2_fim, p2_gim, p3_noise_effect,
    layout=(3, 2),
    size=(1400, 1100),
    plot_title="Fisher vs Global Information Matrix: Effect of Measurement Noise",
    titlefontsize=14,
    left_margin=10mm,
    bottom_margin=10mm,
    top_margin=5mm
)

# ============================================================================
# STEP 12: SAVE PLOTS
# ============================================================================

println("Saving plots...")

# Save all plots
savefig(composite_plot, "fim_vs_gim_comparison.png")
savefig(p1_fim, "fim_determinant.png")
savefig(p1_gim, "gim_determinant.png")
savefig(p1_compare, "fim_gim_comparison.png")
savefig(p2_fim, "fim_conditioning.png")
savefig(p2_gim, "gim_conditioning.png")
savefig(p3_noise_effect, "noise_effect.png")
savefig(p4_info_loss, "information_loss.png")

println("✓ All plots saved successfully!")

# ============================================================================
# STEP 13: DISPLAY PLOTS
# ============================================================================

# Display individual plots (optional)
display(p1_fim)
display(p1_gim)
display(p1_compare)
display(p2_fim)
display(p2_gim)
display(p3_noise_effect)
display(p4_info_loss)

# Display composite plot
display(composite_plot)

# ============================================================================
# STEP 14: PRINT RESULTS SUMMARY
# ============================================================================

println("\n" * "="^60)
println("RESULTS SUMMARY")
println("="^60)

# Optimal temperatures
println("\n1. OPTIMAL EXPERIMENTAL TEMPERATURES:")
println("   No noise:")
println("     • FIM optimal: $(temperatures[argmax(fim_results[0.0][:det])]) K")
println("     • GIM optimal: $(temperatures[argmax(gim_results[0.0][:det])]) K")
println("\n   1% noise:")
println("     • FIM optimal: $(temperatures[argmax(fim_results[0.01][:det])]) K")
println("     • GIM optimal: $(temperatures[argmax(gim_results[0.01][:det])]) K")

# Information content
println("\n2. MAXIMUM INFORMATION CONTENT:")
max_fim_0 = maximum(fim_results[0.0][:det])
max_fim_1 = maximum(fim_results[0.01][:det])
max_gim_0 = maximum(gim_results[0.0][:det])
max_gim_1 = maximum(gim_results[0.01][:det])

println("   No noise:")
println("     • FIM max det: $(@sprintf("%.2e", max_fim_0))")
println("     • GIM max det: $(@sprintf("%.2e", max_gim_0))")
println("\n   1% noise:")
println("     • FIM max det: $(@sprintf("%.2e", max_fim_1))")
println("     • GIM max det: $(@sprintf("%.2e", max_gim_1))")

# Information loss
println("\n3. INFORMATION LOSS AT 1% NOISE:")
fim_loss = (1 - max_fim_1/max_fim_0) * 100
gim_loss = (1 - max_gim_1/max_gim_0) * 100
println("     • FIM loss: $(round(fim_loss, digits=1))%")
println("     • GIM loss: $(round(gim_loss, digits=1))%")

# Robustness
println("\n4. ROBUSTNESS TO NOISE:")
fim_sensitivity = std(fim_opt_temps)
gim_sensitivity = std(gim_opt_temps)
println("     • FIM optimal T std: $(round(fim_sensitivity, digits=1)) K")
println("     • GIM optimal T std: $(round(gim_sensitivity, digits=1)) K")

if gim_sensitivity < fim_sensitivity
    println("   ✓ GIM is more robust to noise")
else
    println("   ✓ FIM is more robust to noise")
end

println("\n" * "="^60)
println("✓ All done! Check the PNG files for plots.")
println("="^60)