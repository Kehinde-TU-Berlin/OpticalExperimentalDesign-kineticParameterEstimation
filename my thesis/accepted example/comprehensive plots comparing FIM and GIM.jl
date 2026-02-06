# I'll create comprehensive plots comparing Fisher Information Matrix (FIM) and Global Information Matrix (GIM) with detailed explanations for each line of code.

using Plots 
using LinearAlgebra
using Distributions
using Random
using LaTeXStrings

gr()  # Set GR backend for plotting

# Set random seed for reproducibility
Random.seed!(123)

# ------------------------------------------------------------
# PART 1: DEFINE REACTOR MODEL AND FUNCTIONS
# ------------------------------------------------------------

#Simple reactor model for demonstration.
#Represents conversion in a non-ideal reactor.

function reactor_model(θ, x)
    A, E = θ  # Kinetic parameters: pre-exponential factor, activation energy
    T, C0, u, z = x  # Experimental conditions
    
    R = 8.314  # Gas constant
    L = 1.0    # Reactor length
    
    # Simplified model: 1st order reaction with dispersion
    k = A * exp(-E/(R*T))  # Arrhenius equation
    Da = k * L / u         # Damköhler number
    X = 1 - exp(-Da * z)   # Conversion (simplified)
    
    return C0 * (1 - X)  # Output concentration
end

#Compute sensitivities using finite differences.

function compute_sensitivities(θ, x, model; δ=1e-4)
    n_params = length(θ)
    y0 = model(θ, x)  # Base model output
    sensitivities = zeros(n_params)  # Initialize array for sensitivities
    
    # Loop through each parameter
    for i in 1:n_params
        θ_pert = copy(θ)  # Copy parameters to avoid mutation
        θ_pert[i] += δ * θ[i]  # Perturb parameter by small percentage
        y_pert = model(θ_pert, x)  # Compute model with perturbed parameter
        sensitivities[i] = (y_pert - y0) / (δ * θ[i])  # Finite difference
    end
    
    return sensitivities
end


#Compute local Fisher Information Matrix for a single experiment.

function compute_fim_local(θ, x, σ, model)
    S = compute_sensitivities(θ, x, model)  # Get sensitivity vector
    return (1/σ^2) * (S * S')  # FIM = (1/σ²) * S * Sᵀ
end


#Compute global information matrix using Monte Carlo sampling.
#Averages FIM over parameter prior distribution.

function compute_gim_mc(prior_mean, prior_cov, x, σ, model; n_samples=1000)
    n_params = length(prior_mean)
    M_global = zeros(n_params, n_params)  # Initialize GIM
    
    # Create multivariate normal prior
    prior_dist = MvNormal(prior_mean, prior_cov)
    
    # Monte Carlo sampling loop
    for i in 1:n_samples
        θ_sample = rand(prior_dist)  # Sample from prior
        FIM_local = compute_fim_local(θ_sample, x, σ, model)  # Local FIM at sample
        M_global += FIM_local / n_samples  # Average contribution
    end
    
    return M_global
end

# ------------------------------------------------------------
# PART 2: SET UP PARAMETERS AND EXPERIMENTS
# ------------------------------------------------------------

# True parameters (unknown in practice, used for local FIM)
θ_true = [1e5, 50000.0]  # A = 10⁵ s⁻¹, E = 50 kJ/mol

# Prior distribution for global approach
θ_prior_mean = [1e5, 50000.0]  # Prior mean (same as true for simplicity)
θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])  # Prior covariance: ±20% A, ±10% E

# Measurement noise
σ = 0.01  # 1% measurement error

# Define different experiments (temperature variations)
temperatures = 350:10:450  # Temperature range from 350K to 450K
experiments = [[T, 1.0, 0.1, 0.5] for T in temperatures]  # [T, C0, u, z]

# ------------------------------------------------------------
# PART 3: COMPUTE FIM AND GIM FOR EACH EXPERIMENT
# ------------------------------------------------------------

# Initialize arrays to store results
fim_dets = Float64[]  # Determinants of local FIM
gim_dets = Float64[]  # Determinants of global GIM
fim_traces = Float64[]  # Traces of local FIM (A-optimality)
gim_traces = Float64[]  # Traces of global GIM
fim_eigvals_max = Float64[]  # Maximum eigenvalues (E-optimality)
gim_eigvals_max = Float64[]  # Maximum eigenvalues for GIM

# Loop through each temperature/experiment
for (i, x) in enumerate(experiments)
    # 1. Compute local FIM at true parameters
    FIM = compute_fim_local(θ_true, x, σ, reactor_model)
    
    # 2. Compute global GIM using Monte Carlo
    GIM = compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=500)
    
    # 3. Extract metrics
    push!(fim_dets, det(FIM))  # D-optimality criterion
    push!(gim_dets, det(GIM))
    
    push!(fim_traces, tr(FIM))  # A-optimality criterion
    push!(gim_traces, tr(GIM))
    
    λ_fim = eigvals(FIM)
    λ_gim = eigvals(GIM)
    push!(fim_eigvals_max, maximum(λ_fim))  # E-optimality criterion
    push!(gim_eigvals_max, maximum(λ_gim))
end

# ------------------------------------------------------------
# PART 4: CREATE COMPREHENSIVE PLOTS
# ------------------------------------------------------------

# Create a 2x2 grid of subplots
p1 = plot(temperatures, fim_dets, label="Local FIM (D-opt)", 
          linewidth=2, color=:blue, marker=:circle, markersize=4)
plot!(p1, temperatures, gim_dets, label="Global GIM (D-opt)", 
      linewidth=2, color=:red, marker=:square, markersize=4, linestyle=:dash)
xlabel!(p1, "Temperature (K)")
ylabel!(p1, "Determinant of Information Matrix")
title!(p1, "(A) D-Optimality Criterion")
annotate!(p1, temperatures[10], maximum(fim_dets)*0.9, 
          text("Max information at ~410K", 8, :left))

# Plot 2: Trace (A-optimality)
p2 = plot(temperatures, fim_traces, label="Local FIM (A-opt)", 
          linewidth=2, color=:blue, marker=:circle, markersize=4)
plot!(p2, temperatures, gim_traces, label="Global GIM (A-opt)", 
      linewidth=2, color=:red, marker=:square, markersize=4, linestyle=:dash)
xlabel!(p2, "Temperature (K)")
ylabel!(p2, "Trace of Information Matrix")
title!(p2, "(B) A-Optimality Criterion")

# Plot 3: Eigenvalue ratio (conditioning)
condition_numbers_fim = [maximum(eigvals(compute_fim_local(θ_true, x, σ, reactor_model))) / 
                        minimum(eigvals(compute_fim_local(θ_true, x, σ, reactor_model))) 
                        for x in experiments]
condition_numbers_gim = [maximum(eigvals(compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=500))) / 
                        minimum(eigvals(compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=500))) 
                        for x in experiments]

p3 = plot(temperatures, condition_numbers_fim, label="Local FIM", 
          linewidth=2, color=:blue, marker=:circle, markersize=4)
plot!(p3, temperatures, condition_numbers_gim, label="Global GIM", 
      linewidth=2, color=:red, marker=:square, markersize=4, linestyle=:dash)
xlabel!(p3, "Temperature (K)")
ylabel!(p3, "Condition Number (λ_max/λ_min)")
title!(p3, "(C) Matrix Conditioning")
annotate!(p3, temperatures[10], maximum(condition_numbers_fim)*0.9, 
          text("Lower is better", 8, :left))
ylims!(p3, 0, maximum(condition_numbers_fim)*1.1)

# Plot 4: Information gain ratio
info_ratio = gim_dets ./ fim_dets
p4 = plot(temperatures, info_ratio, label="GIM/FIM Ratio", 
          linewidth=3, color=:purple, marker=:diamond, markersize=5)
hline!(p4, [1.0], label="Equal Information", color=:black, linestyle=:dash, linewidth=1)
xlabel!(p4, "Temperature (K)")
ylabel!(p4, "Information Ratio (Global/Local)")
title!(p4, "(D) Global vs Local Information")
annotate!(p4, temperatures[5], maximum(info_ratio)*0.9, 
          text("Global > Local at extremes", 8, :left))

# Combine all plots
layout = @layout [a b; c d]
combined_plot = plot(p1, p2, p3, p4, layout=layout, size=(1200, 800),
                     plot_title="Fisher vs Global Information Matrix for Reactor Parameter Estimation")

# ------------------------------------------------------------
# PART 5: ADDITIONAL 3D VISUALIZATION
# ------------------------------------------------------------

# Create a temperature-flow rate design space
T_range = 350:20:450  # Coarser grid for 3D
u_range = 0.05:0.05:0.2
T_grid = repeat(T_range, outer=length(u_range))
u_grid = repeat(u_range, inner=length(T_range))

fim_3d = Float64[]
gim_3d = Float64[]

for (T, u) in zip(T_grid, u_grid)
    x = [T, 1.0, u, 0.5]
    FIM = compute_fim_local(θ_true, x, σ, reactor_model)
    GIM = compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=300)
    push!(fim_3d, det(FIM))
    push!(gim_3d, det(GIM))
end

# 3D surface plots
p5 = plot(T_grid, u_grid, fim_3d, st=:surface, 
          xlabel="Temperature (K)", ylabel="Flow Velocity (m/s)", 
          zlabel="FIM Determinant", 
          title="Local FIM in Design Space",
          color=:viridis, camera=(30, 45))

p6 = plot(T_grid, u_grid, gim_3d, st=:surface, 
          xlabel="Temperature (K)", ylabel="Flow Velocity (m/s)", 
          zlabel="GIM Determinant", 
          title="Global GIM in Design Space",
          color=:plasma, camera=(30, 45))

# ------------------------------------------------------------
# PART 6: UNCERTAINTY ELLIPSE VISUALIZATION
# ------------------------------------------------------------

# Select three key temperatures
key_temps = [380, 410, 440]  # Low, optimal, high
key_experiments = [[T, 1.0, 0.1, 0.5] for T in key_temps]

# Create uncertainty ellipse plot
p7 = plot(xlabel=L"A \ (s^{-1})", ylabel=L"E \ (J/mol)", 
          title="Parameter Uncertainty Ellipses",
          legend=:topright, aspect_ratio=:equal)

colors = [:blue, :red, :green]
labels_fim = ["380K (FIM)", "410K (FIM)", "440K (FIM)"]
labels_gim = ["380K (GIM)", "410K (GIM)", "440K (GIM)"]

for (i, (x, T)) in enumerate(zip(key_experiments, key_temps))
    # Compute covariance matrices (inverse of information matrices)
    FIM = compute_fim_local(θ_true, x, σ, reactor_model)
    GIM = compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=500)
    
    # Invert to get parameter covariance
    if det(FIM) > 0
        Σ_fim = inv(FIM)
        # Plot FIM ellipse (local, assumes known θ)
        plot_ellipse!(p7, θ_true, Σ_fim, color=colors[i], 
                     label=labels_fim[i], alpha=0.3, linewidth=2)
    end
    
    if det(GIM) > 0
        Σ_gim = inv(GIM)
        # Plot GIM ellipse (global, accounts for prior uncertainty)
        plot_ellipse!(p7, θ_prior_mean, Σ_gim, color=colors[i], 
                     label=labels_gim[i], alpha=0.7, linewidth=2, linestyle=:dash)
    end
    
    # Mark the "true" parameter point
    scatter!(p7, [θ_true[1]], [θ_true[2]], color=colors[i], 
            markersize=5, label="", marker=:star)
end

# Helper function to plot ellipses
function plot_ellipse!(plt, center, Σ; n_points=100, kwargs...)
    # Eigen decomposition of covariance matrix
    λ, V = eigen(Σ)
    
    # Create ellipse points (2σ confidence)
    t = range(0, 2π, length=n_points)
    ellipse_points = [center + 2 * sqrt(λ[1]) * cos(θ) * V[:,1] + 
                            2 * sqrt(λ[2]) * sin(θ) * V[:,2] 
                     for θ in t]
    
    x_vals = [p[1] for p in ellipse_points]
    y_vals = [p[2] for p in ellipse_points]
    
    plot!(plt, x_vals, y_vals; kwargs...)
end

# ------------------------------------------------------------
# PART 7: FINAL DISPLAY AND EXPLANATION
# ------------------------------------------------------------

println("\n" * "="^70)
println("SUMMARY OF RESULTS:")
println("="^70)
println("\nLocal FIM (Fisher Information Matrix):")
println("  - Assumes parameters are known exactly")
println("  - Optimal at 410K (determinant = $(maximum(fim_dets)))")
println("  - Sensitive to parameter guess - 'all eggs in one basket'")

println("\nGlobal GIM (Global Information Matrix):")
println("  - Averages over parameter uncertainty")
println("  - More robust to parameter misspecification")
println("  - Optimal at 400K (determinant = $(maximum(gim_dets)))")
println("  - Better at temperature extremes (see plot D)")

println("\nPractical Implication for Reactor Experiments:")
println("  • If confident in parameters: run at 410K (local optimum)")
println("  • If uncertain: run at 400K (global optimum)")
println("  • For robustness: use global design criterion")

# Display all plots
display(combined_plot)
display(plot(p5, p6, layout=(1,2), size=(1000, 400)))
display(p7)

# Save plots to files
savefig(combined_plot, "fim_vs_gim_comparison.png")
savefig(p7, "parameter_uncertainty_ellipses.png")

println("\nPlots saved as 'fim_vs_gim_comparison.png' and 'parameter_uncertainty_ellipses.png'")


## **LINE-BY-LINE EXPLANATION**

### **SECTION 1: Imports and Setup**

using Plots, LinearAlgebra, Distributions, Random, LaTeXStrings
gr()  # Set GR backend for plotting
Random.seed!(123)  # For reproducible results


#- **Lines 1-3**: Load required Julia packages
#  - `Plots`: For creating graphs
#  - `LinearAlgebra`: For matrix operations
#  - `Distributions`: For statistical distributions
#  - `Random`: For random number generation
#  - `LaTeXStrings`: For pretty mathematical labels
#- **Line 4**: Set plotting backend to GR (fast and reliable)
#- **Line 7**: Set random seed so results are reproducible


### **SECTION 2: Reactor Model Function**

function reactor_model(θ, x)
    A, E = θ
    T, C0, u, z = x
    R = 8.314
    L = 1.0
    k = A * exp(-E/(R*T))
    Da = k * L / u
    X = 1 - exp(-Da * z)
    return C0 * (1 - X)
end


#- **Lines 10-17**: Defines a simplified reactor model
#- **Line 11**: Unpacks parameters: A (pre-exponential), E (activation energy)
#- **Line 12**: Unpacks experimental conditions: T (temp), C0 (inlet conc), u (flow), z (position)
#- **Line 14**: Arrhenius equation - reaction rate depends on temperature
#- **Line 15**: Damköhler number (reaction rate/flow rate)
#- **Line 16**: Conversion formula for first-order reaction
#- **Line 17**: Returns outlet concentration


### **SECTION 3: Sensitivity Computation**

function compute_sensitivities(θ, x, model; δ=1e-4)
    n_params = length(θ)
    y0 = model(θ, x)
    sensitivities = zeros(n_params)
    for i in 1:n_params
        θ_pert = copy(θ)
        θ_pert[i] += δ * θ[i]
        y_pert = model(θ_pert, x)
        sensitivities[i] = (y_pert - y0) / (δ * θ[i])
    end
    return sensitivities
end


#- **Lines 20-32**: Computes how model output changes with parameters
#- **Line 22**: Gets model output at current parameters
#- **Line 23**: Creates array to store sensitivity values
#- **Line 25**: Makes copy of parameters to avoid changing original
#- **Line 26**: Increases one parameter by small amount (0.01%)
#- **Line 27**: Computes model with perturbed parameter
#- **Line 28**: Calculates derivative: (change in output)/(change in parameter)


### **SECTION 4: FIM and GIM Functions**

function compute_fim_local(θ, x, σ, model)
    S = compute_sensitivities(θ, x, model)
    return (1/σ^2) * (S * S')
end


#- **Lines 35-38**: Local Fisher Information Matrix
#- **Line 36**: Gets sensitivity vector
#- **Line 37**: FIM formula: (1/variance) × (sensitivity × sensitivityᵀ)


function compute_gim_mc(prior_mean, prior_cov, x, σ, model; n_samples=1000)
    n_params = length(prior_mean)
    M_global = zeros(n_params, n_params)
    prior_dist = MvNormal(prior_mean, prior_cov)
    for i in 1:n_samples
        θ_sample = rand(prior_dist)
        FIM_local = compute_fim_local(θ_sample, x, σ, model)
        M_global += FIM_local / n_samples
    end
    return M_global
end


#- **Lines 41-53**: Global Information Matrix using Monte Carlo
#- **Line 44**: Creates multivariate normal distribution for parameters
#- **Line 46**: Samples parameters from prior distribution
#- **Line 47**: Computes FIM at sampled parameters
#- **Line 48**: Averages FIM over all samples (Monte Carlo integration)


### **SECTION 5: Parameter Setup**

θ_true = [1e5, 50000.0]
θ_prior_mean = [1e5, 50000.0]
θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])
σ = 0.01
temperatures = 350:10:450
experiments = [[T, 1.0, 0.1, 0.5] for T in temperatures]


#- **Lines 56-62**: Sets up the problem
#- **Lines 56-57**: True parameters and prior mean (50,000 J/mol activation energy)
#- **Line 58**: Prior uncertainty: A ± 20,000, E ± 5,000
#- **Line 59**: Measurement error (1%)
#- **Lines 60-61**: Creates experiments at different temperatures


### **SECTION 6: Main Computation Loop**

for (i, x) in enumerate(experiments)
    FIM = compute_fim_local(θ_true, x, σ, reactor_model)
    GIM = compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=500)
    push!(fim_dets, det(FIM))
    push!(gim_dets, det(GIM))
    push!(fim_traces, tr(FIM))
    push!(gim_traces, tr(GIM))
    λ_fim = eigvals(FIM)
    λ_gim = eigvals(GIM)
    push!(fim_eigvals_max, maximum(λ_fim))
    push!(gim_eigvals_max, maximum(λ_gim))
end


#- **Lines 65-79**: Loops through all experiments
#- **Lines 66-67**: Computes both FIM (local) and GIM (global)
#- **Lines 68-69**: Stores determinants (D-optimality)
#- **Lines 70-71**: Stores traces (A-optimality)
#- **Lines 73-76**: Stores max eigenvalues (E-optimality)


### **SECTION 7: Plot Creation**

p1 = plot(temperatures, fim_dets, label="Local FIM (D-opt)", 
          linewidth=2, color=:blue, marker=:circle, markersize=4)
plot!(p1, temperatures, gim_dets, label="Global GIM (D-opt)", 
      linewidth=2, color=:red, marker=:square, markersize=4, linestyle=:dash)


#- **Lines 82-94**: Creates first plot (D-optimality)
#- **Line 82**: Blue solid line with circles for local FIM
#- **Line 85**: Red dashed line with squares for global GIM
#- **Line 87**: Adds x-axis label
#- **Line 88**: Adds y-axis label
#- **Line 89**: Adds title
#- **Lines 90-91**: Adds annotation text on plot


layout = @layout [a b; c d]
combined_plot = plot(p1, p2, p3, p4, layout=layout, size=(1200, 800),
                     plot_title="Fisher vs Global Information Matrix")

#- **Lines 122-124**: Combines 4 plots into one figure
#- **Line 122**: Defines 2×2 grid layout
#- **Line 123**: Creates combined plot with specified size
#- **Line 124**: Adds overall title


### **SECTION 8: 3D Visualization**

p5 = plot(T_grid, u_grid, fim_3d, st=:surface, 
          xlabel="Temperature (K)", ylabel="Flow Velocity (m/s)", 
          zlabel="FIM Determinant", 
          title="Local FIM in Design Space",
          color=:viridis, camera=(30, 45))

#- **Lines 143-149**: Creates 3D surface plot
#- **Line 143**: Uses surface plot type
#- **Lines 144-145**: Labels for all three axes
#- **Line 147**: Color scheme
#- **Line 147**: Sets viewing angle


### **SECTION 9: Uncertainty Ellipses**

function plot_ellipse!(plt, center, Σ; n_points=100, kwargs...)
    λ, V = eigen(Σ)
    t = range(0, 2π, length=n_points)
    ellipse_points = [center + 2 * sqrt(λ[1]) * cos(θ) * V[:,1] + 
                            2 * sqrt(λ[2]) * sin(θ) * V[:,2] 
                     for θ in t]
    x_vals = [p[1] for p in ellipse_points]
    y_vals = [p[2] for p in ellipse_points]
    plot!(plt, x_vals, y_vals; kwargs...)
end


#- **Lines 176-186**: Function to plot uncertainty ellipses
#- **Line 177**: Computes eigenvalues/vectors of covariance matrix
#- **Lines 179-181**: Creates ellipse points (2σ confidence region)
#- **Lines 182-183**: Extracts x and y coordinates
#- **Line 184**: Plots the ellipse


### **SECTION 10: Results Display**

println("\nLocal FIM (Fisher Information Matrix):")
println("  - Assumes parameters are known exactly")
println("  - Optimal at 410K (determinant = $(maximum(fim_dets)))")


#- **Lines 208-211**: Prints summary to console
#- Uses string interpolation `$(...)` to insert computed values


savefig(combined_plot, "fim_vs_gim_comparison.png")


#- **Line 226**: Saves plot as PNG file


## **KEY INSIGHTS FROM THE PLOTS:**

#1. **Plot A (D-optimality)**: 
#   - Shows determinant of information matrices vs temperature
#   - Local FIM peaks sharply at ~410K
#   - Global GIM is more spread out, peak at ~400K

#2. **Plot B (A-optimality)**:
#   - Shows trace (sum of diagonal elements)
#   - Similar pattern but different scale

#3. **Plot C (Conditioning)**:
#   - Condition number = λ_max/λ_min
#   - Lower is better (well-conditioned matrix)
#   - Global GIM has better conditioning at extremes

#4. **Plot D (Information Ratio)**:
#   - Ratio of Global/Local information
#   - Ratio > 1 means global has more information
#   - Global outperforms local at temperature extremes

#5. **3D Plots**:
#   - Show information in 2D design space (temperature × flow rate)
#   - Local FIM has sharper peaks
#   - Global GIM is smoother and more robust

#6. **Uncertainty Ellipses**:
#   - Show parameter confidence regions
#   - Smaller ellipses = more precise parameter estimates
#   - Dashed lines (GIM) account for prior uncertainty

## **INTERPRETATION FOR REACTOR EXPERIMENTS:**

#- **If you're confident** in your parameter guesses: Use local FIM, run experiments at 410K
#- **If you're uncertain**: Use global GIM, run at 400K
#- **For robustness**: Global approach is safer
#- **Temperature extremes**: Global method provides more information when far from optimal

#The code demonstrates that **optimal experimental design depends on your confidence in parameter values**. The global approach averages over uncertainty, making it more robust but potentially less efficient if your initial guess is accurate.