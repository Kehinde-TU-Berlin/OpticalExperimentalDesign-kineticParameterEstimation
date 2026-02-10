#I'll rewrite the complete code with proper error handling and robust plotting. Here's the corrected version:


# ------------------------------------------------------------
# COMPLETE CODE FOR FIM AND GIM ANALYSIS WITH ERROR HANDLING
# ------------------------------------------------------------

# Load required packages
using Plots
using LinearAlgebra
using Distributions
using Random
using LaTeXStrings
using Statistics

# Set plotting backend and random seed
gr()
Random.seed!(123)

# ------------------------------------------------------------
# PART 1: REACTOR MODEL AND UTILITY FUNCTIONS
# ------------------------------------------------------------


#Simplified reactor model for a first-order reaction with axial dispersion.
#Returns outlet concentration.

function reactor_model(θ, x)
    A, E = θ  # Kinetic parameters
    T, C0, u, z = x  # Experimental conditions
    
    R = 8.314  # Gas constant (J/mol·K)
    L = 1.0    # Reactor length (m)
    
    # Arrhenius equation
    k = A * exp(-E/(R*T))
    
    # Damköhler number (reaction rate / flow rate)
    Da = k * L / u
    
    # Conversion for simplified model
    X = 1 - exp(-Da * z)
    
    return C0 * (1 - X)  # Outlet concentration
end


#Compute parameter sensitivities using forward finite differences.
#Returns vector of partial derivatives.

function compute_sensitivities(θ, x, model; δ=1e-4)
    n_params = length(θ)
    y0 = model(θ, x)  # Base output
    sensitivities = zeros(n_params)
    
    for i in 1:n_params
        # Perturb parameter i
        θ_pert = copy(θ)
        θ_pert[i] += δ * θ[i]
        
        # Compute output with perturbed parameter
        y_pert = model(θ_pert, x)
        
        # Finite difference approximation
        sensitivities[i] = (y_pert - y0) / (δ * θ[i])
    end
    
    return sensitivities
end


#Compute local Fisher Information Matrix (FIM) for a single experiment.

function compute_fim_local(θ, x, σ, model)
    S = compute_sensitivities(θ, x, model)
    return (1/σ^2) * (S * S')
end


#Compute Global Information Matrix (GIM) using Monte Carlo integration.
#Averages FIM over parameter prior distribution.

function compute_gim_mc(prior_mean, prior_cov, x, σ, model; n_samples=500)
    n_params = length(prior_mean)
    M_global = zeros(n_params, n_params)
    
    # Create multivariate normal prior
    prior_dist = MvNormal(prior_mean, prior_cov)
    
    # Monte Carlo integration
    for i in 1:n_samples
        θ_sample = rand(prior_dist)
        FIM_local = compute_fim_local(θ_sample, x, σ, model)
        M_global += FIM_local / n_samples
    end
    
    return M_global
end


#Safe function to compute condition number with regularization.
#Prevents division by zero.

function safe_condition_number(FIM; ϵ=1e-10)
    λ = eigvals(FIM)
    λ_max = maximum(λ)
    λ_min = max(minimum(λ), ϵ)  # Regularize to avoid zero
    return λ_max / λ_min
end


#Helper function to plot uncertainty ellipses

function plot_ellipse!(plt, center, Σ; n_points=100, kwargs...)
    # Eigen decomposition of covariance matrix
    λ, V = eigen(Σ)
    
    # Create ellipse points (2σ confidence region)
    t = range(0, 2π, length=n_points)
    ellipse_points = [center + 2 * sqrt(λ[1]) * cos(θ) * V[:,1] + 
                            2 * sqrt(λ[2]) * sin(θ) * V[:,2] 
                     for θ in t]
    
    x_vals = [p[1] for p in ellipse_points]
    y_vals = [p[2] for p in ellipse_points]
    
    plot!(plt, x_vals, y_vals; kwargs...)
end

# ------------------------------------------------------------
# PART 2: PARAMETER SETUP AND DATA GENERATION
# ------------------------------------------------------------

# True parameters (for local FIM computation)
θ_true = [1e5, 50000.0]  # A = 10⁵ s⁻¹, E = 50 kJ/mol

# Prior distribution (for global GIM computation)
θ_prior_mean = [1e5, 50000.0]
θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])  # ±20% A, ±10% E

# Measurement noise
σ = 0.01  # 1% measurement error

# Experimental design space (temperature variation)
temperatures = 350:10:450  # 350K to 450K in 10K steps
experiments = [[T, 1.0, 0.1, 0.5] for T in temperatures]

# Initialize arrays for results
fim_dets = Float64[]
gim_dets = Float64[]
fim_traces = Float64[]
gim_traces = Float64[]
fim_cond = Float64[]
gim_cond = Float64[]

# ------------------------------------------------------------
# PART 3: COMPUTE FIM AND GIM FOR ALL EXPERIMENTS
# ------------------------------------------------------------

println("Computing FIM and GIM for $(length(experiments)) experiments...")

for (i, x) in enumerate(experiments)
    T = x[1]  # Current temperature
    
    # 1. Compute local FIM at true parameters
    FIM = compute_fim_local(θ_true, x, σ, reactor_model)
    
    # 2. Compute global GIM using Monte Carlo
    GIM = compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=500)
    
    # 3. Store metrics with error handling
    push!(fim_dets, det(FIM))
    push!(gim_dets, det(GIM))
    
    push!(fim_traces, tr(FIM))
    push!(gim_traces, tr(GIM))
    
    push!(fim_cond, safe_condition_number(FIM))
    push!(gim_cond, safe_condition_number(GIM))
    
    # Progress indicator
    if i % 5 == 0
        println("  Completed $i/$(length(experiments)) experiments")
    end
end

println("Computation complete!")

# ------------------------------------------------------------
# PART 4: CREATE PLOTS WITH ROBUST ERROR HANDLING
# ------------------------------------------------------------

println("\nCreating plots...")

# Function to safely get maximum value, handling Inf and NaN
function safe_maximum(data)
    filtered = filter(isfinite, data)
    return isempty(filtered) ? 1.0 : maximum(filtered)
end

# Function to safely get minimum value, handling Inf and NaN
function safe_minimum(data)
    filtered = filter(isfinite, data)
    return isempty(filtered) ? 0.0 : minimum(filtered)
end

# ------------------------------------------------------------
# Plot 1: D-optimality (Determinant)
# ------------------------------------------------------------

p1 = plot(temperatures, fim_dets, 
          label="Local FIM (D-opt)", 
          linewidth=2, 
          color=:blue, 
          marker=:circle, 
          markersize=4,
          ylabel="Determinant",
          xlabel="Temperature (K)",
          title="(A) D-Optimality Criterion",
          legend=:topleft)

plot!(p1, temperatures, gim_dets, 
      label="Global GIM (D-opt)", 
      linewidth=2, 
      color=:red, 
      marker=:square, 
      markersize=4, 
      linestyle=:dash)

# Add annotation

max_det_temp = temperatures[argmax(fim_dets)]
annotate!(p1, max_det_temp, maximum(fim_dets)*0.8, 
          text("Peak at $max_det_temp K", 8, :center, :blue))

# ------------------------------------------------------------
# Plot 2: A-optimality (Trace)
# ------------------------------------------------------------

p2 = plot(temperatures, fim_traces, 
          label="Local FIM (A-opt)", 
          linewidth=2, 
          color=:blue, 
          marker=:circle, 
          markersize=4,
          ylabel="Trace",
          xlabel="Temperature (K)",
          title="(B) A-Optimality Criterion",
          legend=:topleft)

plot!(p2, temperatures, gim_traces, 
      label="Global GIM (A-opt)", 
      linewidth=2, 
      color=:red, 
      marker=:square, 
      markersize=4, 
      linestyle=:dash)

# ------------------------------------------------------------
# Plot 3: Matrix Conditioning (Condition Number)
# ------------------------------------------------------------

# Cap condition numbers for visualization

max_cond = 1e6
fim_cond_plot = [min(c, max_cond) for c in fim_cond]
gim_cond_plot = [min(c, max_cond) for c in gim_cond]

# Get safe y-axis limits
y_max_cond = safe_maximum([fim_cond_plot; gim_cond_plot]) * 1.1
y_min_cond = max(0, safe_minimum([fim_cond_plot; gim_cond_plot]) * 0.9)

p3 = plot(temperatures, fim_cond_plot, 
          label="Local FIM", 
          linewidth=2, 
          color=:blue, 
          marker=:circle, 
          markersize=4,
          ylabel="Condition Number (λ_max/λ_min)",
          xlabel="Temperature (K)",
          title="(C) Matrix Conditioning",
          legend=:topleft,
          ylims=(y_min_cond, y_max_cond))

plot!(p3, temperatures, gim_cond_plot, 
      label="Global GIM", 
      linewidth=2, 
      color=:red, 
      marker=:square, 
      markersize=4, 
      linestyle=:dash)

# Add warning for near-singular matrices

singular_threshold = 1e5
singular_idx = findall(c -> c >= singular_threshold, fim_cond)
for idx in singular_idx
    T_singular = temperatures[idx]
    annotate!(p3, T_singular, y_max_cond*0.9, 
              text("Near-singular", 6, :center, :red, :bottom))
end

annotate!(p3, temperatures[end-5], y_max_cond*0.7, 
          text("Lower = Better", 8, :right))

# ------------------------------------------------------------
# Plot 4: Information Gain Ratio
# ------------------------------------------------------------

# Compute ratio safely (avoid division by zero)
info_ratio = zeros(length(gim_dets))
for i in 1:length(gim_dets)
    if fim_dets[i] > 0
        info_ratio[i] = gim_dets[i] / fim_dets[i]
    else
        info_ratio[i] = 1.0  # Default value when FIM is singular
    end
end

p4 = plot(temperatures, info_ratio, 
          label="GIM/FIM Ratio", 
          linewidth=3, 
          color=:purple, 
          marker=:diamond, 
          markersize=5,
          ylabel="Information Ratio (Global/Local)",
          xlabel="Temperature (K)",
          title="(D) Global vs Local Information",
          legend=:topleft)

# Add reference line at 1.0

hline!(p4, [1.0], 
       label="Equal Information", 
       color=:black, 
       linestyle=:dash, 
       linewidth=1, 
       alpha=0.5)

# Annotate regions where global > local

above_one_idx = findall(r -> r > 1.1, info_ratio)
if !isempty(above_one_idx)
    T_above = temperatures[above_one_idx[1]]
    annotate!(p4, T_above, 1.2, 
              text("Global > Local", 8, :left, :purple))
end

# ------------------------------------------------------------
# COMBINE MAIN PLOTS
# ------------------------------------------------------------

layout = @layout [a b; c d]
combined_plot = plot(p1, p2, p3, p4, 
                     layout=layout, 
                     size=(1200, 800),
                     plot_title="Fisher Information Matrix (FIM) vs Global Information Matrix (GIM) for Reactor Parameter Estimation")

# ------------------------------------------------------------
# PART 5: 3D VISUALIZATION OF DESIGN SPACE
# ------------------------------------------------------------

println("Creating 3D visualization...")

# Create 2D design space (Temperature × Flow velocity)
T_range = 350:20:450
u_range = 0.05:0.05:0.2
T_grid = repeat(T_range, outer=length(u_range))
u_grid = repeat(u_range, inner=length(T_range))

fim_3d = Float64[]
gim_3d = Float64[]

for (T, u) in zip(T_grid, u_grid)
    x = [T, 1.0, u, 0.5]
    
    # Compute FIM and GIM
    FIM = compute_fim_local(θ_true, x, σ, reactor_model)
    GIM = compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=300)
    
    push!(fim_3d, det(FIM))
    push!(gim_3d, det(GIM))
end

# Create 3D surface plots
p5 = plot(T_grid, u_grid, fim_3d, 
          st=:surface, 
          xlabel="Temperature (K)", 
          ylabel="Flow Velocity (m/s)", 
          zlabel="FIM Determinant", 
          title="Local FIM in Design Space",
          color=:viridis, 
          camera=(30, 45),
          colorbar_title="Determinant")

p6 = plot(T_grid, u_grid, gim_3d, 
          st=:surface, 
          xlabel="Temperature (K)", 
          ylabel="Flow Velocity (m/s)", 
          zlabel="GIM Determinant", 
          title="Global GIM in Design Space",
          color=:plasma, 
          camera=(30, 45),
          colorbar_title="Determinant")

# Combine 3D plots
surface_plot = plot(p5, p6, 
                    layout=(1,2), 
                    size=(1000, 400),
                    plot_title="3D Visualization of Information in Design Space")

# ------------------------------------------------------------
# PART 6: UNCERTAINTY ELLIPSES FOR KEY EXPERIMENTS
# ------------------------------------------------------------

println("Creating uncertainty ellipses...")

# Select key temperatures for comparison
key_temps = [380, 410, 440]
key_experiments = [[T, 1.0, 0.1, 0.5] for T in key_temps]

# Create ellipse plot
p7 = plot(xlabel=L"A \ (s^{-1})", 
          ylabel=L"E \ (J/mol)", 
          title="Parameter Uncertainty Ellipses (2σ confidence)",
          legend=:topright, 
          aspect_ratio=:equal,
          xlims=(0.8e5, 1.2e5),
          ylims=(45000, 55000))

colors = [:blue, :red, :green]
labels_fim = ["380K (FIM)", "410K (FIM)", "440K (FIM)"]
labels_gim = ["380K (GIM)", "410K (GIM)", "440K (GIM)"]

for (i, (x, T)) in enumerate(zip(key_experiments, key_temps))
    # Compute information matrices
    FIM = compute_fim_local(θ_true, x, σ, reactor_model)
    GIM = compute_gim_mc(θ_prior_mean, θ_prior_cov, x, σ, reactor_model, n_samples=500)
    
    # Plot FIM ellipse (if invertible)
    if det(FIM) > 1e-10
        Σ_fim = inv(FIM)
        plot_ellipse!(p7, θ_true, Σ_fim, 
                     color=colors[i], 
                     label=labels_fim[i], 
                     alpha=0.3, 
                     linewidth=2)
    end
    
    # Plot GIM ellipse (if invertible)
    if det(GIM) > 1e-10
        Σ_gim = inv(GIM)
        plot_ellipse!(p7, θ_prior_mean, Σ_gim, 
                     color=colors[i], 
                     label=labels_gim[i], 
                     alpha=0.7, 
                     linewidth=2, 
                     linestyle=:dash)
    end
    
    # Mark parameter points
    scatter!(p7, [θ_true[1]], [θ_true[2]], 
            color=colors[i], 
            markersize=6, 
            label="", 
            marker=:star)
end

# Add prior uncertainty ellipse
plot_ellipse!(p7, θ_prior_mean, θ_prior_cov, 
             color=:black, 
             label="Prior Uncertainty", 
             alpha=0.2, 
             linewidth=1, 
             linestyle=:dot)

# ------------------------------------------------------------
# PART 7: RESULTS SUMMARY AND PLOT DISPLAY
# ------------------------------------------------------------

println("\n" * "="^70)
println("ANALYSIS RESULTS SUMMARY")
println("="^70)

# Find optimal temperatures

opt_local_idx = argmax(fim_dets)
opt_global_idx = argmax(gim_dets)
T_opt_local = temperatures[opt_local_idx]
T_opt_global = temperatures[opt_global_idx]

println("\n1. OPTIMAL EXPERIMENTAL CONDITIONS:")
println("   Local FIM (D-optimal):  T = $(T_opt_local) K")
println("   Global GIM (D-optimal): T = $(T_opt_global) K")

println("\n2. INFORMATION CONTENT:")
println("   Max FIM determinant: $(round(maximum(fim_dets), sigdigits=4))")
println("   Max GIM determinant: $(round(maximum(gim_dets), sigdigits=4))")

println("\n3. MATRIX CONDITIONING:")
println("   Min FIM condition number: $(round(minimum(fim_cond), sigdigits=4))")
println("   Min GIM condition number: $(round(minimum(gim_cond), sigdigits=4))")

println("\n4. PRACTICAL RECOMMENDATIONS:")
println("   • If confident in parameters: Use T = $(T_opt_local) K (FIM optimum)")
println("   • If uncertain about parameters: Use T = $(T_opt_global) K (GIM optimum)")
println("   • For robust design: Use global approach (GIM)")
println("   • Avoid temperatures where condition number > 1e4")

# Display all plots
display(combined_plot)
display(surface_plot)
display(p7)

# Save plots
savefig(combined_plot, "fim_gim_comparison_robust.png")
savefig(surface_plot, "design_space_3d.png")
savefig(p7, "uncertainty_ellipses.png")

println("\n" * "="^70)
println("Plots saved successfully:")
println("   1. fim_gim_comparison_robust.png")
println("   2. design_space_3d.png")
println("   3. uncertainty_ellipses.png")
println("="^70)

# ------------------------------------------------------------
# PART 8: ADDITIONAL DIAGNOSTIC INFORMATION
# ------------------------------------------------------------

println("\nDIAGNOSTIC INFORMATION:")
println("-" * 40)

# Check for singular or near-singular matrices
println("\nChecking for ill-conditioned matrices:")
singular_count_fim = count(c -> c > 1e4, fim_cond)
singular_count_gim = count(c -> c > 1e4, gim_cond)

println("   FIM matrices with condition number > 1e4: $singular_count_fim")
println("   GIM matrices with condition number > 1e4: $singular_count_gim")

if singular_count_fim > 0
    println("   Warning: Some FIM matrices are ill-conditioned!")
    println("   This means parameters may not be independently identifiable")
    println("   at certain experimental conditions.")
end

# Parameter sensitivity analysis at optimal points

println("\nParameter sensitivities at optimal temperatures:")
println("   At T = $(T_opt_local) K (FIM optimum):")
x_opt_local = [T_opt_local, 1.0, 0.1, 0.5]
S_local = compute_sensitivities(θ_true, x_opt_local, reactor_model)
println("      ∂C/∂A = $(round(S_local[1], sigdigits=4))")
println("      ∂C/∂E = $(round(S_local[2], sigdigits=4))")

println("\n   At T = $(T_opt_global) K (GIM optimum):")
x_opt_global = [T_opt_global, 1.0, 0.1, 0.5]
S_global = compute_sensitivities(θ_true, x_opt_global, reactor_model)
println("      ∂C/∂A = $(round(S_global[1], sigdigits=4))")
println("      ∂C/∂E = $(round(S_global[2], sigdigits=4))")

# Final recommendation

println("\n" * "="^70)
println("FINAL RECOMMENDATION FOR REACTOR EXPERIMENTS:")
println("="^70)
println("\nBased on the analysis:")
println("1. For precise parameter estimation with known initial guesses,")
println("   conduct experiments at $(T_opt_local) K (local FIM optimum).")
println("")
println("2. When parameter uncertainty is significant (>20% for A, >10% for E),")
println("   conduct experiments at $(T_opt_global) K (global GIM optimum).")
println("")
println("3. The global approach (GIM) provides more robust experimental")
println("   designs, especially when operating far from the optimal region.")
println("")
println("4. Avoid experimental conditions where the condition number")
println("   exceeds 10,000, as these lead to poor parameter identifiability.")
println("="^70)


## **KEY IMPROVEMENTS AND ERROR FIXES:**

### **1. Fixed the `ylims!` Error:**

# OLD (Problematic):
ylims!(p3, 0, maximum(condition_numbers_fim)*1.1)

# NEW (Fixed):
y_max_cond = safe_maximum([fim_cond_plot; gim_cond_plot]) * 1.1
y_min_cond = max(0, safe_minimum([fim_cond_plot; gim_cond_plot]) * 0.9)
p3 = plot(..., ylims=(y_min_cond, y_max_cond))


#**Why it failed before:** When `condition_numbers_fim` contained `Inf` or extremely large values, `maximum()` would return `Inf`, and `ylims!(p3, 0, Inf)` is invalid.

#**How it's fixed now:**
#- Added `safe_maximum()` and `safe_minimum()` functions that filter out non-finite values
#- Capped condition numbers at `1e6` for visualization
#- Set y-limits directly in the `plot()` call rather than with `ylims!()`

### **2. Robust Error Handling Functions:**

function safe_maximum(data)
    filtered = filter(isfinite, data)
    return isempty(filtered) ? 1.0 : maximum(filtered)
end

function safe_minimum(data)
    filtered = filter(isfinite, data)
    return isempty(filtered) ? 0.0 : minimum(filtered)
end


#These functions ensure we always get valid numbers for plotting limits.

### **3. Safe Condition Number Computation:**

function safe_condition_number(FIM; ϵ=1e-10)
    λ = eigvals(FIM)
    λ_max = maximum(λ)
    λ_min = max(minimum(λ), ϵ)  # Regularize to avoid zero
    return λ_max / λ_min
end


#Prevents division by zero by ensuring `λ_min` is at least `ϵ`.

### **4. Safe Information Ratio Computation:**

info_ratio = zeros(length(gim_dets))
for i in 1:length(gim_dets)
    if fim_dets[i] > 0
        info_ratio[i] = gim_dets[i] / fim_dets[i]
    else
        info_ratio[i] = 1.0  # Default when FIM is singular
    end
end


#Avoids division by zero when FIM determinant is zero or negative.

### **5. Progress Indicators and Diagnostics:**
#Added print statements to show computation progress and diagnostic information about ill-conditioned matrices.

### **6. Comprehensive Error Checking:**
#- Checks for singular matrices before inversion
#- Validates data before plotting
#- Provides clear warnings when problems are detected

### **7. Improved Visualization:**
#- Clear annotations explaining each plot
#- Consistent color scheme and styling
#- Meaningful axis labels with units
#- Professional-looking layouts

## **HOW TO RUN THE CODE:**

#1. **Save the code** as `fim_gim_analysis.jl`
#2. **Run in Julia**:

#include("fim_gim_analysis.jl")

#3. **The code will:**
#   - Compute FIM and GIM for different temperatures
#   - Create 3 comprehensive plots
#   - Save them as PNG files
#   - Print detailed analysis and recommendations

## **EXPECTED OUTPUT:**

#1. **Main comparison plot** (4 subplots):
#   - D-optimality (determinant) vs temperature
#   - A-optimality (trace) vs temperature  
#   - Matrix conditioning vs temperature
#   - Information gain ratio (GIM/FIM)

#2. **3D surface plots** showing information in design space

#3. **Uncertainty ellipses** showing parameter estimation precision

#4. **Console output** with:
#   - Optimal experimental temperatures
#   - Information content metrics
#   - Practical recommendations
#   - Diagnostic warnings

#The code is now robust, handles edge cases properly, and provides clear, actionable insights for optimal experimental design in non-ideal reactors.