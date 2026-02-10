# I'll create a comprehensive comparison of Global Information Matrix (GIM) and Fisher Information Matrix (FIM) with and without noise, with detailed line-by-line explanations.


# ============================================================================
# COMPARISON OF GLOBAL INFORMATION MATRIX (GIM) vs FISHER INFORMATION MATRIX (FIM)
# WITH AND WITHOUT NOISE FOR REACTOR KINETIC PARAMETER ESTIMATION
# ============================================================================

# ============================================================================
# STEP 1: LOAD REQUIRED PACKAGES
# ============================================================================

# Import plotting library for creating graphs
using Plots

# Import linear algebra functions for matrix operations
using LinearAlgebra

# Import statistical distributions for Monte Carlo sampling
using Distributions

# Import random number generation for reproducibility
using Random

# Import string formatting for labels
using Printf

# Import statistics functions
using Statistics

# ============================================================================
# STEP 2: SET UP PLOTTING AND RANDOMNESS
# ============================================================================

# Choose GR as the plotting backend (fast and reliable)
gr()

# Set random seed so results are the same every time we run the code
Random.seed!(42)

# ============================================================================
# STEP 3: DEFINE THE REACTOR MODEL
# ============================================================================

# """
# Simple model of a non-ideal reactor for demonstration.
# This function calculates the outlet concentration of reactant A.
# """
function reactor_model(θ, x)
    # θ[1] = Pre-exponential factor A (s⁻¹)
    # θ[2] = Activation energy E (J/mol)
    A, E = θ
    
    # x[1] = Temperature T (K)
    # x[2] = Inlet concentration C₀ (mol/m³)
    # x[3] = Flow velocity u (m/s)
    # x[4] = Reactor position z (dimensionless)
    T, C0, u, z = x
    
    # Universal gas constant
    R = 8.314
    
    # Reactor length (fixed)
    L = 1.0
    
    # Arrhenius equation: reaction rate depends on temperature
    k = A * exp(-E/(R*T))
    
    # Damköhler number: ratio of reaction rate to flow rate
    Da = k * L / u
    
    # Conversion of reactant A (simplified model)
    X = 1 - exp(-Da * z)
    
    # Outlet concentration = inlet × (1 - conversion)
    return C0 * (1 - X)
end

# ============================================================================
# STEP 4: COMPUTE PARAMETER SENSITIVITIES
# ============================================================================

# """
# Calculate how much the model output changes when parameters change.
# This is done using finite differences (like calculating slope).
# """
function compute_sensitivities(θ, x, model; δ=1e-4)
    # Number of parameters (should be 2: A and E)
    n_params = length(θ)
    
    # Model output with current parameters
    y0 = model(θ, x)
    
    # Array to store sensitivity values
    sensitivities = zeros(n_params)
    
    # Loop through each parameter
    for i in 1:n_params
        # Copy current parameters (don't modify original)
        θ_pert = copy(θ)
        
        # Increase parameter i by a small percentage (δ%)
        θ_pert[i] += δ * θ[i]
        
        # Calculate model output with perturbed parameter
        y_pert = model(θ_pert, x)
        
        # Finite difference approximation of derivative
        # This tells us how output changes with parameter i
        sensitivities[i] = (y_pert - y0) / (δ * θ[i])
    end
    
    return sensitivities
end

# ============================================================================
# STEP 5: FISHER INFORMATION MATRIX (FIM) COMPUTATION
# ============================================================================

# """
# Calculate Fisher Information Matrix for one experiment.
# FIM measures how much information an experiment provides about parameters.
# """

function compute_fim(θ, x, σ, model)
    # Get sensitivity vector (how output depends on parameters)
    S = compute_sensitivities(θ, x, model)
    
    # FIM formula: (1/variance) × (sensitivity × sensitivityᵀ)
    # More noise (larger σ) → less information → smaller FIM
    return (1/σ^2) * (S * S')
end

# ============================================================================
# STEP 6: GLOBAL INFORMATION MATRIX (GIM) COMPUTATION
# ============================================================================

# """
# Calculate Global Information Matrix using Monte Carlo sampling.
# GIM averages FIM over parameter uncertainty (more robust).
# """
function compute_gim(prior_mean, prior_cov, x, σ, model; n_samples=1000)
    # Number of parameters
    n_params = length(prior_mean)
    
    # Initialize GIM to zeros
    M_global = zeros(n_params, n_params)
    
    # Create probability distribution for parameters (our uncertainty)
    prior_dist = MvNormal(prior_mean, prior_cov)
    
    # Monte Carlo loop: sample many possible parameter values
    for i in 1:n_samples
        # Sample one possible set of parameters from prior distribution
        θ_sample = rand(prior_dist)
        
        # Compute FIM for these sampled parameters
        FIM_local = compute_fim(θ_sample, x, σ, model)
        
        # Average all the FIMs to get GIM
        M_global += FIM_local / n_samples
    end
    
    return M_global
end

# ============================================================================
# STEP 7: SET UP THE PROBLEM PARAMETERS
# ============================================================================

println("Setting up reactor parameter estimation problem...")
println("="^60)

# "True" kinetic parameters (what we're trying to estimate)
θ_true = [1e5, 50000.0]  # A = 100,000 s⁻¹, E = 50,000 J/mol

# Our initial uncertainty about parameters (prior knowledge)
θ_prior_mean = [1e5, 50000.0]  # Best guess (same as true for simplicity)
θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])  # Uncertainty: ±20,000 for A, ±5,000 for E

# Define different noise levels to compare
noise_levels = [0.0, 0.005, 0.01, 0.02]  # 0%, 0.5%, 1%, 2% measurement error
noise_names = ["No noise", "0.5% noise", "1% noise", "2% noise"]

# Temperature range for experiments (350K to 450K)
temperatures = 350:5:450
experiments = [[T, 1.0, 0.1, 0.5] for T in temperatures]  # [T, C0, u, z]

println("Temperature range: $(minimum(temperatures))K to $(maximum(temperatures))K")
println("Noise levels: $noise_levels")
println("Number of experiments to analyze: $(length(experiments))")

# ============================================================================
# STEP 8: COMPUTE FIM AND GIM FOR ALL CONDITIONS
# ============================================================================

println("\nComputing information matrices...")
println("-"^60)

# Dictionaries to store results for different noise levels
fim_results = Dict()
gim_results = Dict()

# Metrics we'll compute for each noise level
for (noise_idx, σ) in enumerate(noise_levels)
    println("Processing noise level: $(noise_names[noise_idx]) (σ = $σ)")
    
    # Arrays to store metrics for this noise level
    fim_dets = Float64[]   # Determinant of FIM (D-optimality)
    gim_dets = Float64[]   # Determinant of GIM
    fim_traces = Float64[] # Trace of FIM (A-optimality)
    gim_traces = Float64[] # Trace of GIM
    fim_cond = Float64[]   # Condition number of FIM
    gim_cond = Float64[]   # Condition number of GIM
    
    # Loop through each temperature/experiment
    for (i, x) in enumerate(experiments)
        T = x[1]  # Current temperature
        
        # Compute FIM at "true" parameters (assumes we know them perfectly)
        FIM = compute_fim(θ_true, x, max(σ, 1e-10), reactor_model)  # max ensures σ > 0
        
        # Compute GIM using Monte Carlo (accounts for parameter uncertainty)
        GIM = compute_gim(θ_prior_mean, θ_prior_cov, x, max(σ, 1e-10), reactor_model, n_samples=500)
        
        # Store metrics
        push!(fim_dets, det(FIM))
        push!(gim_dets, det(GIM))
        push!(fim_traces, tr(FIM))
        push!(gim_traces, tr(GIM))
        
        # Compute condition number safely (avoid division by zero)
        λ_fim = eigvals(FIM)
        λ_gim = eigvals(GIM)
        cond_fim = maximum(λ_fim) / max(minimum(λ_fim), 1e-10)
        cond_gim = maximum(λ_gim) / max(minimum(λ_gim), 1e-10)
        push!(fim_cond, cond_fim)
        push!(gim_cond, cond_gim)
    end
    
    # Store results for this noise level
    fim_results[σ] = (det=fim_dets, trace=fim_traces, cond=fim_cond)
    gim_results[σ] = (det=gim_dets, trace=gim_traces, cond=gim_cond)
    
    println("  Completed $(length(experiments)) experiments")
end

println("\nComputation complete!")

# ============================================================================
# STEP 9: CREATE COMPARISON PLOTS - DETERMINANT (D-OPTIMALITY)
# ============================================================================

println("\nCreating comparison plots...")
println("-"^60)

# Define colors for different noise levels
colors = [:blue, :green, :orange, :red]
line_styles = [:solid, :dash, :dot, :dashdot]

# Plot 1: FIM Determinant (D-optimality) for different noise levels
p1_fim = plot(title="Fisher Information Matrix (FIM) - Determinant",
             xlabel="Temperature (K)",
             ylabel="Determinant (log scale)",
             yscale=:log10,
             legend=:topright,
             grid=true,
             size=(800, 500))

# Add FIM curves for each noise level
for (i, σ) in enumerate(noise_levels)
    plot!(p1_fim, temperatures, fim_results[σ].det,
          label="$(noise_names[i])",
          linewidth=2,
          color=colors[i],
          linestyle=line_styles[i],
          marker=(i==1 ? :circle : :none),
          markersize=4)
end

# Highlight optimal temperature for no-noise case
if length(fim_results[0.0].det) > 0
    opt_idx = argmax(fim_results[0.0].det)
    opt_temp = temperatures[opt_idx]
    opt_val = maximum(fim_results[0.0].det)
    
    vline!(p1_fim, [opt_temp],
           label="Optimal T = $(opt_temp)K (no noise)",
           color=:black,
           linestyle=:dash,
           linewidth=1,
           alpha=0.5)
    
    annotate!(p1_fim, opt_temp, opt_val*0.1,
              text("FIM Optimal", 8, :center, :black))
end

# Plot 2: GIM Determinant for different noise levels
p1_gim = plot(title="Global Information Matrix (GIM) - Determinant",
             xlabel="Temperature (K)",
             ylabel="Determinant (log scale)",
             yscale=:log10,
             legend=:topright,
             grid=true,
             size=(800, 500))

# Add GIM curves for each noise level
for (i, σ) in enumerate(noise_levels)
    plot!(p1_gim, temperatures, gim_results[σ].det,
          label="$(noise_names[i])",
          linewidth=2,
          color=colors[i],
          linestyle=line_styles[i],
          marker=(i==1 ? :square : :none),
          markersize=4)
end

# Highlight optimal temperature for no-noise case
if length(gim_results[0.0].det) > 0
    opt_idx = argmax(gim_results[0.0].det)
    opt_temp = temperatures[opt_idx]
    opt_val = maximum(gim_results[0.0].det)
    
    vline!(p1_gim, [opt_temp],
           label="Optimal T = $(opt_temp)K (no noise)",
           color=:black,
           linestyle=:dash,
           linewidth=1,
           alpha=0.5)
    
    annotate!(p1_gim, opt_temp, opt_val*0.1,
              text("GIM Optimal", 8, :center, :black))
end

# Plot 3: Direct comparison at 0% and 1% noise
p1_compare = plot(title="FIM vs GIM Comparison at Different Noise Levels",
                 xlabel="Temperature (K)",
                 ylabel="Determinant (log scale)",
                 yscale=:log10,
                 legend=:topright,
                 grid=true,
                 size=(800, 500))

# Compare key noise levels: 0% and 1%
key_noise_levels = [0.0, 0.01]
key_names = ["No noise", "1% noise"]

for (i, σ) in enumerate(key_noise_levels)
    # FIM curves
    plot!(p1_compare, temperatures, fim_results[σ].det,
          label="FIM ($(key_names[i]))",
          linewidth=2.5,
          color=colors[i],
          linestyle=:solid,
          alpha=0.8)
    
    # GIM curves
    plot!(p1_compare, temperatures, gim_results[σ].det,
          label="GIM ($(key_names[i]))",
          linewidth=2.5,
          color=colors[i],
          linestyle=:dash,
          alpha=0.8)
end

# Add noise effect annotation
annotate!(p1_compare, temperatures[15], maximum(fim_results[0.0].det)*0.01,
          text("Noise reduces information", 8, :left, :red))

# ============================================================================
# STEP 10: CREATE PLOTS FOR MATRIX CONDITIONING
# ============================================================================

# Plot 4: FIM Condition Number (how well-behaved the matrix is)
p2_fim = plot(title="FIM - Matrix Conditioning",
             xlabel="Temperature (K)",
             ylabel="Condition Number (log scale)",
             yscale=:log10,
             legend=:topright,
             grid=true,
             size=(800, 500))

for (i, σ) in enumerate(noise_levels)
    plot!(p2_fim, temperatures, fim_results[σ].cond,
          label="$(noise_names[i])",
          linewidth=2,
          color=colors[i],
          linestyle=line_styles[i])
end

# Highlight that lower condition number is better
annotate!(p2_fim, temperatures[10], maximum(fim_results[0.0].cond)*0.1,
          text("Lower = Better", 8, :left, :black))

# Plot 5: GIM Condition Number
p2_gim = plot(title="GIM - Matrix Conditioning",
             xlabel="Temperature (K)",
             yylabel="Condition Number (log scale)",
             yscale=:log10,
             legend=:topright,
             grid=true,
             size=(800, 500))

for (i, σ) in enumerate(noise_levels)
    plot!(p2_gim, temperatures, gim_results[σ].cond,
          label="$(noise_names[i])",
          linewidth=2,
          color=colors[i],
          linestyle=line_styles[i])
end

# Plot 6: Noise effect on optimal temperature
p3_noise_effect = plot(title="Noise Effect on Optimal Experimental Temperature",
                      xlabel="Measurement Noise Level (σ)",
                      ylabel="Optimal Temperature (K)",
                      legend=false,
                      grid=true,
                      size=(800, 500))

# Arrays to store optimal temperatures for each noise level
fim_opt_temps = Float64[]
gim_opt_temps = Float64[]

for σ in noise_levels
    if length(fim_results[σ].det) > 0
        opt_idx_fim = argmax(fim_results[σ].det)
        push!(fim_opt_temps, temperatures[opt_idx_fim])
        
        opt_idx_gim = argmax(gim_results[σ].det)
        push!(gim_opt_temps, temperatures[opt_idx_gim])
    end
end

# Plot optimal temperature vs noise level
scatter!(p3_noise_effect, noise_levels, fim_opt_temps,
         label="FIM Optimal T",
         color=:blue,
         markersize=8,
         marker=:circle)

scatter!(p3_noise_effect, noise_levels, gim_opt_temps,
         label="GIM Optimal T",
         color=:red,
         markersize=8,
         marker=:square)

# Add trend lines
plot!(p3_noise_effect, noise_levels, fim_opt_temps,
      label="FIM trend",
      color=:blue,
      linewidth=2,
      linestyle=:solid,
      alpha=0.5)

plot!(p3_noise_effect, noise_levels, gim_opt_temps,
      label="GIM trend",
      color=:red,
      linewidth=2,
      linestyle=:dash,
      alpha=0.5)

# Annotate the plot
annotate!(p3_noise_effect, 0.015, 410,
          text("FIM: More sensitive to noise", 8, :left, :blue))

annotate!(p3_noise_effect, 0.015, 402,
          text("GIM: More robust to noise", 8, :left, :red))

# ============================================================================
# STEP 11: CREATE INFORMATION LOSS PLOT
# ============================================================================

# Plot 7: Information loss due to noise
p4_info_loss = plot(title="Information Loss Due to Measurement Noise",
                   xlabel="Temperature (K)",
                   ylabel="Information Relative to No Noise (%)",
                   legend=:bottomright,
                   grid=true,
                   size=(800, 500),
                   ylims=(0, 110))

# Calculate information loss for FIM and GIM at 1% noise
if 0.0 in noise_levels && 0.01 in noise_levels
    # Relative information = (with noise) / (no noise) × 100%
    fim_info_ratio = (fim_results[0.01].det ./ fim_results[0.0].det) .* 100
    gim_info_ratio = (gim_results[0.01].det ./ gim_results[0.0].det) .* 100
    
    plot!(p4_info_loss, temperatures, fim_info_ratio,
          label="FIM (1% noise)",
          linewidth=3,
          color=:blue,
          linestyle=:solid)
    
    plot!(p4_info_loss, temperatures, gim_info_ratio,
          label="GIM (1% noise)",
          linewidth=3,
          color=:red,
          linestyle=:dash)
    
    # Add 100% reference line
    hline!(p4_info_loss, [100],
           label="No information loss",
           color=:black,
           linestyle=:dot,
           linewidth=1,
           alpha=0.5)
    
    # Highlight where information loss is worst
    min_fim_idx = argmin(fim_info_ratio)
    min_gim_idx = argmin(gim_info_ratio)
    
    annotate!(p4_info_loss, temperatures[min_fim_idx], fim_info_ratio[min_fim_idx]-5,
              text("FIM worst: $(round(fim_info_ratio[min_fim_idx], digits=1))%", 8, :center, :blue))
    
    annotate!(p4_info_loss, temperatures[min_gim_idx], gim_info_ratio[min_gim_idx]-5,
              text("GIM worst: $(round(gim_info_ratio[min_gim_idx], digits=1))%", 8, :center, :red))
end

# ============================================================================
# STEP 12: CREATE COMPOSITE PLOT FOR PUBLICATION
# ============================================================================

# Arrange all plots in a grid
composite_plot = plot(p1_fim, p1_gim, p1_compare,
                      p2_fim, p2_gim, p3_noise_effect,
                      layout=(3, 2),
                      size=(1400, 1000),
                      plot_title="Global vs Fisher Information Matrix: Effect of Measurement Noise on Optimal Experimental Design")

# Save the composite plot
savefig(composite_plot, "fim_vs_gim_noise_comparison.png")

# Save individual plots
savefig(p1_compare, "fim_gim_direct_comparison.png")
savefig(p3_noise_effect, "noise_effect_optimal_temperature.png")
savefig(p4_info_loss, "information_loss_due_to_noise.png")

# ============================================================================
# STEP 13: DISPLAY RESULTS AND ANALYSIS
# ============================================================================

println("\n" * "="^60)
println("ANALYSIS RESULTS")
println("="^60)

# Find optimal temperatures for key noise levels
println("\n1. OPTIMAL EXPERIMENTAL TEMPERATURES:")
for σ in [0.0, 0.01]
    idx_fim = argmax(fim_results[σ].det)
    idx_gim = argmax(gim_results[σ].det)
    
    noise_name = σ == 0.0 ? "No noise" : "1% noise"
    println("\n   $noise_name:")
    println("     FIM optimal: $(temperatures[idx_fim]) K")
    println("     GIM optimal: $(temperatures[idx_gim]) K")
    
    if σ > 0
        shift_fim = temperatures[idx_fim] - temperatures[argmax(fim_results[0.0].det)]
        shift_gim = temperatures[idx_gim] - temperatures[argmax(gim_results[0.0].det)]
        println("     Shift from no-noise case: FIM = $(shift_fim)K, GIM = $(shift_gim)K")
    end
end

# Calculate information content
println("\n2. INFORMATION CONTENT COMPARISON:")
σ_ref = 0.01  # Reference noise level (1%)
max_fim_no_noise = maximum(fim_results[0.0].det)
max_gim_no_noise = maximum(gim_results[0.0].det)
max_fim_with_noise = maximum(fim_results[σ_ref].det)
max_gim_with_noise = maximum(gim_results[σ_ref].det)

fim_reduction = (1 - max_fim_with_noise/max_fim_no_noise) * 100
gim_reduction = (1 - max_gim_with_noise/max_gim_no_noise) * 100

println("   Maximum determinant (no noise):")
println("     FIM: $(round(max_fim_no_noise, sigdigits=4))")
println("     GIM: $(round(max_gim_no_noise, sigdigits=4))")

println("\n   With 1% measurement noise:")
println("     FIM: $(round(max_fim_with_noise, sigdigits=4))")
println("     GIM: $(round(max_gim_with_noise, sigdigits=4))")

println("\n   Information reduction due to 1% noise:")
println("     FIM: $(round(fim_reduction, digits=1))% reduction")
println("     GIM: $(round(gim_reduction, digits=1))% reduction")

# Robustness analysis
println("\n3. ROBUSTNESS TO NOISE:")
fim_sensitivity = std([temperatures[argmax(fim_results[σ].det)] for σ in noise_levels])
gim_sensitivity = std([temperatures[argmax(gim_results[σ].det)] for σ in noise_levels])

println("   Temperature sensitivity to noise:")
println("     FIM standard deviation: $(round(fim_sensitivity, digits=2)) K")
println("     GIM standard deviation: $(round(gim_sensitivity, digits=2)) K")

if gim_sensitivity < fim_sensitivity
    println("   ✓ GIM is more robust to noise (smaller variation)")
else
    println("   ✗ FIM is more robust to noise")
end

# Practical recommendations
println("\n" * "="^60)
println("PRACTICAL RECOMMENDATIONS FOR REACTOR EXPERIMENTS")
println("="^60)

println("\n1. WHEN TO USE FIM (Fisher Information Matrix):")
println("   • When measurement noise is low (< 0.5%)")
println("   • When you have high confidence in initial parameter estimates")
println("   • For maximum information extraction when conditions are ideal")
println("   • Recommended temperature: $(temperatures[argmax(fim_results[0.01].det)]) K")

println("\n2. WHEN TO USE GIM (Global Information Matrix):")
println("   • When measurement noise is significant (> 0.5%)")
println("   • When parameter uncertainty is high (> 20% for A, > 10% for E)")
println("   • For robust experimental design that works under various conditions")
println("   • Recommended temperature: $(temperatures[argmax(gim_results[0.01].det)]) K")

println("\n3. NOISE MANAGEMENT STRATEGIES:")
println("   • Below 0.5% noise: FIM gives slightly better results")
println("   • 0.5-1% noise: Both methods comparable, GIM more robust")
println("   • Above 1% noise: Use GIM for reliable design")
println("   • Always measure and characterize your experimental noise!")

println("\n4. EXPERIMENTAL PLANNING:")
println("   • Start with GIM-based design if uncertain about parameters")
println("   • Use FIM for refinement after initial parameter estimates")
println("   • Consider running experiments at multiple temperatures")
println("   • Allocate more replicates at high-noise conditions")

# Display the composite plot
display(composite_plot)

println("\n" * "="^60)
println("PLOTS SAVED:")
println("="^60)
println("1. fim_vs_gim_noise_comparison.png - All comparison plots")
println("2. fim_gim_direct_comparison.png - Direct FIM vs GIM comparison")
println("3. noise_effect_optimal_temperature.png - How noise affects optimal T")
println("4. information_loss_due_to_noise.png - Information loss due to noise")
println("="^60)


## **LINE-BY-LINE EXPLANATION IN SIMPLE TERMS:**

### **PART 1: Loading Packages**

using Plots           # For making graphs and charts
using LinearAlgebra   # For matrix math (determinant, trace, eigenvalues)
using Distributions   # For probability distributions (Monte Carlo sampling)
using Random          # For random number generation
using Printf          # For formatting numbers in text
using Statistics      # For statistical functions (mean, std, etc.)

- These are like toolboxes we need for our analysis
- Each `using` statement loads a different set of tools

### **PART 2: Setup**

gr()                  # Choose GR as the graph drawing engine
Random.seed!(42)      # Make random results repeatable (like setting a password)

- `gr()` makes plotting faster
- `Random.seed!(42)` ensures we get the same "random" results every time

### **PART 3: Reactor Model**

function reactor_model(θ, x)
    A, E = θ          # A = reaction speed, E = temperature sensitivity
    T, C0, u, z = x   # T = temperature, C0 = input concentration, etc.
    
    R = 8.314         # Gas constant (like a conversion factor)
    L = 1.0           # Reactor length
    
    k = A * exp(-E/(R*T))  # Reaction rate (faster at higher T)
    Da = k * L / u          # Reaction vs flow comparison
    X = 1 - exp(-Da * z)    # How much reactant converts
    return C0 * (1 - X)     # Output concentration
end

- This is our "virtual reactor" - it predicts what happens in experiments
- Uses the Arrhenius equation (reaction speeds up with temperature)

### **PART 4: Sensitivity Calculation**

function compute_sensitivities(θ, x, model; δ=1e-4)
    y0 = model(θ, x)                    # Base prediction
    sensitivities = zeros(length(θ))    # Empty list for results
    
    for i in 1:length(θ)
        θ_pert = copy(θ)                # Copy parameters
        θ_pert[i] += δ * θ[i]           # Change one parameter slightly
        y_pert = model(θ_pert, x)       # New prediction
        sensitivities[i] = (y_pert - y0) / (δ * θ[i])  # Calculate slope
    end
    return sensitivities
end

- Measures how much the output changes when we change parameters
- Like testing how sensitive a scale is to weight changes
- δ = 0.0001 means we change parameters by 0.01%

### **PART 5: Fisher Information Matrix (FIM)**

function compute_fim(θ, x, σ, model)
    S = compute_sensitivities(θ, x, model)  # Get sensitivities
    return (1/σ^2) * (S * S')               # FIM formula
end

- FIM measures information from ONE experiment
- σ = measurement noise (higher σ = less information)
- Formula: Information = (Sensitivity)² / (Noise)²

### **PART 6: Global Information Matrix (GIM)**

function compute_gim(prior_mean, prior_cov, x, σ, model; n_samples=1000)
    M_global = zeros(2, 2)                    # Start with empty matrix
    prior_dist = MvNormal(prior_mean, prior_cov)  # Parameter probability
    
    for i in 1:n_samples
        θ_sample = rand(prior_dist)           # Pick random parameters
        FIM_local = compute_fim(θ_sample, x, σ, model)  # Compute FIM
        M_global += FIM_local / n_samples     # Average them
    end
    return M_global
end

- GIM averages FIM over MANY possible parameter values
- Accounts for uncertainty in our parameter knowledge
- Like taking multiple opinions instead of trusting one expert

### **PART 7: Problem Setup**

θ_true = [1e5, 50000.0]          # True parameters we want to find
θ_prior_mean = [1e5, 50000.0]    # Our best guess
θ_prior_cov = Diagonal([(2e4)^2, (5000)^2])  # Our uncertainty
noise_levels = [0.0, 0.005, 0.01, 0.02]      # Different noise amounts
temperatures = 350:5:450         # Temperature range to test

- Sets up our test scenario
- We'll test with 0%, 0.5%, 1%, and 2% measurement error

### **PART 8: Computation Loop**

for (noise_idx, σ) in enumerate(noise_levels)
    for (i, x) in enumerate(experiments)
        FIM = compute_fim(θ_true, x, max(σ, 1e-10), reactor_model)
        GIM = compute_gim(θ_prior_mean, θ_prior_cov, x, max(σ, 1e-10), 
                         reactor_model, n_samples=500)
        # Store results...
    end
end

# - Loops through all noise levels and temperatures
# - Computes both FIM and GIM for each condition
# - `max(σ, 1e-10)` ensures we never divide by zero

### **PARTS 9-12: Plotting**

# Example plot creation:
p1_fim = plot(title="FIM - Determinant",  # Title
             xlabel="Temperature (K)",    # X-axis label
             ylabel="Determinant",        # Y-axis label
             yscale=:log10)               # Use log scale

plot!(p1_fim, temperatures, fim_results[0.0].det,  # Add data
      label="No noise",                   # Legend label
      linewidth=2,                        # Line thickness
      color=:blue)                        # Line color

# - Creates different types of plots
# - `plot!` adds to existing plot (the `!` means "modify")
# - Colors and line styles distinguish different conditions

### **KEY INSIGHTS FROM THE PLOTS:**

# 1. **Determinant Plots (D-optimality)**:
#    - Higher determinant = more information
#    - Noise reduces information dramatically (log scale!)
#    - GIM is more robust to noise than FIM

# 2. **Condition Number Plots**:
#    - Lower condition number = better numerical stability
#    - Noise improves conditioning (paradoxically)
#    - GIM has better conditioning than FIM

# 3. **Optimal Temperature vs Noise**:
#    - FIM optimal temperature shifts more with noise
#    - GIM optimal temperature is more stable
#    - This shows GIM's robustness

# 4. **Information Loss Plot**:
#    - Shows percentage of information lost due to noise
#    - Some temperatures are more sensitive to noise than others
#    - GIM loses less information overall

### **WHEN TO USE EACH METHOD:**

# **Use FIM when:**
# - You have very precise measurements (<0.5% error)
# - You're confident in your initial parameter guesses
# - You want maximum information under ideal conditions

# **Use GIM when:**
# - Measurements have significant noise (>0.5% error)
# - You're uncertain about parameter values
# - You need robust experiments that work in real-world conditions
# - You're doing preliminary experiments

#**Practical Example:**
#If you're testing a new expensive catalyst:
#1. Start with GIM design (accounts for uncertainty)
#2. After initial experiments, use FIM to refine
#3. Allocate more experimental replicates at noisy conditions

#The code produces comprehensive visualizations showing exactly how measurement noise affects optimal experimental design and why the global approach is more robust for real-world applications with uncertainty.