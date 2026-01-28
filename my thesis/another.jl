using QuasiMonteCarlo

# Basic 1D sampling
lb = 0.0
ub = 1.0
n_samples = 10

samples = QuasiMonteCarlo.sample(n_samples, lb, ub, HaltonSample())
println("1D Halton samples: ", samples)

# Multi-dimensional sampling
lb = [0.0, 0.0]  # lower bounds for each dimension
ub = [1.0, 2.0]  # upper bounds for each dimension
n_samples = 8

samples = QuasiMonteCarlo.sample(n_samples, lb, ub, HaltonSample())
println("2D Halton samples:")
println(samples)


=======================================================================================================================================

# Sample multiple parameters together
n_experiments = 12
lb = [200.0, 0.1]   # [Temperature_low, Concentration_low]
ub = [400.0, 1.0]   # [Temperature_high, Concentration_high]

samples = QuasiMonteCarlo.sample(n_experiments, lb, ub, HaltonSample())

# Extract individual parameters
temperatures = samples[1, :]    # First row: temperatures
concentrations = samples[2, :]  # Second row: concentrations

println("Temperatures: ", temperatures)
println("Concentrations: ", concentrations)




=========================================================================================================================================



using QuasiMonteCarlo

function generate_experimental_design(Nexps; T_range=(200.0, 400.0), CA_range=(0.1, 1.0))
    lb = [T_range[1], CA_range[1]]
    ub = [T_range[2], CA_range[2]]
    
    samples = QuasiMonteCarlo.sample(Nexps, lb, ub, HaltonSample())
    
    temperatures = samples[1, :]
    concentrations = samples[2, :]
    
    return temperatures, concentrations
end

# Usage in your parameter estimation function
function run_parameter_estimation(; Nexps=5, add_noise=true, sigma=1e-3)
    # Generate optimal experimental conditions using Halton sampling
    Temp, ca_ins = generate_experimental_design(Nexps)
    
    # Rest of your function remains the same
    ca_exper = ca_exp(ca_ins, Temp; Nexps=Nexps, k=[1.0, 20000], add_noise=add_noise, sigma=sigma)
    
    result = parameter_estimator(; Nexps=Nexps, ca_exp=ca_exper, ca_in=ca_ins, Temp=Temp, 
                                initial_guess=[0.0, 0.0])
    
    return result, (Temp, ca_ins, ca_exper)
end
