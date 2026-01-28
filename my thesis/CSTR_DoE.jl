using NLsolve
using Optim
using QuasiMonteCarlo
using Distributions
using Statistics

function CSTR_model(ca_in, Temp; tau=5, n=1, k=0.0)
    sol = nlsolve((F, CA) -> concentration_compute(F, CA; k=k, ca_in=ca_in, Temp=Temp, n=n, tau=tau), [0.1], ftol=1e-15, show_trace=false) #using the non-linear solver to compute the concentrations

    # #Desired Solution
    ca_out = sol.zero

    return ca_out
end

#random points generator for one variable. For more variables enhance the size of lb and ub
function random_points_generator(Nexps; lb=[0.0], ub=[1.0], Sampling=HaltonSample())
    s = QuasiMonteCarlo.sample(Nexps, lb, ub, Sampling)
    return s
end

#Function to compute experimental outlet concentrations 
function ca_exp(ca_ins, Temp; Nexps=0, k=1.0, add_noise=false, sigma=1e-6, N_repeats=5, order=1.0)
    ca_out_without_noise = zeros(Nexps)
    ca_out_matrix = zeros(N_repeats, Nexps)

    for i in 1:Nexps
        ca_out_without_noise[i] = CSTR_model(ca_ins[i], Temp[i]; k=k, n=order)[1]
        if add_noise
            ca_out_matrix[:, i] = ca_out_without_noise[i] .+ rand(Normal(0, sigma), N_repeats)
        else
            ca_out_matrix[:, i] = ca_out_without_noise[i] .+ rand(Normal(0, 0.0), N_repeats)
        end
    end

    ca_out = mean(ca_out_matrix, dims=1)

    return ca_out

end

#
function concentration_compute(F, CA; k, n=order, Temp=0.0, ca_in=0.0, tau=0.0)
    F[1] = 1 / tau * (ca_in - CA[1]) - k[1] * exp(-k[2] / (8.314 * Temp)) * CA[1]^n
end

#non-linear solver to compute the outlet concentrations for CSTR 
function ca_model(k; ca_in=0.0, Temp=0.0, n=0.0, tau=5.0, ra!)
    sol = nlsolve((F, CA) -> ra!(F, CA; k=k, ca_in=ca_in, Temp=Temp, n=n, tau=tau), [0.1], ftol=1e-15, show_trace=false)
    CA = sol.zero
    return CA
end

#Function to estimate the reaction parameters k 
function parameter_estimator(; Nexps=1, ca_exp=0.0, ca_in=0.0, Temp=0.0, order=1, initial_guess=[0.0, 0.0])
    function param_estim(k)
        return sum((ca_exp[i] - ca_model(k; ca_in=ca_in[i], Temp=Temp[i], n=order, tau=5.0, (ra!)=concentration_compute)[1])^2 for i in 1:Nexps)
    end

    solver = LBFGS() #The solver we use for the inverse problem
    k0 = initial_guess #The initial guess for the parameters
    res = Optim.optimize(param_estim, k0, solver, Optim.Options(show_trace=false, g_abstol=1e-15))
    @show Optim.minimizer(res)
    @show Optim.minimum(res)
    @show Optim.g_converged(res)
end


function run_parameter_estimation(; Nexps=5, Sampling=HaltonSample(), add_noise=true, sigma=1e-3, order=1)
    Temp = random_points_generator(Nexps; lb=[200.0], ub=[400.0], Sampling=Sampling)
    ca_ins = random_points_generator(Nexps; lb=[0.1], ub=[1.0], Sampling=Sampling) #This function needs to be replaced  with optimal points 
    ca_exper = ca_exp(ca_ins, Temp; Nexps=Nexps, k=[1.0, 20000], add_noise=add_noise, sigma=sigma, order=order)
    parameter_estimator(; Nexps=Nexps, ca_exp=ca_exper, ca_in=ca_ins, Temp=Temp, order=order, initial_guess=[0.0, 0.0])
end


#TODOs
#add more parameters / species flexibly
#add weighted least square formulation also 

