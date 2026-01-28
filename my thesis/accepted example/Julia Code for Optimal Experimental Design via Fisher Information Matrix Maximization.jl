using JuMP, Ipopt, DifferentialEquations, ForwardDiff, LinearAlgebra

# Constants and nominal parameters
const R = 8.314  # Gas constant (J/mol·K)
const T = 300.0  # Fixed temperature (K)
const σ = 0.05   # Measurement noise standard deviation
const ns = 5     # Number of measurement times
const Δt = 2.0   # Minimum time interval
const tfinal = 20.0  # Final time
θ_nom = [1.0, 10000.0]  # Nominal [A, Ea]
u0 = [1.0]  # Initial concentration
tspan = (0.0, tfinal)

# ODE model for batch reactor
function ode!(du, u, p, t)
    A, Ea = p
    k = A * exp(-Ea / (R * T))
    du[1] = -k * u[1]
end

# Function to compute det(FIM) for given t_vec and θ
function compute_det(t_vec, θ)
    t_sorted = sort(t_vec)  # Ensure sorted times
    function get_measurements(θ)
        prob = ODEProblem(ode!, u0, tspan, θ)
        sol = solve(prob, Tsit5(), saveat=t_sorted)
        return vec(sol[1, :])  # Vector of measurements (scalar state)
    end
    jac = ForwardDiff.jacobian(get_measurements, θ)  # (ns x 2) Jacobian dy/dθ
    F = jac' * jac / σ^2  # FIM assuming independent noise
    return det(F + 1e-6 * I)  # Regularized for stability
end

# Wrapper for JuMP (takes variable arguments t1, t2, ...)
function my_det(t::Float64...)
    return compute_det(collect(t), θ_nom)
end

# Set up JuMP model
model = Model(Ipopt.Optimizer)
@variable(model, t[1:ns])
@constraint(model, t[1] >= 0.0)
@constraint(model, t[ns] <= tfinal)
for s in 1:ns-1
    @constraint(model, t[s+1] - t[s] >= Δt)
end

# Register and set nonlinear objective
JuMP.register(model, :my_det, ns, my_det; autodiff=true)
@NLobjective(model, Max, my_det(t[1], t[2], t[3], t[4], t[5]))  # Adjust for ns

# Random restarts to find global optimum
best_obj = -Inf
best_t = nothing
for i in 1:50
    # Generate feasible random initial guess (sorted, satisfying min diffs)
    deltas = Δt .+ exp.(randn(ns-1))  # Positive increments > Δt
    t_init = cumsum([rand() * Δt; deltas])
    t_init = t_init * (tfinal / t_init[end])  # Scale to fit within tfinal
    set_start_value.(t, t_init)
    
    optimize!(model)
    
    if termination_status(model) == LOCALLY_SOLVED
        obj_val = objective_value(model)
        if obj_val > best_obj
            best_obj = obj_val
            best_t = value.(t)
        end
    end
end

println("Optimal measurement times: ", best_t)
println("Maximized det(FIM): ", best_obj)