module Example_Inverse_Problem_Validation_2D


using Revise
using Triangulate
using SimplexGridFactory
using ExtendableGrids
using VoronoiFVM
using LinearAlgebra
using NLsolve
using TimerOutputs
using LinearSolve
using ExtendableSparse
using Statistics
using Distributions
using QuasiMonteCarlo
using DelimitedFiles
using AMGCLWrap
using AlgebraicMultigrid
using GridVisualize
#using Plots
#using CairoMakie
using inverse_problem_RBS_CFD_het_cat
#using BenchmarkTools
using LaTeXStrings
using Serialization
using Base.Threads
import Random
import Plots
#include("ROM_Par_Sensitivity.jl")
function mixture_average_diffusivity(D, inlet_mass_frac, molar_weights; nospec=6)
    #Calculation of ZRS diffusivities 
    M = 0.0
    MInv = 0.0
    rho = 0.0
    Diff_Inv = zeros(nospec)
    Diff_Sum = 0.0
    c = 0.0 #Correction Velocity

    for i in 1:nospec
        MInv += (inlet_mass_frac[i]) / (molar_weights[i])
    end

    M = 1 / MInv

    for i in 1:nospec
        for j in 1:nospec
            if j != i
                Diff_Inv[i] += (inlet_mass_frac[j]) * M / (D[i, j] * molar_weights[j]) + (inlet_mass_frac[i]) * M * (inlet_mass_frac[j]) / ((1 - (inlet_mass_frac[i])) * D[i, j] * molar_weights[i])
                # Diff_Inv[i,2]=u[j,2]*M/(D[i,j]*molar_weights[j])+u[i,2]*M*u[j,2]/((1-u[i,2])*D[i,j]*molar_weights[i])
            end
        end
    end

    Diff_const = 1 ./ Diff_Inv


    return Diff_const
end

function L_BFs(xcoord, levels, λ, L)

    part_length = λ / levels
    ##For Level=1 this would be the single basis normal approach 
    if levels == 1
        return 1.0
    elseif levels == 2
        mid_points = [L / 2]
    else
        mid_points = collect(range(((L - λ) / 2) + part_length / 2, ((L + λ) / 2) - part_length / 2, levels - 1))
    end

    #println(mid_points)

    #New coordinates values, all between 0-1. 
    #Then apply the if else condition to the function 
    if xcoord <= minimum(mid_points)
        #rel_coord = (xcoord - (L - λ) / 2) / (minimum(mid_points) - (L - λ) / 2)
        #return rel_coord
        rel_coord = (xcoord - (L - λ) / 2) / (minimum(mid_points) - (L - λ) / 2)
        return -rel_coord + 1
    elseif xcoord >= maximum(mid_points)
        #rel_coord = (xcoord - maximum(mid_points)) / (maximum(mid_points) - (L + λ) / 2)
        #return rel_coord + 1
        rel_coord = (xcoord - maximum(mid_points)) / (-maximum(mid_points) + (L + λ) / 2)
        return rel_coord
    else
        #println(mid_points)
        for (i, v) in enumerate(mid_points)
            if xcoord >= v && xcoord <= (v + mid_points[i+1]) / 2 #need to filter case where i+1 doesnt exist
                # rel_coord = (xcoord - v) / ((v + mid_points[i+1]) / 2 - v)
                # return -rel_coord + 1
                rel_coord = (-xcoord + (v + mid_points[i+1]) / 2) / (v - (v + mid_points[i+1]) / 2)
                return 1 + rel_coord
            elseif xcoord <= v #&& xcoord <= (v + mid_points(i + 1)) / 2
                # rel_coord = (xcoord - (v + mid_points[i-1]) / 2) / (v - (v + mid_points[i-1]) / 2)
                # return rel_coord
                rel_coord = (xcoord - v) / ((v + mid_points[i-1]) / 2 - v)
                return rel_coord
            else
            end
        end
    end

end

function main(; nref=1, T=493, P_total=10, levels=1, RBS=false, St=0.0, ratio=0.01, Mtcat=1, initvalue=0.1, k0=0.0, Ea=0.0, RBS_full=false, catcell=1, inlet_MFs=[0.3, 0.7, 0.0], strategy=nothing, unknown_storage=:dense, assembly=:cellwise)
    # parameters to be varied
    #T = T
    L = 0.1
    H = 0.026
    nospec = size(St, 2)
    Nreacs = size(St, 1)
    xsp = 0.5 * (1 - ratio)
    xep = 0.5 * (1 + ratio)
    #molar_weights = [44.01, 2.016, 18.01528, 32.04, 28.01, 28.0134]
    #molar_weights =  [44.01, 2.016, 18.01528]
    #mixture_density = density_rechner(molar_weights, inlet_MFs, P_total, T)

    #inlet = inlet_MFs#[0.3, 0.7, 0.0]
    λ = L * ratio
    levels = levels
    v = 1.5e-1 #Set to 0.15
    #nospec=5; #
    arr_out = []
    fac = 0.0

    D = binary_diff(T, P_total)

    #D_mix_avg = mixture_average_diffusivity(D, inlet_MFs, molar_weights; nospec)

    # define a Hagen-Poiseuille flow velocity field
    function fhagen_poiseuille(x, y)
        yh = y / H
        return 4 * v * yh * (1.0 - yh), 0
    end


    h = 1.0 / convert(Float64, nref - 1)
    X = collect(0.0:h:L)
    Y = collect(0.0:h:H)
    grid = VoronoiFVM.Grid(X, Y)
    bfacemask!(grid, [xsp * L, 0], [xep * L, 0], 5)
    bfacemask!(grid, [xep * L, 0], [L, 0], 6)


    evelo = edgevelocities(grid, fhagen_poiseuille)
    bfvelo = bfacevelocities(grid, fhagen_poiseuille)


    Di = zeros(nospec)
    #Di=rand(nospec)
    fill!(Di, 1e-5)
    function constant_diff_flux!(f, u, edge, data) #Need to revisit this with some shortcuts and exact types 
        @timeit "Flux Time" begin

            for i in 1:nospec
                vd = evelo[edge.index] / Di[i]
                bp = fbernoulli(vd)
                bm = fbernoulli(-vd)
                f[i] = Di[i] * (bp * u[i, 1] - bm * u[i, 2])
            end
        end
    end

    # function fitting_flux!(f, u, edge, data)
    #     # c=0.0
    #     # for i in 1:nospec
    #     #     c += D_mix_avg[i] * (u[i, 2] - u[i, 1]) #Calculating the correction velocities
    #     # end
    #     for i in 1:nospec
    #         vd = (evelo[edge.index]) / D_mix_avg[i]
    #         bp = fbernoulli(vd)
    #         bm = fbernoulli(-vd)
    #         f[i] = D_mix_avg[i] * (bp * u[i, 1] - bm * u[i, 2])
    #     end
    # end

    function bconditions!(y, u, node, data)
        nindex = node.index

        for i in 1:(nospec)
            #outflow bc for the outlet
            if node.region == 2
                y[i] = bfvelo[node.ibnode, node.ibface] * u[i]
                push!(arr_out, node.ibface)
                #println(node.ibface)
                #println(size(arr_out))
                fac = node.fac
            end

        end


        for i in 1:(nospec)

            #neuman condition for our catalyst and inert boundaries
            boundary_neumann!(y, u, node, species=i, region=3, value=0)
            if RBS == true
                if RBS_full == true
                    # if node.region == 5
                    #     if nindex == catcell
                    #         @show catcell
                    #         y[i] = -1 * St_Inv[i]
                    #     end
                    # end
                    if node.region == 5
                        if nindex == catcell
                            y[i] = -1 #* molar_weights[i] / mixture_density #* St[1, i]
                        end
                    end
                else
                    boundary_neumann!(y, u, node, species=i, region=5, value=1.0 * L_BFs(node[1], levels, λ, L))
                end
                #dirichlet condition for our inlet
                boundary_dirichlet!(y, u, node, species=i, region=4, value=0.0)#inlet_mol_frac#[i])
            else
                #For Reactions Requiring Pressure Conversion
                #boundary_neumann!(y, u, node, species=i, region=5, value=(1 * molar_weights[i] / mixture_density * (St[1, i] * rf(u, k0; T, density=mixture_density, molar_weights=molar_weights)[1] + St[2, i] * rf(u, k0; T, density=mixture_density, molar_weights=molar_weights)[2])))#

                #For Reactions Not Requiring Pressure Conversion
                boundary_neumann!(y, u, node, species=i, region=5, value=((St[1, i] * rf(u, k0; T)[1])))#
                #boundary_neumann!(y, u, node, species=i, region=5, value=sum(1 * St_Rober[j, i] * rf(u, k0; T)[j] for j in 1:3))# + St[2, i] * rf(u, k0; T)[2])

                #dirichlet condition for our inlet
                boundary_dirichlet!(y, u, node, species=i, region=4, value=inlet_MFs[i])#inlet_mol_frac#[i])
            end
        end

    end


    dec = 2 # remove this parameter after testing !
    # DBinary = Symmetric(rand(6, 6))

    #data = MyData{nospec}(DBinary, grid, evelo, bfvelo, dec)
    data = nothing
    ysys = VoronoiFVM.System(grid; flux=constant_diff_flux!,
        bcondition=bconditions!, data, assembly)

    for i in 1:nospec
        enable_species!(ysys, i, [1])
    end

    arr_out

    Γ_inlet = [4]
    Γ_outlet = [2]
    Γ_catalyst = [5]



    #bfaceindices = unique(arr_out)
    #println(size(bnodes))

    @info "Strategy: $(strategy)"
    control = SolverControl(strategy, ysys)
    @info control.method_linear
    #method_linear = KrylovJL_BICGSTAB(), precon_linear = SA_AMGPreconditioner(), keepcurrent_linear = true,  tol_round=0.0, tol_mono=0.0, 
    @timeit "Overall Time" begin
        tsol = VoronoiFVM.solve(ysys; maxiters=1000, inival=initvalue, abstol=1e-12, reltol=0.0, verbose=false, tol_round=0.0, tol_mono=0.0, log=true, damp_initial=0.15) #, damp_grow = 1.2) #0.15,damp_grow = 1.5)
    end

    factory = VoronoiFVM.TestFunctionFactory(ysys)
    tfc_rea = testfunction(factory, [Γ_inlet; Γ_outlet; [1]; [3]; [6]], [Γ_catalyst;])
    tfc_in = testfunction(factory, [Γ_outlet; Γ_catalyst; [1]; [3]; [6]], [Γ_inlet;])
    tfc_out = testfunction(factory, [Γ_inlet; Γ_catalyst; [1]; [3]; [6]], [Γ_outlet;])

    I = integrate(ysys, tfc_rea, tsol)
    # Iin = integrate(ysys, tfc_in, tsol)
    Iout = integrate(ysys, tfc_out, tsol)

    #print_timer()
    sub = subgrid(grid, [5], boundary=true, transform=(a, b) -> (a[1] = b[1]))
    sub2 = subgrid(grid, [1], boundary=true, transform=(a, b) -> (a[1] = b[1]))
    sub3 = subgrid(grid, [6], boundary=true, transform=(a, b) -> (a[1] = b[1]))
    sub4 = subgrid(grid, [2], boundary=true, transform=(a, b) -> (a[1] = b[2]))

    return tsol, sub, sub2, sub3, sub4, grid, bfvelo, arr_out, fac

end

function Inverse_Problem_Paras(; nref=2500, ratio=0.01, nspec=0, T=493, inlet_MFs=0.0, St=St) #TODO should go in source files 
    d = main(nref=nref, RBS=true, ratio=ratio, inlet_MFs=inlet_MFs, T=T, St=St)
    #a = yout_weighted(d)
    #b=ycat_weighted(d)
    snapshot_A = []
    snapshot_B = []
    for i in 1:nspec
        A = 0 .+ view(d[1][i, :], d[5]) #TODO write A,B for all species with different diffusivity 
        B = 0 .+ view(d[1][i, :], d[2])
        push!(snapshot_A, A)
        push!(snapshot_B, B)
    end
    return snapshot_A, snapshot_B
    #return a, B
end

function experimental_data(sample; Nspec=0.0, Nexps=0)
    Y_in = zeros(Nspec, Nexps)
    for i in 1:Nspec-1
        Y_in[i, :] = sample[i, :]
    end
    Y_in[end, :] = 1 .- sum(Y_in[i, :] for i in 1:(Nspec-1))


    Temp = sample[end, :] #zeros(Nexps)

    return Y_in, Temp
end

function random_points_generator(; Nexps=3, nspecs=6, lb=[0.1, 0.2, 0.0, 0.0, 0.0, 300], ub=[0.4, 0.5, 0.01, 0.01, 0.01, 600], Sampling=HaltonSample())
    s = QuasiMonteCarlo.sample(Nexps, lb, ub, Sampling)
    Yin, Temp = experimental_data(s; Nspec=nspecs, Nexps=Nexps)
    return Yin, Temp
end

function experiments(; Y_in, Temp, P_total, Nexps, ratio, N_repeats, std_data, Nspec, k_true)
    St =  [-2 -1 2] #[-1 -3 1 1 0 0; -1 -1 1 0 1 0]
    P_total = P_total
    molar_weights =[1.0,1.0,1.0]# [44.01, 2.016, 18.01528, 32.04, 28.01, 28.0134]
    mixture_density = zeros(Nexps)
    Yexp = zeros(Nspec, Nexps)
    @info "Collecting Experimental Data using FOM CFD Simulations"
    @info "Total Number of Experiments is $Nexps"
    for i in 1:Nexps
        mixture_density[i] = 1.0 #density_rechner(molar_weights, Y_in[:, i], P_total, Temp[i])
        d = main(nref=2500, inlet_MFs=Y_in[:, i], ratio=ratio, St=St, k0=k_true, T=Temp[i]) #T + i * ΔT
        Yout = youts(d, Nspec=Nspec) #yout_weighted(d;Nspecs=Nspec) #
        for j in 1:Nspec
            Yexp[j, i] = Yout[j]
        end
    end
    σ_data = std_data
    Yexp_with_Error = Array{Float64}(undef, N_repeats, Nspec, Nexps)
    for i in 1:Nexps
        for n in 1:Nspec
            Yexp_with_Error[:, n, i] = Yexp[n, i] .+ rand(Normal(0, σ_data), N_repeats)
        end
    end

    return Yexp_with_Error
end

function parameter_estimator(; ratio, nspec, Y_in, Temp, P_total, St, nref=2500, nreac, Nexps, Y_out, unknown_parameters, IG, N_repeats, σ_data, RBS_full=false)
    single_snapshot_A = []
    single_snapshot_B = []
    rbs_snapshot = []
    srbs_time = 0.0
    rbs_time = 0.0
    mixture_density = zeros(Nexps)
    X = fill(σ_data, nspec)
    V = (Diagonal(X) .^ 2) #./ 12 
    @info "Evaluating Kinetic Parameters"
    molar_weights = [1.0,1.0,1.0]#[44.01, 2.016, 18.01528, 32.04, 28.01, 28.0134]
    B_RBS = 0.0
    for j in 1:Nexps
        srbs = @timed A, B = Inverse_Problem_Paras(; nref=nref, ratio=ratio, nspec=nspec, inlet_MFs=Y_in[:, j], T=Temp[j], St=St)
        mixture_density[j] = 1.0 #density_rechner(molar_weights, Y_in[:, j], P_total, Temp[j])
        push!(single_snapshot_A, A)
        push!(single_snapshot_B, B)
        # #srbs_time += srbs.time
        if RBS_full == true
            @info "Conducting Offline Step for Full Reduced Basis"
            rbs = @timed B_RBS = RBS_Snapshots(main; nref=nref, ratio=ratio, St=St, Nexps=Nexps, nspec=nspec, nreac=nreac, inlet_MFs=Y_in[:, j], T=Temp[j])
            push!(rbs_snapshot, B_RBS)
            rbs_time += rbs.time
        else
            B_RBS = 0.0
        end
    end

    if RBS_full == true
        @info "Estimating Parameters using RBS"
        k, _ = newton_optimizer(single_snapshot_A, single_snapshot_B, Y_in, Y_out; mixture_density=mixture_density, Initial_Guess=IG, molar_weights=molar_weights, B_RBS=rbs_snapshot, st=St, cov=V, dof=unknown_parameters, RBS=true, print=true, lm=true, T=Temp, Nexps=Nexps, N_measurements=N_repeats)
    else
        @info "Estimating Parameters using Single Reduced Basis with Forward Solver"
        k, _ = newton_optimizer(single_snapshot_A, single_snapshot_B, Y_in, Y_out; Fwd=true, mixture_density=mixture_density, molar_weights=molar_weights, Initial_Guess=IG, B_RBS=B_RBS, st=St, cov=V, dof=unknown_parameters, RBS=false, print=true, lm=true, T=Temp, Nexps=Nexps, N_measurements=N_repeats)
    end 

    par = k 

    return par
end

#RUN this function to exectute the complete parameter estimation workflow 
function complete_workflow(; Nexps=5, ratio=0.1, nparas=2, std_data=1e-6, RBS_full=false)
    St=[-2 -1 2]
    P_total = 50
    nspec = size(St, 2)
    nreac = size(St, 1)
    @info "computing the experiments"
    k_true=[4000.0,4000.0]
    Y_in, Temp = random_points_generator(Nexps=Nexps, nspecs=nspec, lb=[0.1, 0.1, 300], ub=[0.5, 0.5, 600], Sampling=HaltonSample())
    Y_out = experiments(; Y_in=Y_in, P_total=P_total, Temp=Temp, Nexps=Nexps, ratio=ratio, N_repeats=50, std_data=std_data, Nspec=nspec, k_true=k_true) #
    @info "estimating paremeters"
    IG = fill(0.1, 2)
    k = parameter_estimator(; ratio=ratio, nspec=nspec, Y_in=Y_in, Temp=Temp, P_total=P_total, St=St, nref=2500, nreac=nreac, Nexps=Nexps, Y_out=Y_out, unknown_parameters=nparas, IG=IG, N_repeats=50, σ_data=std_data, RBS_full=RBS_full)
    return k
end

# ============================================================
# CFD-ROM OED USING main.jl + parameter_estimator
# Random Design + FIM + GIM
# Noise: 0%, 5%, 10%, 20%
# ============================================================

function safe_inv_oed(A; λ=1e-8)
    n = size(A, 1)
    return inv(Matrix(A) + λ * I(n))
end

function oed_safe_nref(; ratio, requested_nref, min_catalyst_nodes=5)
    L = 0.1
    λ = L * ratio
    needed = ceil(Int, 1 + min_catalyst_nodes / λ)
    return max(requested_nref, needed, 350)
end

function safe_logdet_oed(A; λ=1e-8)
    n = size(A, 1)
    return logdet(Symmetric(Matrix(A) + λ * I(n)))
end

function cfd_outlet_oed(Yin, T, k; St, ratio, nref)
    d = main(
        nref=nref,
        inlet_MFs=Yin,
        T=T,
        St=St,
        ratio=ratio,
        k0=k,
        RBS=false
    )
    return youts(d, Nspec=size(St, 2))
end

function cfd_sensitivity_oed(Yin, T, k; St, ratio, nref)
    nspec = size(St, 2)
    nparas = length(k)
    J = zeros(nspec, nparas)

    for p in 1:nparas
        h = 1e-4 * max(abs(k[p]), 1.0)

        kp = copy(k)
        km = copy(k)

        kp[p] += h
        km[p] -= h

        yp = cfd_outlet_oed(Yin, T, kp; St=St, ratio=ratio, nref=nref)
        ym = cfd_outlet_oed(Yin, T, km; St=St, ratio=ratio, nref=nref)

        J[:, p] .= (yp .- ym) ./ (2h)
    end

    return J
end

function build_candidate_database_oed(; Y_candidates, Temp_candidates,
                                      k_true, St, ratio, nref)
    Ncandidates = size(Y_candidates, 2)
    nspec = size(St, 2)

    Yclean = zeros(nspec, Ncandidates)
    Jlist = Vector{Matrix{Float64}}(undef, Ncandidates)

    for i in 1:Ncandidates
        @info "Building CFD-ROM candidate $i / $Ncandidates"

        Yclean[:, i] .= cfd_outlet_oed(
            Y_candidates[:, i],
            Temp_candidates[i],
            k_true;
            St=St,
            ratio=ratio,
            nref=nref
        )

        Jlist[i] = cfd_sensitivity_oed(
            Y_candidates[:, i],
            Temp_candidates[i],
            k_true;
            St=St,
            ratio=ratio,
            nref=nref
        )
    end

    return Yclean, Jlist
end

function candidate_fim_oed(J; noise_percent, yscale=1.0)
    σ = noise_percent == 0.0 ? 1e-8 : max(noise_percent / 100 * yscale, 1e-8)
    W = (1 / σ^2) * I(size(J, 1))
    return Matrix(J' * W * J)
end

function select_random_oed(; Nexps)
    return collect(1:Nexps)
end

function select_fim_oed(; Jlist, Yclean, noise_percent, Nexps)
    nparas = size(Jlist[1], 2)
    F = zeros(nparas, nparas)

    selected = Int[]
    remaining = collect(1:length(Jlist))

    while length(selected) < Nexps && !isempty(remaining)
        best_idx = remaining[1]
        best_score = -Inf

        for idx in remaining
            yscale = maximum(abs.(Yclean[:, idx]))
            Fi = candidate_fim_oed(Jlist[idx]; noise_percent=noise_percent, yscale=yscale)

            Ftest = F + Fi

            score =
                safe_logdet_oed(Ftest; λ=1e-6) -
                1e-4 * tr(safe_inv_oed(Ftest; λ=1e-6))

            if score > best_score
                best_score = score
                best_idx = idx
            end
        end

        push!(selected, best_idx)

        yscale = maximum(abs.(Yclean[:, best_idx]))
        F .+= candidate_fim_oed(Jlist[best_idx]; noise_percent=noise_percent, yscale=yscale)

        filter!(x -> x != best_idx, remaining)
    end

    return selected
end

function select_gim_oed(; Jlist, Yclean, noise_percent, Nexps)
    Ncandidates = length(Jlist)
    nparas = size(Jlist[1], 2)

    Fims = Matrix{Float64}[]

    for i in 1:Ncandidates
        yscale = maximum(abs.(Yclean[:, i]))
        push!(Fims, candidate_fim_oed(Jlist[i]; noise_percent=noise_percent, yscale=yscale))
    end

    GIM = sum(Fims)
    GIMinv = safe_inv_oed(GIM; λ=1e-6)

    selected = Int[]
    remaining = collect(1:Ncandidates)

    F = zeros(nparas, nparas)

    while length(selected) < Nexps && !isempty(remaining)
        best_idx = remaining[1]
        best_score = -Inf

        for idx in remaining
            Fi = Fims[idx]
            Ftest = F + Fi

            d_score = safe_logdet_oed(Ftest; λ=1e-6)
            global_score = tr(GIMinv * Fi)
            conditioning_penalty = tr(safe_inv_oed(Ftest; λ=1e-6))

            score = d_score + global_score - 1e-4 * conditioning_penalty

            if score > best_score
                best_score = score
                best_idx = idx
            end
        end

        push!(selected, best_idx)
        F .+= Fims[best_idx]
        filter!(x -> x != best_idx, remaining)
    end

    return selected
end

function fim_history_oed(; selected, Jlist, Yclean, noise_percent)
    nparas = size(Jlist[1], 2)
    F = zeros(nparas, nparas)

    fim_hist = zeros(length(selected))
    unc_hist = zeros(length(selected))

    for i in 1:length(selected)
        idx = selected[i]
        yscale = maximum(abs.(Yclean[:, idx]))

        F .+= candidate_fim_oed(Jlist[idx]; noise_percent=noise_percent, yscale=yscale)

        fim_hist[i] = safe_logdet_oed(F; λ=1e-6)
        unc_hist[i] = tr(safe_inv_oed(F; λ=1e-6))
    end

    return fim_hist, unc_hist
end

function noisy_data_oed(; selected, Y_candidates, Temp_candidates,
                        k_true, St, ratio, nref, noise_percent,
                        N_repeats)
    nspec = size(St, 2)
    Nexps = length(selected)

    Y_out = Array{Float64}(undef, N_repeats, nspec, Nexps)

    for e in 1:Nexps
        idx = selected[e]

        yclean = cfd_outlet_oed(
            Y_candidates[:, idx],
            Temp_candidates[idx],
            k_true;
            St=St,
            ratio=ratio,
            nref=nref
        )

        σ = noise_percent == 0.0 ? 1e-10 : noise_percent / 100 * maximum(abs.(yclean))

        for r in 1:N_repeats
            Y_out[r, :, e] .= yclean .+ rand(Normal(0.0, σ), nspec)
        end
    end

    return Y_out
end

function estimate_selected_design_oed(; selected, Y_candidates, Temp_candidates,
                                      k_true, IG, St, ratio, P_total,
                                      nref, noise_percent, N_repeats,
                                      RBS_full=false)

    nspec = size(St, 2)
    nreac = size(St, 1)
    Nexps = length(selected)

    Y_design = Y_candidates[:, selected]
    T_design = Temp_candidates[selected]

    Y_out = noisy_data_oed(
        selected=selected,
        Y_candidates=Y_candidates,
        Temp_candidates=Temp_candidates,
        k_true=k_true,
        St=St,
        ratio=ratio,
        nref=nref,
        noise_percent=noise_percent,
        N_repeats=N_repeats
    )

    σ_data = noise_percent == 0.0 ? 1e-10 : noise_percent / 100

    try
        k_est = parameter_estimator(
            ratio=ratio,
            nspec=nspec,
            Y_in=Y_design,
            Temp=T_design,
            P_total=P_total,
            St=St,
            nref=nref,
            nreac=nreac,
            Nexps=Nexps,
            Y_out=Y_out,
            unknown_parameters=length(k_true),
            IG=IG,
            N_repeats=N_repeats,
            σ_data=σ_data,
            RBS_full=RBS_full
        )

        return collect(k_est), true

    catch err
        @warn "parameter_estimator failed for $Nexps experiments at $(noise_percent)% noise"
        @warn err
        return copy(IG), false
    end
end

function parameter_history_oed(; selected, Y_candidates, Temp_candidates,
                               k_true, IG, St, ratio, P_total,
                               nref, noise_percent, N_repeats,
                               RBS_full=false)

    Nexps = length(selected)
    k_hist = Vector{Vector{Float64}}(undef, Nexps)
    err_hist = zeros(Nexps)
    conv_hist = falses(Nexps)

    last_k = copy(IG)

    for m in 1:Nexps
        if m < length(k_true) + 1
            k_hist[m] = copy(IG)
            err_hist[m] = norm(IG .- k_true) / norm(k_true)
            conv_hist[m] = false
        else
            @info "Calling parameter_estimator with $m selected experiments"

            k_est, ok = estimate_selected_design_oed(
                selected=selected[1:m],
                Y_candidates=Y_candidates,
                Temp_candidates=Temp_candidates,
                k_true=k_true,
                IG=last_k,
                St=St,
                ratio=ratio,
                P_total=P_total,
                nref=nref,
                noise_percent=noise_percent,
                N_repeats=N_repeats,
                RBS_full=RBS_full
            )

            if ok
                last_k = k_est
            end

            k_hist[m] = copy(last_k)
            err_hist[m] = norm(last_k .- k_true) / norm(k_true)
            conv_hist[m] = ok
        end
    end

    return k_hist, err_hist, conv_hist
end

function plot_oed_comparison(results; k_true, Nexps)
    x = collect(1:Nexps)

    p_err = Plots.plot(
        xlabel="Number of experiments",
        ylabel="Relative parameter error",
        title="Parameter Estimation Error",
        legend=:topright
    )

    p_fim = Plots.plot(
        xlabel="Number of experiments",
        ylabel="log(det(FIM))",
        title="Fisher Information Growth",
        legend=:bottomright
    )

    p_unc = Plots.plot(
        xlabel="Number of experiments",
        ylabel="trace(inv(FIM))",
        title="Parameter Uncertainty Reduction",
        legend=:topright
    )

    p_k1 = Plots.plot(
        xlabel="Number of experiments",
        ylabel="Estimated k₁",
        title="Convergence of k₁",
        legend=:bottomright
    )

    p_k2 = Plots.plot(
        xlabel="Number of experiments",
        ylabel="Estimated k₂",
        title="Convergence of k₂",
        legend=:bottomright
    )

    for row in results
        label = "$(row.method), $(row.noise)%"

        Plots.plot!(p_err, x, row.error_history, marker=:circle, label=label)
        Plots.plot!(p_fim, x, row.fim_history, marker=:circle, label=label)
        Plots.plot!(p_unc, x, row.uncertainty_history, marker=:circle, label=label)

        k1 = [v[1] for v in row.k_history]
        k2 = [v[2] for v in row.k_history]

        Plots.plot!(p_k1, x, k1, marker=:circle, label=label)
        Plots.plot!(p_k2, x, k2, marker=:circle, label=label)
    end

    Plots.hline!(p_k1, [k_true[1]], linestyle=:dash, label="true k₁")
    Plots.hline!(p_k2, [k_true[2]], linestyle=:dash, label="true k₂")

    Plots.savefig(p_err, "cfd_oed_parameter_error.png")
    Plots.savefig(p_fim, "cfd_oed_fim_growth.png")
    Plots.savefig(p_unc, "cfd_oed_uncertainty_reduction.png")
    Plots.savefig(p_k1, "cfd_oed_k1_convergence.png")
    Plots.savefig(p_k2, "cfd_oed_k2_convergence.png")

    return p_err, p_fim, p_unc, p_k1, p_k2
end

function cfd_oed_parameter_estimator_workflow(; Nexps=5,
                                              Ncandidates=3,
                                              ratio=0.1,
                                              nref=1,
                                              P_total=50,
                                              k_true=[4000.0, 4000.0],
                                              IG=[3000.0, 3000.0],
                                              N_repeats=8,
                                              RBS_full=false)

    St = [-2 -1 2]
    nref = oed_safe_nref(ratio=ratio, requested_nref=nref)
    @info "Using safe CFD nref = $nref"
    nspec = size(St, 2)

    noise_levels = [0.0, 5.0]    # add , 10.0, 20.0]

    lb = [0.1, 0.1, 300.0]
    ub = [0.45, 0.45, 600.0]

    @info "Generating candidate experimental space"
    Y_candidates, Temp_candidates = random_points_generator(
        Nexps=Ncandidates,
        nspecs=nspec,
        lb=lb,
        ub=ub,
        Sampling=HaltonSample()
    )

    @info "Building CFD-ROM sensitivity database"
    @show Yclean, Jlist = build_candidate_database_oed(
        Y_candidates=Y_candidates,
        Temp_candidates=Temp_candidates,
        k_true=k_true,
        St=St,
        ratio=ratio,
        nref=nref
    )

    all_results = []

    for noise in noise_levels
        @info "Noise level: $(noise)%"

        selected_random = select_random_oed(Nexps=Nexps)

        selected_fim = select_fim_oed(
            Jlist=Jlist,
            Yclean=Yclean,
            noise_percent=noise,
            Nexps=Nexps
        )

        selected_gim = select_gim_oed(
            Jlist=Jlist,
            Yclean=Yclean,
            noise_percent=noise,
            Nexps=Nexps
        )

        designs = [
            (:Random, selected_random),
            (:FIM, selected_fim),
            (:GIM, selected_gim)
        ]

        for d in designs
            method = d[1]
            selected = d[2]

            fim_hist, unc_hist = fim_history_oed(
                selected=selected,
                Jlist=Jlist,
                Yclean=Yclean,
                noise_percent=noise
            )

            @show k_hist, err_hist, conv_hist = parameter_history_oed(
                selected=selected,
                Y_candidates=Y_candidates,
                Temp_candidates=Temp_candidates,
                k_true=k_true,
                IG=IG,
                St=St,
                ratio=ratio,
                P_total=P_total,
                nref=nref,
                noise_percent=noise,
                N_repeats=N_repeats,
                RBS_full=RBS_full
            )

            push!(
                all_results,
                (
                    method=method,
                    noise=noise,
                    selected=selected,
                    k_history=k_hist,
                    error_history=err_hist,
                    fim_history=fim_hist,
                    uncertainty_history=unc_hist,
                    converged=conv_hist
                )
            )
        end
    end

    plots = plot_oed_comparison(all_results; k_true=k_true, Nexps=Nexps)

    open("cfd_oed_summary.csv", "w") do io
        println(io, "method,noise_percent,experiment,k1,k2,relative_error,logdet_fim,trace_inv_fim,converged")
        for row in all_results
            for i in 1:Nexps
                println(
                    io,
                    "$(row.method),$(row.noise),$(i),$(row.k_history[i][1]),$(row.k_history[i][2]),$(row.error_history[i]),$(row.fim_history[i]),$(row.uncertainty_history[i]),$(row.converged[i])"
                )
            end
        end
    end

    open("cfd_oed_selected_experiments.csv", "w") do io
        println(io, "method,noise_percent,order,candidate_index,Y1_in,Y2_in,Y3_in,Temperature")
        for row in all_results
            for i in 1:Nexps
                idx = row.selected[i]
                println(
                    io,
                    "$(row.method),$(row.noise),$(i),$(idx),$(Y_candidates[1,idx]),$(Y_candidates[2,idx]),$(Y_candidates[3,idx]),$(Temp_candidates[idx])"
                )
            end
        end
    end

    return (
        candidates=(Y_candidates=Y_candidates, Temp_candidates=Temp_candidates),
        clean_outputs=Yclean,
        sensitivities=Jlist,
        results=all_results,
        plots=plots
    )
end

end
