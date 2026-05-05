# run_optimal_design.jl
using Revise
using .Example_Inverse_Problem_Validation_2D
using inverse_problem_RBS_CFD_het_cat
include("optimal_design.jl")
using .OptimalExperimentalDesign

# Main execution
function run()
    println("="^60)
    println("OPTIMAL EXPERIMENTAL DESIGN FOR CATALYTIC REACTOR")
    println("="^60)
    
    # Run the optimal design workflow
    history, experiments, param_history, unc_history, plot = OptimalExperimentalDesign.main_optimal_design()
    
    println("\n" * "="^60)
    println("OPTIMAL DESIGN COMPLETED SUCCESSFULLY")
    println("="^60)
    
    return history, experiments, param_history, unc_history, plot
end

# Execute
if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
