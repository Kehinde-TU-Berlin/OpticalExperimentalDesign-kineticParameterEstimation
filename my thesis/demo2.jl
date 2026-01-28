using NLPModels
using NLPModelsIpopt


function runoptimization(params, stressmax)


    function obj(x)
       mass, stress = structure(x, params)
       return mass

    end

    function obj(x)
       mass, stress = structure(x, params)
       return stress - stressmax

    end


    model =ADNLPModel(myobj, x0, mycon, [-Inf, -Inf], [0.0, 0.0])
    output = ipopt(model)
    xstar = output.solution
    fstar = output.objective

end


params= [100.0, 3.0]
stressmax = [1.0, 5.0]
runoptimization(params, stressmax)