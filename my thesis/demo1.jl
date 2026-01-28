using NLPModels
using ADNLPModels
using NLPModelsIpopt
using Ipopt

myobj(x)=(1 - x[1])^2 + 100(x[2] - x[1]^2)^2
mycon(x) = [x[1]^2 + x[2]^2 - 1; [x[1] + 3*x[2] - 5]]

x0 = [1.2, 0.5]
mycon(x0)

model =ADNLPModel(myobj, x0, mycon, [-Inf, -Inf], [0.0, 0.0])
output = ipopt(model)

xstar = output.solution
fstar = output.objective

mycon(xstar)