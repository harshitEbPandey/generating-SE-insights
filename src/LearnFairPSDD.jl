module LearnFairPSDD

using LogicCircuits
using ProbabilisticCircuits
using DataFrames
using DataStructures

include("structures.jl")
include("data.jl")
include("logging.jl")

include("parameters.jl")
include("structure_inits.jl")
include("predictions.jl")
include("models.jl")
include("compareSE.jl")
end
