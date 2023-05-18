using DataStructures
using DataFrames
using CSV
using LogicCircuits
using ProbabilisticCircuits
using Combinatorics
using Random
include("structures.jl")
include("data.jl")
include("compareSE.jl")

println("in testStub")

name = "synthetic"
struct_type = "FairPC"
T = STRUCT_STR2TYPE[struct_type]
SV = "S"
fold = 1
num_X = 20
train_x, valid_x, test_x = load_data(name, T, SV; fold=fold, num_X=num_X)
train_x1, valid_x1, test_x1 = convert2nonlatent(train_x, valid_x, test_x)
println("testStub : Working with $(train_x1.header)")
println("testStub : Working with $(train_x1.SV)")
println("testStub : Working with $(train_x1.S)")
println("testStub : Working with $(train_x1.D)")

#read pc

k_list = [20]
println("in testStub : Calling get_query")
k_dfs = get_query(train_x1, k_list)
for df in k_dfs
    println(typeof(df))
    data_mat = Matrix(df[:,1:21])
    println(typeof(data_mat))
    println(data_mat[1,:])
end
println(raw"created dfs for each k in k_list")

