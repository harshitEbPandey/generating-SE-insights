using CSV
using Combinatorics
using DataFrames
using DataStructures
using Dates
using LinearAlgebra
using LogicCircuits
using ProbabilisticCircuits
using Random
using Statistics
using StatsBase: sample

export create_dataframe, get_query, convert_k, prediction_bottom, predict_all_se

function create_dataframe(inst::Array{Bool, 1}, k::Int)
    n = length(inst)
    m = binomial(n, k)
    indices = collect(combinations(1:n, k))
    data = Matrix{Union{Missing, Bool}}(missing, m, n)
    for i in 1:m
        for j in indices[i]
            data[i, j] = inst[j]
        end
    end
    println("dataframe for inst $(inst) created!")
    return DataFrame(data, :auto)
end

function convert_k(data)
    data[!,:x9] = fill(missing, size(data,1))
    data
end

"""
    get_query(test_x1::FairDataset, k_list)
    Samples n_samples from the dataset and creates 
    partial assignment dataframes for given values 
    of k (k_list).
    Input -> Fair Dataset, list of k
    Output -> list of partial assignment dataframes
    for each sampled instance
"""
function get_query(test_x1::FairDataset, k_list)
    n_samples = 10
    D_col = test_x1.D
    SV_col = test_x1.S
    
    """
    # random but equal stratified selection
    note : stratification wont work as priors will 
    not match leading to confusing EP calculations
    
    # one_df = filter(row -> row.x8 == 1, test_x1.data)
    # zero_df = filter(row -> row.x8 == 0, test_x1.data)
    # one_rows_idx = randperm(nrow(one_df))[1:n_samples]
    # zero_rows_idx = randperm(nrow(zero_df))[1:n_samples]
    
    # # select the rows from the DataFrame
    # rows_idx = vcat(one_rows_idx, zero_rows_idx)
    # sample_df = test_x1.data[rows_idx, :]
    """
    sample_rows = sample(1:nrow(test_x1.data), n_samples, replace=false)
    sample_df = test_x1.data[sample_rows, :]

    data_mat = Matrix(sample_df)
    outdir = "/Users/harshit/Documents/GitHub/generating-SE-insights/analysis/data/exp4"
    CSV.write(joinpath(outdir, "sampled_instances_$(n_samples).csv"), sample_df)

    list_of_dfs = Vector{DataFrame}()

    for k in k_list
        res = create_dataframe(data_mat[1,1:21],k)
        res[!,:x22] = fill(data_mat[1,D_col:D_col][1], size(res,1))
        for i in 2:size(data_mat, 1)
            inst = data_mat[i,1:21]
            perms = create_dataframe(inst,k)
            perms[!,:x22] = fill(data_mat[i,D_col:D_col][1], size(perms,1))
            res = vcat(res,perms)
        end
        push!(list_of_dfs, res)
        print("writing $(k)_test.csv to - ")
        println(joinpath(outdir, "$(k)_test.csv"))
        CSV.write(joinpath(outdir, "$(k)_test.csv"), res)
    end
    list_of_dfs
end

function prediction_bottom(fairpc::StructType, fairdata, flag)
    outdir = raw"/Users/harshit/Documents/GitHub/generating-SE-insights/analysis/data/exp4"
    @inline get_node_id(id::⋁NodeIds) = id.node_id
    @inline get_node_id(id::⋀NodeIds) = @assert false
    results = Dict()
    data = fairdata
    data = reset_end_missing(data)

    _, flows, node2id = marginal_flows(fairpc.pc, data)
    if fairpc isa NonLatentStructType
        D = get_node_id(node2id[node_D(fairpc)])
        n_D = get_node_id(node2id[node_not_D(fairpc)])
        P_D = exp.(flows[:, D])
        # @assert all(flows[:, D] .<= 0.0)
        @assert all(P_D .+ exp.(flows[:, n_D]) .≈ 1.0)
        P_D = min.(1.0, P_D)
        results["P(D|e)"] = P_D
    end

    if fairpc isa LatentStructType
        Df = get_node_id(node2id[node_Df(fairpc)])
        n_Df = get_node_id(node2id[node_not_Df(fairpc)])
        P_Df = exp.(flows[:, Df])
        @assert all(P_Df .+ exp.(flows[:, n_Df]) .≈ 1.0)
        # @assert all(flows[:, Df] .<= 0.0)
        P_Df = min.(1.0, P_Df)    
        results["P(Df|e)"] = P_Df
    end
    println("Writing EP values to file")
    CSV.write(joinpath(outdir, "EP_k_$(flag)"), results, bufsize=2^26)
end

function predict_all_se(T, result_circuits, log_opts, train_x, test_x, flag)
    for (key, value) in result_circuits
        dir = joinpath(log_opts["outdir"], key)
        (pc, vtree) = value
        if !isdir(dir)
            mkpath(dir)
        end
        run_fairpc = T(pc, vtree, train_x.S, train_x.D)
        prediction_bottom(run_fairpc,test_x,flag)
    end
end