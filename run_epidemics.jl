#Add the number of processors
# Pkg.add("LightGraphs")
# Pkg.add("Cubature")
# Pkg.add("Distributions")
# Pkg.add("JLD")
push!(LOAD_PATH, pwd())
using TwoLevelGraphs
using SlurmNodes
using JLD
addprocs(get_list_of_nodes())
# addprocs(8)
using EpidemicsSimulations
@everywhere using PyCall
@everywhere @pyimport networkx as nx
@everywhere using TwoLevelGraphs,GraphGeneration, GraphCreation
@everywhere using SIS,IM,Epidemics,Distributions

CLUSTERING = 7
TWO_DEGREE = 6
GAMMA = 5
SCALE_FREE = 4
TWO_LEVEL = 3
REGULAR = 2
RANDOM = 1

verbose = false

########## Set up model ###############


c_r = 0.3
N = 400
y_n = 0.1

n_n = Int(N*y_n)#y_n*N
beta = get_beta(N,c_r,n_n)#4.0/(c_r*n_n)
alpha = get_alpha(N,c_r,n_n)#(N*beta)/n_n

k_range = [10]
sigma_k_range = [10]#[0.2,1,3,5,10,15,20]
C_range = [0.1]

####only for two-level graphs####
m = 20 #nodes per subnode
n = Int(N/m)
l = 10#Int(m/2)#internal
r = 10 #Int(m/2)#2 #external
#################################


graph_type_range = [GAMMA]

if verbose println(N, ' ' ,alpha, ' ',beta) end

num_trials = 10_000
num_trials_mixed = num_trials
fixation_threshold = 1.0
###Set to true if we want by-node information on infecteds (much more data!)
carry_by_node_information = false
###Set to false only if we want to simulate in a well-mixedm model
graph_model_range = [true]

in_parallel = true
compact = true

for k in k_range
for sigma_k in sigma_k_range
for C in C_range
for graph_type in graph_type_range
for graph_model in graph_model_range

	@eval @everywhere N = $N
	@eval @everywhere k = $k
	@eval @everywhere sigma_k = $(sigma_k)
    @eval @everywhere C = $C
	graph_data = nothing
	G = 0
	if graph_type == REGULAR
	    graph_fn = () -> LightGraphs.random_regular_graph(N,k)
	elseif graph_type == RANDOM
	    graph_fn = () -> LightGraphs.erdos_renyi(N,1.0*k/(N-1))
	elseif graph_type == SCALE_FREE 
	    graph_fn = () -> LightGraphs.barabasi_albert(N,Int(round(k/2)),Int(round(k/2)))
	elseif graph_type == GAMMA
	    graph_fn = () -> graph_from_gamma_distribution(N,k,sigma_k)
	    graph_data = sigma_k
    elseif graph_type == CLUSTERING
        # @eval @everywhere d = Binomial(k,1)
        # graph_fn = () -> create_graph(N,k,:rand_clust,C,deg_distr=d)
        graph_fn = () -> create_graph(N,k,:watts_strogatz,C)
        graph_data = C
	elseif graph_type == TWO_DEGREE
            graph_fn = () -> graph_from_two_degree_distribution(N,k,sigma_k)
            graph_data = sigma_k
    	elseif graph_type == TWO_LEVEL

	    t = TwoLevel(N,m,l,r)
		@eval @everywhere t = $t
	    graph_data = TwoLevelGraph(LightGraphs.Graph(),t,get_clusters(t))
	    # graph_fn = () -> make_two_level_random_graph(t)[1]
	    graph_fn = () -> generate_regular_two_level_graph(t)
	end

	graph_information = GraphInformation(graph_fn,LightGraphs.Graph(),carry_by_node_information,graph_data,graph_type)

	params = Dict{AbstractString,Any}("N" => N, "alpha" => alpha,
	"beta" => beta, "fixation_threshold" => fixation_threshold,
	"in_parallel" => in_parallel, "num_trials" => num_trials,
	"num_trials_mixed" => num_trials_mixed,
	"graph_information"=>graph_information,"verbose"=>verbose,
	"graph_type"=>graph_type,"compact"=>compact)


    println("k = $k, graph_model = $graph_model, graph_type = $(graph_type)")
    params["k"] = k
    params["graph_model"] = graph_model
    params["graph_type"] = graph_type
    # #share among processors
    # for p in procs()
    #     remotecall_fetch(p,(x,y,z) -> (params["k"] = x; params["graph_model"] = y;params["graph_type"] = z),k,graph_model,graph_type)
    # end

    save_epidemics_results(params)

end
end
end
end
end


