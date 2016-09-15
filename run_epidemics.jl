function save_epidemics_results(params)
	data_dir_path = "../data/"  #"/mnt/D/windows/MIT/classes/6/338/project/data/"

@everywhere begin
	N::Int = params["N"]
	alpha::Float64 = params["alpha"]
	beta::Float64 = params["beta"]
	k = params["k"]
	verbose = params["verbose"]
	fixation_threshold = params["fixation_threshold"]
	in_parallel = params["in_parallel"]
	num_trials = params["num_trials"]
	num_trials_mixed = params["num_trials_mixed"]
	graph_model = params["graph_model"]
  graph_information = params["graph_information"]

	#function f1(x::Float64) 1 + alpha::Float64*x end
	#function f2(x::Float64)  1 + beta::Float64 end
	#function f3(x::Float64) 1 + beta + get_s_eff(x,alpha,beta,k) end
	#im_normal = InfectionModel(f1,f2);
	#im_effective = InfectionModel(f3,f2);


	im_normal = InfectionModel(x -> 1 + alpha*x , x -> 1 + beta);
	im_effective = InfectionModel(x -> 1 + beta + get_s_eff(x,alpha,beta,k) , x -> 1 + beta);



	y_n, y_minus, y_plus, y_p,critical_determinant = get_parameters(N,alpha,beta,verbose)

	end


	println("running in parallel on $(nprocs()-1) nodes...")
	tic()
	if graph_model
	@time runs =
        run_epidemics_parallel(num_trials, () -> run_epidemic_graph(N,im_normal,graph_information,fixation_threshold),in_parallel);
	else
		if k < N-1
			@time runs =
			run_epidemics_parallel(num_trials_mixed, () -> run_epidemic_well_mixed(N,im_effective,fixation_threshold),in_parallel);
		else
			@time runs =
			run_epidemics_parallel(num_trials_mixed, () -> run_epidemic_well_mixed(N,im_normal,fixation_threshold),in_parallel);
		end
	end
	elapsed = toc()
	println("done after $elapsed seconds.")


	structure = graph_model ? "graph" : "mixed"
	filename = "epidemics_$(structure)_$(now()).jld"
	save(data_dir_path * filename,"params",params,"runs",runs)


	timing_filename = "timing_$(structure)_$(now()).jld"
	save(data_dir_path * timing_filename,"params",params,"elapsed",elapsed)

	timing_log_filename = "timing_log.out"
	f = open(timing_log_filename,"a")
	nt = graph_model ? num_trials : num_trials_mixed
	write(f,"$N $k $nt $elapsed $graph_model $(now())\n")

end



#Add the number of processors
# Pkg.add("LightGraphs")
# Pkg.add("Cubature")
# Pkg.add("Distributions")
# Pkg.add("JLD")
using SIS,IM,PayloadGraph, Epidemics,SlurmNodes, TwoLevelGraphs
using JLD

addprocs(get_list_of_nodes())

@everywhere begin

using SIS,IM,PayloadGraph, Epidemics, TwoLevelGraphs

TWO_LEVEL = 3
REGULAR = 2
RANDOM = 1

verbose = false

########## Set up model ###############

#y_n = 0.1
c_r = 0.18
N = 400
n_n = 20#y_n*N
beta = 4.0/(c_r*n_n)
alpha = (N*beta)/n_n
k_range = [4,12]

####only for two-level graphs####
m = 20 #nodes per subnode
n = Int(N/m)
l = Int(m/2)#internal
r = 2# Int(m/2)#2 #external
#################################


graph_type_range = [TWO_LEVEL,REGULAR]

if verbose println(N, ' ' ,alpha, ' ',beta) end

num_trials_mixed = 10_000
num_trials = 10_000
fixation_threshold = 1.0#8*n_n/N
###Set to true if we want by-node information on infecteds (much more data!)
carry_by_node_information = false
###Set to false only if we want to simulate in a well-mixedm model
graph_model_range = [true]

in_parallel = true

end


for k in k_range
for graph_type in graph_type_range
for graph_model in graph_model_range
	graph_data = nothing
	if graph_type == REGULAR
	    graph_fn = () -> LightGraphs.random_regular_graph(N,k)
	elseif graph_type == RANDOM
	    graph_fn = () -> LightGraphs.erdos_renyi(N,1.0*k/(N-1))
	elseif graph_type == TWO_LEVEL


	    t = TwoLevel(N,m,l,r)
	    graph_data = TwoLevelGraph(LightGraphs.Graph(),t,get_clusters(t))
	    graph_fn = () -> make_two_level_random_graph(t)[1]
	end

	graph_information = GraphInformation(graph_fn,LightGraphs.Graph(),carry_by_node_information,graph_data)

	@everywhere begin
	params = Dict{AbstractString,Any}("N" => N, "alpha" => alpha, "beta" => beta, "fixation_threshold" => fixation_threshold,"in_parallel" => in_parallel, "num_trials" => num_trials, "num_trials_mixed" => num_trials_mixed,"graph_information"=>graph_information,"verbose"=>verbose,"graph_type"=>graph_type)
	end


    println("k = $k, graph_model = $graph_model")
    #share among processors
    for p in procs()
        remotecall_fetch(p,(x,y,z) -> (params["k"] = x; params["graph_model"] = y;params["graph_type"] = z),k,graph_model,graph_type)
    end
    save_epidemics_results(params)
end
end
end


