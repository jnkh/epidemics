__precompile__()
module Epidemics

using SIS,Distributions, IM, LightGraphs,PayloadGraph, Dierckx,GraphGeneration,Distributed,QuadGK,Random,SparseArrays
import TwoLevelGraphs,DegreeDistribution

export

RandomGraphType,random_rg,regular_rg,two_level_rg,scale_free_rg,gamma_rg,two_degree_rg,clustering_rg,complete_rg,custom_rg,

run_epidemic_graph,run_epidemic_well_mixed,run_epidemics_parallel,run_epidemics,
run_epidemic_well_mixed_two_level,
EpidemicRun,
prepare_for_saving,

s,get_s_eff,get_s_eff_exact,
normed_distribution, P_w_th,get_y_eff,get_y_eff_exact,
get_c_r,get_n_n,
get_binomial_error,

get_sizes, get_num_fixed,GraphInformation,
get_dt_two_level, update_n_two_level,
get_p_reach, CompactEpidemicRuns, get_n_plus, get_n_minus,
run_epidemic_graph_experimental,
run_epidemic_graph_gillespie,
get_s_eff_degree_distribution,
get_s_eff_degree_distribution_gamma,get_s_eff_degree_distribution_scale_free,
get_p_k_barabasi_albert,get_p_k_gamma,

get_alpha,get_beta,get_c_r,get_n_n,QuadraticEpidemicParams,get_QuadraticEpidemicParams,

get_p_reach_well_mixed_simulation,get_p_reach_well_mixed_two_level_simulation,
get_p_reach,get_p_reach_theory,get_p_reach_sim,SimulationResult,PreachResult,
TheoryResult,get_theory_result,get_graph_information,get_simulation_result,
get_simulation_yy_pp,get_p_reach_and_runs_sim,get_pfix_sim,get_pfix_th,get_pfix,

sparsity_pfix_prediction,
sparsity_pfix_prediction_large_N,
sparsity_cascade_condition,
sparsity_cascade_condition_simple,
sparsity_get_yn,

clean_result,graph_to_mat,mat_to_graph,to_lightgraphs_graph




#Content

@enum RandomGraphType random_rg=1 regular_rg=2 two_level_rg=3 scale_free_rg=4 gamma_rg=5 two_degree_rg=6 clustering_rg=7 complete_rg=8 custom_rg=9

function graph_is_connected(g::LightGraphs.Graph)
    parents = LightGraphs.dijkstra_shortest_paths(g,1).parents[2:end]
    return size(parents) == 0 || minimum(parents) > 0
end

function guarantee_connected(graph_fn)
    g = graph_fn()
    resampled = 0
    # while(!graph_is_connected(g))
    while(!LightGraphs.is_connected(g))
        g = graph_fn()
        resampled += 1
    end
    if resampled > 0
        println("Resampled Graph $resampled times.")
    end
    return g
end

mutable struct GraphInformation
    graph_fn
    graph
    carry_by_node_info::Bool
    carry_temporal_info::Bool
    pregenerate_graph::Bool
    data
    graph_type::RandomGraphType
end

function GraphInformation(graph_fn,g,carry_by_node_info::Bool,data,graph_type::RandomGraphType)
    return GraphInformation(graph_fn,g,carry_by_node_info,true,false,data,graph_type)
end

function GraphInformation()
    return GraphInformation(x -> x,LightGraphs.Graph(),false,true,false,nothing,random_rg)
end

mutable struct EpidemicRun
    times::Array{Float64,1}
    infecteds_vs_time::Array{Float64,1}
    size::Float64
    max_reach::Float64
    fixed::Bool
    infecteds_by_nodes_vs_time::Array{Array{Int,1},1}
    graph_information::GraphInformation
end

struct CompactEpidemicRuns
    sizes::Array{Float64,1}
    y_reach::Array{Float64,1}
    p_reach::Array{Float64,1}
    num_trials::Int
end

function CompactEpidemicRuns(runs::Array{EpidemicRun,1},N::Int)
    sizes = get_sizes(runs)
    y_reach,p_reach = get_p_reach(runs,N)
    return CompactEpidemicRuns(sizes,y_reach,p_reach,length(runs))
end

function set_max_reach(er::EpidemicRun)
    er.max_reach = get_max_reach(er)
end

function EpidemicRun(infecteds_vs_time::Array{Float64,1},size::Float64,max_reach::Float64,fixed::Bool)
    times = zeros(Float64,length(infecteds_vs_time))
    er = EpidemicRun(times,infecteds_vs_time,size,max_reach,fixed,[],GraphInformation())
    return er
end

function EpidemicRun(infecteds_vs_time::Array{Float64,1},size::Float64,fixed::Bool)
    times = zeros(Float64,length(infecteds_vs_time))
    er = EpidemicRun(times,infecteds_vs_time,size,0.0,fixed,[],GraphInformation())
    set_max_reach(er)
    return er
end

function EpidemicRun(times::Array{Float64,1},infecteds_vs_time::Array{Float64,1},size::Float64,fixed::Bool)
    er =  EpidemicRun(times,infecteds_vs_time,size,0.0,fixed,[],GraphInformation())
    set_max_reach(er)
    return EpidemicRun
end


function get_sizes(runs::Array{EpidemicRun,1})
    return filter(x -> x < Inf,[e.size for e in runs])
end

function get_num_fixed(runs::Array{EpidemicRun,1})
    sum = 0
    for r in runs
        if r.fixed
            sum += 1
        end
    end
    return sum
end

function get_max_reach(run::EpidemicRun)
    return maximum(run.infecteds_vs_time)
end

function get_max_reaches(runs::Array{EpidemicRun, 1})
    return [er.max_reach for er in runs]
end

function get_p_reach(runs::Array{EpidemicRun, 1})
    max_reaches = get_max_reaches(runs)
    sorted_max_reaches = sort(max_reaches)
    xvals = unique(sorted_max_reaches)
    yvals = 0*similar(xvals)
    for r in sorted_max_reaches
        idx = findfirst(x -> x == r, xvals)
        yvals[1:idx] .+= 1
    end

    return xvals,yvals/length(max_reaches)
end


function get_p_reach(runs::Array{EpidemicRun, 1},N::Real)
    xvals,yvals = get_p_reach(runs)
    return xvals/N,yvals
end

function get_binomial_error(p_est,n_samples)
    return sqrt.(p_est.*(1.0.-p_est)./n_samples)
end

###
function get_simulation_yy_pp(gi::GraphInformation,N,alpha,beta,num_trials_th=100;in_parallel=false,num_trials_sim=1000,use_theory=true,verbose=false)
    if use_theory
        tr = get_theory_result(N,alpha,beta,gi,num_trials_th)
        yy = tr.pr.yy
        pp = tr.pr.pp
    else
        sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel)
        yy = sr.prsim.yy
        pp = sr.prsim.pp
    end
    pfix,success = get_pfix_from_yy_pp(yy,pp,verbose)
    return yy,pp,pfix,success
end

function get_pfix_from_yy_pp(yy,pp,verbose=false)
    pfix = pp[end]
    success = true
    if yy[end] < 0.99
        if verbose
            println("simulation didn't reach 1.0")
            println("theory suggests at least $(1/pfix)")
        end
        success = false
        pfix = 0
    end
    return pfix,success
end



#get graph information for the different types of graphs
function get_graph_information(graph_type::RandomGraphType;N=400,k = 10,sigma_k = 10,m=20,l=19,r=1,custom_g = nothing,carry_by_node_information=false,carry_temporal_info=true,pregenerate_graph=false)
    graph_fn = nothing
    graph_data = nothing
    G = 0
    if graph_type == regular_rg
        graph_data = k
        graph_fn = () -> LightGraphs.random_regular_graph(N,k)
    elseif graph_type == random_rg
        graph_fn = () -> LightGraphs.erdos_renyi(N,1.0*k/(N-1))
    elseif graph_type == scale_free_rg
        graph_fn = () -> LightGraphs.barabasi_albert(N,Int(round(k/2)),Int(round(k/2)))
    elseif graph_type == gamma_rg
        graph_fn = () -> graph_from_gamma_distribution(N,k,sigma_k)
        graph_data = (k,sigma_k)
    elseif graph_type == clustering_rg
        # @eval @everywhere d = Binomial(k,1)
        # graph_fn = () -> create_graph(N,k,:rand_clust,C,deg_distr=d)
        graph_fn = () -> create_graph(N,k,:watts_strogatz,C)
        graph_data = C
    elseif graph_type == two_degree_rg
            graph_fn = () -> graph_from_two_degree_distribution(N,k,sigma_k)
            graph_data = (k,sigma_k)
    elseif graph_type == two_level_rg
        t = TwoLevelGraphs.TwoLevel(N,m,l,r)
        # @eval @everywhere t = $t
        graph_data = TwoLevelGraphs.TwoLevelGraph(LightGraphs.Graph(),t,TwoLevelGraphs.get_clusters(t))
        # graph_fn = () -> make_two_level_random_graph(t)[1]
        graph_fn = () -> TwoLevelGraphs.generate_regular_two_level_graph(t)
    elseif graph_type == complete_rg
        graph_fn = () -> LightGraphs.CompleteGraph(N)
    elseif graph_type == custom_rg
        graph_fn = () -> custom_g
    else
        println("Unkown Graph Type, returning Void GraphInformation")
    end
    this_graph = LightGraphs.Graph()
    if pregenerate_graph
        this_graph = graph_to_mat(graph_fn())
    end

    graph_information = GraphInformation(graph_fn,this_graph,carry_by_node_information,carry_temporal_info,pregenerate_graph,graph_data,graph_type)
    return graph_information
end

function get_p_reach_sim(N,alpha,beta,num_trials,graph_information;in_parallel=false,fixation_threshold=1.0)
    im_normal = InfectionModelLinear(alpha,beta);
    runssim = run_epidemics_parallel(num_trials,() -> run_epidemic_graph_gillespie(N,im_normal,graph_information,fixation_threshold),in_parallel);
    yy,pp =  get_p_reach(runssim,N)
    return PreachResult(yy,pp,num_trials)
end

function get_p_reach_and_runs_sim(N,alpha,beta,num_trials,graph_information;in_parallel=false,fixation_threshold=1.0)
    im_normal = InfectionModelLinear(alpha,beta);
    runssim = run_epidemics_parallel(num_trials,() -> run_epidemic_graph_gillespie(N,im_normal,graph_information,fixation_threshold),in_parallel);
    yy,pp =  get_p_reach(runssim,N)
    pfix,_ = get_pfix_from_yy_pp(yy,pp)
    return PreachResult(yy,pp,num_trials),pfix,runssim
end

function get_p_reach_gamma_theory(N,alpha,beta,sigma_k,k,num_trials,graph_information,hypergeometric=true)
    min_degree = 3
    degr_distr = get_p_k_gamma(sigma_k,k,min_degree)
    p_k,p_k_neighbor,mean_k = DegreeDistribution.get_p_k_as_vec(degr_distr,N);
    graph_fn = graph_information.graph_fn
    #p_k_exp,p_k_neighbor_exp = DegreeDistribution.create_p_k_p_k_neighbor_from_graph_fn(graph_fn,100)
    # yy_wm,pp_wm,_ = DegreeDistribution.get_p_reach_well_mixed_by_degree_simulation(N,alpha,beta,p_k,p_k_neighbor,num_trials,hypergeometric)
    yy_wm,pp_wm,_ = DegreeDistribution.get_p_reach_well_mixed_by_degree_simulation_batch(N,alpha,beta,p_k,p_k_neighbor,num_trials,hypergeometric=hypergeometric)
    return yy_wm,pp_wm
end


function get_p_reach_theory(N,alpha,beta,graph_information,num_trials)
    yy = 10.0.^range(log10(1/N),stop=0,length=1000)
    pp = 0*similar(yy)
    graph_type = graph_information.graph_type
    if graph_type == two_level_rg
        has_two_level = true
        apply_finite_size = true
        num_points = num_trials
        t = graph_information.data.t
        yy,pp,s_eff_two_level = TwoLevelGraphs.get_p_reach_theory(t,alpha,beta,N,apply_finite_size,num_points)
    elseif graph_type == scale_free_rg 
        im = InfectionModel(x -> 1 + beta + get_s_eff_degree_distribution_scale_free(x,alpha,beta,k,N) , x -> 1 + beta)
        pp = P_reach_fast(im,N,1.0/N,yy,true)
    elseif graph_type == gamma_rg
        k,sigma_k = graph_information.data
        yy,pp = get_p_reach_gamma_theory(N,alpha,beta,sigma_k,k,num_trials,graph_information)
    elseif graph_type == two_degree_rg
        k,sigma_k = graph_information.data
        tdp = compute_two_degree_params(k,sigma_k)
#             yy_wm,pp_wm,_ = get_p_reach_well_mixed_two_degree_simulation(alpha,beta,N,tdp,num_trials)
        degr_distr = get_p_k_two_degree(tdp)
        p_k,p_k_neighbor,mean_k = get_p_k_as_vec(degr_distr,N);
        yy,pp,_ = get_p_reach_well_mixed_by_degree_simulation(N,alpha,beta,p_k,p_k_neighbor,num_trials_wm,hypergeometric)
    elseif graph_type == regular_rg 
        k = graph_information.data
        im = InfectionModel(x -> 1 + beta + get_s_eff_exact(x,alpha,beta,k,N) , x -> 1 + beta)
        pp = P_reach_fast(im,N,1.0/N,yy,true)
    elseif graph_type == clustering_rg 
        C = graph_information.data
        yy,pp,edge_counts = GraphClustering.get_p_reach_well_mixed_with_clustering(N,k,C,alpha,beta,num_trials_wm,1-1/N);
#         t =get_optimal_tl_params(N,k,C)
#         yy,pp,s_eff_two_level = get_p_reach_theory(t,alpha,beta,N,apply_finite_size,num_points)
    elseif graph_type == complete_rg
        im = InfectionModel(x -> 1 + beta + s(x,alpha,beta) , x -> 1 + beta)
        pp = P_reach_fast(im,N,1.0/N,yy,true)
    end
    return PreachResult(yy,pp,num_trials)
end


struct PreachResult
    yy::Array{Float64,1}
    pp::Array{Float64,1}
    num_trials::Int
end



struct SimulationResult
    prsim::PreachResult
    prth::PreachResult
    N::Int
    alpha::Float64
    beta::Float64
    graph_information::GraphInformation
end

struct TheoryResult
    pr::PreachResult
    N::Int
    alpha::Float64
    beta::Float64
    graph_information::GraphInformation
end

function clean_result(sr::SimulationResult)
    sr.graph_information.graph=nothing
    sr.graph_information.graph_fn=nothing
    sr.graph_information.data=nothing
end

function clean_result(tr::TheoryResult)
    tr.graph_information.graph=nothing
    tr.graph_information.graph_fn=nothing
    tr.graph_information.data=nothing
end

function get_pfix_sim(sr::SimulationResult)
    return get_pfix_from_yy_pp(sr.prsim.yy,sr.prsim.pp)
end

function get_pfix_th(sr::SimulationResult)
    return get_pfix_from_yy_pp(sr.prth.yy,sr.prth.pp)
end

function get_pfix(tr::TheoryResult)
    return get_pfix_from_yy_pp(tr.pr.yy,tr.pr.pp)
end

function SimulationResult(yysim,ppsim,num_trials_sim,yyth,ppth,num_trials_th,N,alpha,beta,gi)
    prsim = PreachResult(yysim,ppsim,num_trials_sim)
    prth = PreachResult(yyth,ppth,num_trials_th)
    return SimulationResult(prsim,prth,N,alpha,beta,gi)
end





function TheoryResult(yyth,ppth,num_trials_th,N,alpha,beta,gi)
    pr = PreachResult(yysim,ppsim,num_trials_sim)
    return TheoryResult(pr,N,alpha,beta,gi)
end

function get_theory_result(N,alpha,beta,gi,num_trials_th;in_parallel=false,fixation_threshold=1.0)
    pr = get_p_reach_theory(N,alpha,beta,gi,num_trials_th);
    return TheoryResult(pr,N,alpha,beta,gi)
end

function get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim;in_parallel=false,fixation_threshold=1.0)
    prsim = get_p_reach_sim(N,alpha,beta,num_trials_sim,gi,in_parallel=in_parallel,fixation_threshold=fixation_threshold)
    prth = get_p_reach_theory(N,alpha,beta,gi,num_trials_th);
    return SimulationResult(prsim,prth,N,alpha,beta,gi)
end

function prepare_for_saving(gi::GraphInformation)
    gi.graph_fn = nothing
end

function prepare_for_saving(si::SimulationResult)
    prepare_for_saving(si.graph_information)
end


###########################
### Epidemic on a Graph ###
###########################



function run_epidemic_graph_experimental(N::Int,im::InfectionModel,graph_information::GraphInformation,fixation_threshold=1.0)
    fixed=false
    shuffle_nodes = false
    #construct graph
    g = guarantee_connected(graph_information.graph_fn)
#     graph_information.graph = g
    carry_by_node_info::Bool = graph_information.carry_by_node_info

    #create payload graph
    p = create_graph_from_value(g,SUSCEPTIBLE)
    infecteds::Array{Float64,1} = []
    infecteds_by_nodes::Array{Array{Int,1},1} = []


    set_payload(p,rand(1:length(get_payload(p))),INFECTED)
    frac = get_fraction_of_type(p,INFECTED)
    push!(infecteds,N*frac)
    if carry_by_node_info
        graph_information.graph = g
        push!(infecteds_by_nodes,copy(get_payload(p)))
    end

    new_types = fill(SUSCEPTIBLE,N)# convert(SharedArray,fill(SUSCEPTIBLE,N))

    while frac > 0
        if !(frac < 1 && frac < fixation_threshold)
            fixed = true
            break
        end
        update_graph_experimental(p,im,new_types)
        frac = get_fraction_of_type(p,INFECTED)
        push!(infecteds,N*frac)
        if carry_by_node_info
            push!(infecteds_by_nodes,copy(get_payload(p)))
        end
        if shuffle_nodes
            if graph_information.graph_type != two_level_rg
                shuffle_payload(p)
            else
                shuffle_payload_by_cluster(p,graph_information.data.clusters)
            end
        end
    end

    size = im.dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,0.0,fixed,infecteds_by_nodes,graph_information)
end

function graph_to_mat(g::LightGraphs.Graph)
    n = LightGraphs.nv(g)
    # mat = spzeros(Int64,n,n)
    m = LightGraphs.ne(g)
    I = zeros(Int64,m)
    J = zeros(Int64,m)
    V = ones(Int64,m)
    for (i,e) in enumerate(edges(g))
        I[i] = e.dst
        J[i] = e.src
    end
    return sparse(I,J,V)
end

function mat_to_graph(mat::SparseMatrixCSC)
    n = size(mat)[1]
    g = LightGraphs.Graph(n)
    for e in findall(!iszero,mat)
        LightGraphs.add_edge!(g,e[1],e[2])
    end
    return g
end

to_lightgraphs_graph(m::SparseMatrixCSC) = mat_to_graph(m)
to_lightgraphs_graph(g::LightGraphs.Graph) = g

#gillespie version
function run_epidemic_graph_gillespie(N::Int,im::Union{InfectionModel,InfectionModelLinear},graph_information::GraphInformation,fixation_threshold=1.0)
    fixed=false
    shuffle_nodes = false
    #construct graph
    if graph_information.pregenerate_graph
        g = to_lightgraphs_graph(graph_information.graph)
        @assert(graph_is_connected(g))
    else
        g = guarantee_connected(graph_information.graph_fn)
    end

    @assert(N == LightGraphs.nv(g))
#     graph_information.graph = g
    carry_by_node_info::Bool = graph_information.carry_by_node_info
    carry_temporal_info::Bool = graph_information.carry_temporal_info

    #create payload graph
    infecteds::Array{Float64,1} = []
    times::Array{Float64,1} = []
    infecteds_by_nodes::Array{Array{Int,1},1} = []
    max_reach = 1.0
    rates = RateArray(fill(0.0,nv(g)))
    neighbor_numbers = fill(0,nv(g))
    payload = fill(SUSCEPTIBLE,nv(g))


    payload[rand(1:length(payload))] = INFECTED
    compute_all_neighbor_numbers_of_type(g,payload,neighbor_numbers,INFECTED)
    t = 0
    num_infected = 1
    frac = 1.0*num_infected/N
    if carry_temporal_info
        push!(infecteds,num_infected)
        push!(times,t)
        if carry_by_node_info
            graph_information.graph = g
            push!(infecteds_by_nodes,copy(payload))
        end
    end

    compute_all_rates(g,payload,im,rates,neighbor_numbers) 

    while frac > 0
        if !(frac < 1 && frac < fixation_threshold)
            fixed = true
            break
        end

        dt,num_infected = update_graph_gillespie(g,payload,im,rates,neighbor_numbers,num_infected)
        t += dt
        frac = 1.0*num_infected/N
        if num_infected > max_reach
            max_reach = num_infected
        end
        if carry_temporal_info
            push!(infecteds,num_infected)
            push!(times,t)
            if carry_by_node_info
                push!(infecteds_by_nodes,copy(payload))
            end
        end
        if shuffle_nodes
            if graph_information.graph_type != two_level_rg
                shuffle!(payload)
                compute_all_neighbor_numbers_of_type(g,payload,neighbor_numbers,INFECTED)
                compute_all_rates(g,payload,im,rates,neighbor_numbers) 
            else
                #shuffle payload by cluster
                clusters = graph_information.data.clusters
                for cluster in clusters
                    payload[cluster] = shuffle(payload[cluster])
                end
            end
        end
    end

    size = im.dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(times,infecteds,size,max_reach,fixed,infecteds_by_nodes,graph_information)
end


### Epidemic on a Graph ###
function run_epidemic_graph(N::Int,im::Union{InfectionModel,InfectionModelLinear},graph_information::GraphInformation,fixation_threshold=1.0)
    fixed=false
    shuffle_nodes = false
    #construct graph
    g = guarantee_connected(graph_information.graph_fn)
#     graph_information.graph = g
    carry_by_node_info::Bool = graph_information.carry_by_node_info

    #create payload graph
    p = create_graph_from_value(g,SUSCEPTIBLE)
    infecteds::Array{Float64,1} = []
    infecteds_by_nodes::Array{Array{Int,1},1} = []


    set_payload(p,rand(1:length(get_payload(p))),INFECTED)
    frac = get_fraction_of_type(p,INFECTED)
    push!(infecteds,N*frac)
    max_reach = 1.0
    if carry_by_node_info
        graph_information.graph = g
        push!(infecteds_by_nodes,copy(get_payload(p)))
    end

    new_types = fill(SUSCEPTIBLE,N)# convert(SharedArray,fill(SUSCEPTIBLE,N))

    while frac > 0
        if !(frac < 1 && frac < fixation_threshold)
            fixed = true
            break
        end
        update_graph(p,im,new_types)
        frac = get_fraction_of_type(p,INFECTED)
        if num_infected > max_reach
            max_reach = num_infected
        end
        push!(infecteds,N*frac)
        if carry_by_node_info
            push!(infecteds_by_nodes,copy(get_payload(p)))
        end
        if shuffle_nodes
            if graph_information.data == nothing
                shuffle_payload(p)
            else
                shuffle_payload_by_cluster(p,graph_information.data.clusters)
            end
        end
    end

    size = im.dt*sum(infecteds)
    if fixed
        size = Inf
    end


    return EpidemicRun(im.dt*collect(1:length(infecteds)),infecteds,size,max_reach,fixed,infecteds_by_nodes,graph_information)
end


### Well Mixed Case ###
function run_epidemic_well_mixed(N,im,fixation_threshold=1.0)
    infecteds::Array{Float64,1} = []
    n = 1
    fixed=false
    push!(infecteds,n)


    while n > 0
        if !(n < N && n < N*fixation_threshold)
            fixed = true
            break
        end
        n = update_n(n,N,im)
        push!(infecteds,n)
    end

    size = im.dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,fixed)
end

function update_n(n::Int,N::Int,im::Union{InfectionModel,InfectionModelLinear})
    y = n/N
    delta_n_plus = rand(Binomial(N-n,y*p_birth(im,y)))
    delta_n_minus = rand(Binomial(n,(1-y)*p_death(im,y)))
    return n + delta_n_plus - delta_n_minus
end



### Well Mixed Case Two Level###
### Well Mixed Case Two Level###
function run_epidemic_well_mixed_two_level(dt::AbstractFloat,N::Int,p_birth_fn,p_death_fn,fixation_threshold=1.0)
    infecteds::Array{Float64,1} = []
    n = 1
    fixed=false
    push!(infecteds,n)

    while n > 0
        if !(n < N && n < N*fixation_threshold)
            fixed = true
            break
        end
        n = update_n_two_level(dt,n,N,p_birth_fn,p_death_fn)
        push!(infecteds,n)
    end

    size = dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,fixed)
end

function get_dt_two_level(alpha::AbstractFloat,beta::AbstractFloat)
    desired_rate = 0.01
    max_rate = maximum([(1 + alpha),(1 + beta)])
    dt = desired_rate/max_rate
    return float(dt)
end

function update_n_two_level(dt::AbstractFloat,n::Int,N::Int,p_birth_interp,p_death_interp)
    y = n/N
    p_birth = y*evaluate(p_birth_interp,y)*dt
    p_death = (1-y)*evaluate(p_death_interp,y)*dt

    if p_birth <= 0
        delta_n_plus = 0
    else
       delta_n_plus = rand(Binomial(N-n,p_birth))
    end

    if p_death <= 0
        delta_n_minus = 0
    else
        delta_n_minus = rand(Binomial(n,p_death))
    end

    return n + delta_n_plus - delta_n_minus

end




###performing many runs###

function run_epidemics(num_runs::Int,run_epidemic_fn)
    runs = EpidemicRun[]

    for i in 1:num_runs
        run = run_epidemic_fn()
        push!(runs,run)
    end
    #get rid of fixed oneus

    return runs
end


function run_epidemics_parallel(num_runs::Int,run_epidemic_fn,parallel=true)
    # mapfn = parallel ? pmap : map
    if parallel
        ret = pmap( _ -> run_epidemic_fn() ,1:num_runs,batch_size=1000)
    else
        ret = map( _ -> run_epidemic_fn() ,1:num_runs)
    end
    for val in ret
        if isa(val,RemoteException)
            print(val)
            throw(val)
        end
    end
    #runs = ret
    runs::Array{EpidemicRun,1} = ret
    return runs
end


function get_p_reach_well_mixed_simulation(im,N,num_runs=10000)
    runs = run_epidemics(num_runs, () -> run_epidemic_well_mixed(N,im,1.0));
    yy,pp = get_p_reach(runs,N)
end

function get_p_reach_well_mixed_two_level_simulation(t,alpha,beta,N,num_runs=10000;num_points=200)
    _,_,_,_,s_birth,s_death,s,splus = get_interpolations(t,alpha,beta,true,num_points)
    dt = get_dt_two_level(alpha,beta)
    run_epidemic_fn = () -> run_epidemic_well_mixed_two_level(dt,N,s_birth,s_death,1.0)

    runs_two_level = run_epidemics(num_runs,run_epidemic_fn);
    yvals,pvals = get_p_reach(runs_two_level,N)
    return yvals,pvals
end

########Two Degree##########

########### Utility Functions ##################


function unzip(zipped::Array{Tuple{A,B},1}) where {A,B}
    l = length(zipped)
    a = Array{A,1}(l)
    b = Array{B,1}(l)
    for (i,(a_el,b_el)) in enumerate(zipped)
        a[i] = a_el
        b[i] = b_el
    end
    return a,b
end

function unzip(zipped)
    l = length(zipped)
    a = []
    b = []
    for (i,(a_el,b_el)) in enumerate(zipped)
        push!(a,a_el)
        push!(b,b_el)
    end
    return a,b
end

#theory for sparse networks

function sparsity_pfix_prediction_large_N(alpha,beta,k) #only works if yn is negative, i.e. only positive selection
    yn = sparsity_get_yn(alpha,beta,k)
    return alpha*abs(yn)
end


function sparsity_pfix_prediction(N,alpha,beta,k)
    yn = (beta/alpha - 1/k)
    if yn < 0
        return 1/N*(yn + (yn^2 + 2.0/(alpha*N))^0.5)^(-1.0)
    else #only valid if cascade_condition is < 1, i.e. in neutral selection regime
        return (N*yn + (2.0*N/alpha)^0.5)^(-1.0) 
    end
end

   
function sparsity_cascade_condition(N,alpha,beta,k) #if this is greater than 1, cascades cannot occur because we have a regime of negative selection
    yn = sparsity_get_yn(alpha,beta,k)
    if beta/alpha - 1/k > 0
        return N*0.5*alpha*yn^2
#         return N*(beta/k - 0.5*(alpha/k^2 + beta^2/alpha))
    else
        return 0
    end
end

function sparsity_cascade_condition_simple(alpha,beta,k) #if this is less than 0, scaling is at worst 1/sqrt(N) or better
    return sparsity_get_yn(alpha,beta,k)
end

function sparsity_get_yn(alpha,beta,k)
    return beta/alpha - 1/k
end
 
    



function normed_distribution(x,px)
    return px./sum(diff(x).*px[1:end-1])
end

P_w_th(w,s) = exp(-s.^2*w./4).*w.^(-1.5)./(2*sqrt(pi).*(1 .+ s))

function s(y,alpha::Float64,beta::Float64)
    return alpha*y - beta
end

function get_y_eff(y,k::Int)
    return y.*(1 + (1-y)./(y*k))
end

function get_s_eff(y,alpha::Float64,beta::Float64,k::Int)
    return alpha*get_y_eff(y,k) - beta
end

function get_y_eff_exact(y,k::Real,N::Int)
    return (y + ((1-y)*(N-k))./(k*N))
end

function get_s_eff_exact(y,alpha,beta,k,N)
    return alpha*get_y_eff_exact(y,k,N) - beta
    # eps = 1e-6 #for numerical stability
    # delta_plus = N/(N-1)
    # delta_minus = 1 + 1./(eps + N.*(1-y))
    # y_plus = y.*delta_plus
    # return delta_plus - delta_minus + alpha*delta_plus*(y_plus + (1-y_plus)/k) - beta*delta_minus
end

function get_s_eff_degree_distribution(y,alpha,beta,p_k::Function,N::Int)
    s_eff_tot = 0
    for k = 1:N-1
        if k == 1
            s_eff_tot = p_k(k)*get_s_eff_exact(y,alpha,beta,k,N)
        else
            s_eff_tot += p_k(k)*get_s_eff_exact(y,alpha,beta,k,N)
        end
    end
    return s_eff_tot
end

function get_s_eff_degree_distribution_scale_free(y,alpha,beta,k::Int,N::Int)
    p_k = get_p_k_barabasi_albert(k)
    return get_s_eff_degree_distribution(y,alpha,beta,p_k,N)
end

function get_s_eff_degree_distribution_gamma(y,alpha,beta,k::Real,sigma_k::Real,N::Int,min_degree=3)
    p_k = get_p_k_gamma(sigma_k,k,min_degree)
    return get_s_eff_degree_distribution(y,alpha,beta,p_k,N)
end

function get_p_k_gamma(sigma_k,k,min_degree=1)
    k,alpha = get_gamma_params(k,sigma_k)
    d = Gamma(k,alpha)
    function p_k(x)
        pdf_fn(y) = pdf(d,y)
        if x < min_degree - 0.5
            return 0
        elseif x < min_degree + 0.5
            return quadgk(pdf_fn,0,min_degree+0.5)[1]
        else
            return quadgk(pdf_fn,x-0.5,x+0.5)[1]
        end
    end
    return p_k
end

function get_p_k_barabasi_albert(k)
    m = Int(round(k/2))
    normalizer = zeta(3) - sum((1.0*collect(1:m-1)).^(-3))
    function p_k(x)
        if x < m return 0 end
        return (1.0*x)^(-3)/normalizer
    end
    return p_k
end


function get_n_plus(y,alpha,beta,k,N)
    eps = 1e-6 #for numerical stability
    delta_plus = N/(N-1)
    y_plus = y.*delta_plus
    return delta_plus*(1 + alpha*(y_plus + (1-y_plus)/k))
end

function get_n_minus(y,alpha,beta,k,N)
    eps = 1e-6 #for numerical stability
    delta_minus = 1 + 1.0/(eps + N.*(1-y))
    return delta_minus.*(1 + beta)
end

struct QuadraticEpidemicParams
    N::Int
    alpha::Float64
    beta::Float64
    c_r::Float64
    n_n::Float64
end

function QuadraticEpidemicParams(N,alpha,beta)
    return QuadraticEpidemicParams(N,alpha,beta,get_c_r(N,alpha,beta),get_n_n(N,alpha,beta))
end

function get_QuadraticEpidemicParams(N,c_r,n_n)
    return QuadraticEpidemicParams(N,get_alpha(N,c_r,n_n),get_beta(N,c_r,n_n),c_r,n_n)
end

function get_c_r(N,alpha,beta)
    return 4*alpha/(beta^2*N)
end

function get_n_n(N,alpha,beta)
    return beta/alpha*N
end

function get_beta(N,c_r,n_n)
    return 4.0/(c_r*n_n)
end

function get_alpha(N,c_r,n_n)
    return (N*get_beta(N,c_r,n_n))/n_n
end


end
