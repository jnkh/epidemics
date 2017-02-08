module DataAnalysis

using SIS,IM,PayloadGraph,PyPlot, Epidemics,JLD, TwoLevelGraphs,Dierckx,Plotting
import LightGraphs

export get_infecteds_by_clusters_vs_time,

summarize_p_reach_data,load_p_reach_data,

get_two_level_states_from_runs,get_mean_distribution_from_states,

get_two_level_states



function load_p_reach_data(path)
    d = load(path)

    params = d["params"]
    compact = params["compact"]
    runs = d["runs"]
    if compact
        sizes = runs.sizes
        pvals = runs.p_reach
        yvals = runs.y_reach
        num_trials = runs.num_trials
    else
        num_fixed = get_num_fixed(runs) #d["num_fixed"]
        sizes = get_sizes(runs) #d["sizes"]
        yvals,pvals = get_p_reach(runs,params["N"])
        num_trials = length(runs)
    end
    
    epidemic_params = QuadraticEpidemicParams(params["N"],params["alpha"],params["beta"])
    
    k = params["k"]
    graph_information = params["graph_information"]
    graph_type = params["graph_type"]
    return params,yvals,pvals,epidemic_params,k,graph_information,graph_type,runs,num_trials
end

function summarize_p_reach_data(path)
    params,yvals,pvals,epidemic_params,k,graph_information,graph_type,runs,num_trials = load_p_reach_data(path)
    N = epidemic_params.N
    alpha = epidemic_params.alpha
    beta = epidemic_params.beta
    n_n = epidemic_params.n_n
    c_r = epidemic_params.c_r
 
    println("N = $N, k = $k, y_n = $(n_n/N), c_r = $(c_r)")
    println("alpha = $(alpha), beta = $(beta)")
    println("Graph Type: $(graph_type)")
    if graph_type == Int(two_level_rg)
        t = graph_information.data.t
        println("k_i = $(t.l), k_e = $(t.r)")
    end
    println("num trials: $(num_trials)")
        
end

#figure out which cluster the infecteds are in from the raw data
function get_infecteds_by_clusters_vs_time(clusters::Array{Array{Int,1},1},infecteds_by_nodes_vs_time::Array{Array{Int,1},1})
    infecteds_by_clusters_vs_time::Array{Array{Int,1},1} = []
    num_clusters = length(clusters)
    for infecteds_by_nodes in infecteds_by_nodes_vs_time
        infecteds_by_clusters = zeros(Int,num_clusters)
        for (i,clust) in enumerate(clusters)
            for node in clust
                if infecteds_by_nodes[node] == 1
                    infecteds_by_clusters[i] += 1
                end
            end
        end
        push!(infecteds_by_clusters_vs_time,infecteds_by_clusters)
    end
    return infecteds_by_clusters_vs_time
    
end

#we have infecteds vs. cluster idx and want to know how many clusters of infectivity level i there are for every possible i=1:m
function get_infection_distribution_over_clusters(t::TwoLevel,infecteds_by_clusters)
    assert(t.n == length(infecteds_by_clusters))
    infection_distribution = zeros(Int,t.m+1)
    for (i,num_infected) in enumerate(infecteds_by_clusters)
        infection_distribution[num_infected+1] += 1
    end
    return infection_distribution
end

function get_two_level_with_distribution(t_template::TwoLevel,infection_distribution::Array{Int,1})
    assert(t_template.m + 1 == length(infection_distribution))
    t = TwoLevel(t_template)
    t.a = infection_distribution
    make_consistent(t)
    return t
end

function get_two_level_states(tg::TwoLevelGraph,infecteds_by_nodes_vs_time::Array{Array{Int,1},1})
    two_level_states::Array{TwoLevel,1} = []
    infecteds_by_clusters_vs_time = get_infecteds_by_clusters_vs_time(tg.clusters,infecteds_by_nodes_vs_time)
    for infecteds_by_clusters in infecteds_by_clusters_vs_time
        push!(two_level_states,get_two_level_with_distribution(tg.t,get_infection_distribution_over_clusters(tg.t,infecteds_by_clusters)))
    end
    return two_level_states
end

function get_two_level_states_from_runs(runs::Array{EpidemicRun,1})
    tg_raw = runs[1].graph_information.data
    t_raw = runs[1].graph_information.data.t
    t::TwoLevel = TwoLevel(t_raw.a,t_raw.N,t_raw.m,t_raw.n,t_raw.i,t_raw.r,t_raw.l)
    tg::TwoLevelGraph = TwoLevelGraph(tg_raw.g,t,tg_raw.clusters)
    two_level_states::Array{TwoLevel,1} = []
    for run in runs
        two_level_states = vcat(two_level_states,get_two_level_states(tg,run.infecteds_by_nodes_vs_time))
    end
    return two_level_states
end

function get_mean_distribution_from_states(two_level_states::Array{TwoLevel,1},y_desired::AbstractFloat,tol= 0.005)
    counter = 0
    t = two_level_states[1]
    accum = zeros(length(t.a))
    for t in two_level_states
        if y_desired-tol< t.i/t.N < y_desired+tol
            counter += 1
            accum += t.a
        end
    end
    accum /= counter
    println("$counter instances")
    return accum
end

end