module DataAnalysis

using SIS,IM,PayloadGraph,PyPlot, Epidemics,JLD, TwoLevelGraphs,Dierckx,Plotting
import LightGraphs

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
        if y_desired-tolerance< t.i/t.N < y_desired+tolerance
            counter += 1
            accum += t.a
        end
    end
    accum /= counter
    println("$counter instances")
    return accum
end

end