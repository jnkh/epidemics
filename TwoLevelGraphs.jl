module TwoLevelGraphs

using LightGraphs, Distributions

export TwoLevel, is_valid, get_num_infected, distribute_randomly, make_consistent, TwoLevelGraph, get_clusters, make_two_level_random_graph

type TwoLevel
    a::Array{Int,1} #number communities with [idx] infected nodes
    N::Int #total number of nodes
    m::Int #number of nodes per community
    n::Int #number of communities
    i::Int #total number of infecteds
    r::Int #outside connections
    l::Int #internal connections
    
end
    
function TwoLevel(N::Int,m::Int)
    a = zeros(Int,m+1)
    n = Int(N/m)
    l = Int(m/2)
    r = Int(m/2)
    i = 0
    return TwoLevel(a,N,m,n,i,r,l)
end


function TwoLevel(N::Int,m::Int,l::Int,r::Int)
    a = zeros(Int,m+1)
    n = Int(N/m)
    i = 0
    return TwoLevel(a,N,m,n,i,r,l)
end



function is_valid(t::TwoLevel)
    #check normalization
    valid = true
    valid = valid && sum(t.a) == t.n
    valid = valid && sum(t.a .*collect(0:t.m)) == t.i
    return valid
end

function get_num_infected(t::TwoLevel)
    return sum(t.a .* collect(0:t.m))
end

function distribute_randomly(t::TwoLevel,n::Int)
    indeces = rand(1:length(t.a),n)
    for idx in indeces
        t.a[idx] +=1
    end
    t.n = sum(t.a)
    t.i = get_num_infected(t)
end

function make_consistent(t::TwoLevel)
    t.n = sum(t.a)
    t.i = get_num_infected(t)
end

type TwoLevelGraph
    g::LightGraphs.Graph
    t::TwoLevel
    clusters::Array{Array{Int,1},1}
end

function TwoLevelGraph(t::TwoLevel)
    g,clusters = make_two_level_random_graph(t)
    return TwoLevelGraph(g,t,clusters)
end

function get_clusters(t::TwoLevel)
    ##produce clusters
    clusters = [Int[] for _ in 1:t.n]
    node_idx = 1
    for clust = 1:t.n
        for idx = 1:t.m
            push!(clusters[clust],node_idx)
            node_idx += 1
        end
    end
    clusters
end

#This only works in an unbiased way if the subgraphs have the same sizes.
function make_two_level_random_graph(t::TwoLevel)
    g = LightGraphs.Graph(t.N)
    
    clusters = get_clusters(t)
    
    #get intra-cluster edges
    edges = []
    for clust in clusters
        edges = vcat(edges,get_edges_for_subgraph(clust,num_internal_edges(clust,t)))
    end
    
    #get between-cluster edges
    edges = vcat(edges,get_edges_for_supergraph(clusters,num_external_edges(clusters,t)))

    for e in edges
        add_edge!(g,e)
    end
#     end
    return g,clusters
end

function get_edges_for_supergraph(clusters::Array{Array{Int,1},1},num_edges::Int)
    possible_edges = []
    for clust in clusters
        for v in clust
            for w in get_out_nodes(clusters,v)
                if w > v
                    push!(possible_edges,Pair(v,w))
                end
            end
        end
    end
    return sample(possible_edges,num_edges)
end
   
#     possible_edges = []
#     for i = 1:length(clusters)
#         for j = i+1:length(clusters)
#             push!(possible_edges,Pair(i,j))
#         end
#     end
#     super_edges = sample(possible_edges,num_edges)
#     edges = []
#     for se in super_edges
#         e = Pair(sample(clusters[se[1]]),sample(clusters[se[2]]))
#         push!(edges,e)
#     end
#    return edges
#end
    
    
function num_internal_edges(cluster::Array{Int,1},t::TwoLevel)
    num_desired = Int(length(cluster)*t.l/2)
    num_trials = Int(length(cluster)*(length(cluster) - 1)/2)
    total_edges = rand(Binomial(num_trials,num_desired/num_trials))
    #total_edges = num_desired #size of cluster times number of internal edges per node
    return total_edges
end

function num_external_edges(clusters::Array{Array{Int,1},1},t::TwoLevel)
    num_desired = Int(t.N*t.r/2)
    num_trials = Int(t.N*(t.N-1)/2)
    total_edges = rand(Binomial(num_trials,num_desired/num_trials))
    #total_edges = num_desired
    return total_edges
end

function get_edges_for_subgraph(cluster::Array{Int,1},num_edges::Int)
    g = LightGraphs.erdos_renyi(length(cluster),num_edges)
    edges = collect(LightGraphs.edges(g))
    new_edges = copy(edges)
    for (i,e) in enumerate(edges)
        new_edges[i] = Pair(cluster[e[1]],cluster[e[2]])
    end
    return new_edges
end

function get_edges_for_subgraph(cluster::Array{Int,1},p_edges::Float64)
    g = LightGraphs.erdos_renyi(length(cluster),p_edges)
    edges = collect(LightGraphs.edges(g))
    new_edges = copy(edges)
    for (i,e) in enumerate(edges)
        new_edges[i] = Pair(cluster[e[1]],cluster[e[2]])
    end
    return new_edges
end

function get_in_degree(clusters::Array{Array{Int,1},1},node::Int,g::LightGraphs.Graph)
    cluster_idx = get_cluster_idx(clusters,node)
    neighbors = neighbors(g,node)
    in_nodes = get_in_nodes(clusters,node)
    return length(intersect(neighbors,in_nodes))
end

function get_out_degree(clusters::Array{Array{Int,1},1},node::Int,g::LightGraphs.Graph)
    cluster_idx = get_cluster_idx(clusters,node)
    neighbors = neighbors(g,node)
    out_nodes = get_out_nodes(clusters,node)
    return length(intersect(neighbors,out_nodes))
end

function get_cluster_idx(clusters::Array{Array{Int,1},1},node::Int)
    return filter(x -> node in clusters[x],[idx for idx in 1:length(clusters)])[1]
end


        
function get_in_nodes(clusters::Array{Array{Int,1},1},node::Int)
    cluster_idx = get_cluster_idx(clusters,node)   
    to_sample = copy(clusters[cluster_idx])
    self_idx = findfirst(to_sample,node)
    splice!(to_sample,self_idx)
    return to_sample
end

function get_out_nodes(clusters::Array{Array{Int,1},1},node::Int)
    cluster_idx = get_cluster_idx(clusters,node)   
    clusters = copy(clusters)
    splice!(clusters,cluster_idx)
    to_sample = vcat(clusters...)
    return to_sample
end


function sample_out_edges(clusters::Array{Array{Int,1},1},node::Int,num::Int)
    to_sample = get_out_nodes(clusters,node)
    return sample(to_sample,num,replace=false)
end

function sample_in_edges(clusters::Array{Array{Int,1},1},node::Int,num::Int)
    to_sample = get_in_nodes(clusters,node)
    return sample(to_sample,num,replace=false)
end


end