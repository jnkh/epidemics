module GraphCreation

using PyCall, LightGraphs, Distributions
using RandomClusteringGraph

export create_graph, networkx_to_lightgraph, read_edgelist, read_edgelist_julia

function create_graph(N,k,graph_type=:erdos_renyi,clustering=0.6;deg_distr=nothing)
    p_edge = k/(N-1)
    if graph_type == :erdos_renyi
        g = LightGraphs.erdos_renyi(N,p_edge)
    elseif graph_type == :watts_strogatz
        g = watts_strogatz_with_custering(N,Int(round(k)),clustering)
    elseif graph_type == :random_regular
        g = LightGraphs.random_regular_graph(N,Int(round(k)))
    elseif graph_type == :powerlaw_cluster
        g = powerlaw_cluster_graph(N,Int(round(0.5*(N-1)*p_edge)),1.0)
    elseif graph_type == :fb
        g = read_edgelist("../data/graphs/facebook_combined.txt")
    elseif graph_type == :fb_gamma
        g = read_edgelist("../data/graphs/fb_to_gamma_fit_clustering.dat")
    elseif graph_type == :fb_normal
        g = read_edgelist("../data/graphs/fb_to_normal_fit_clustering.dat")
    elseif graph_type == :fb_bter
        g = read_edgelist("../data/graphs/bter_fit_facebook.dat")
    elseif graph_type == :gamma_fb
        g = random_clustering_graph(Gamma(1.0,k),N,clustering)
    elseif graph_type == :rand_clust
        if deg_distr == nothing
            deg_distr = Gamma(1.0,k)
        end
        g = random_clustering_graph(deg_distr,N,clustering)
    else
        error("invalid graph type")
    end
    g
end


function watts_strogatz_get_clustering(N,K,beta)
    c1,c0 = watts_strogatz_clustering_limits(N,K)
    return c1 + (c0-c1)*(1-beta)^3
end

function watts_strogatz_get_beta(N,K,C)
    c1,c0 = watts_strogatz_clustering_limits(N,K)
    beta = 1 - ((C - c1)/(c0 - c1))^(1.0/3)
    return beta
end

function watts_strogatz_clustering_limits(N,K)
    c1 = K/N
    c0 = 3*(K-2)/(4*(K-1))
    return c1,c0
end

function watts_strogatz_with_custering(N,K,C)
    c1,c0 = watts_strogatz_clustering_limits(N,K)
    C = clamp(C,c1,c0)
    beta = watts_strogatz_get_beta(N,K,C)
    return LightGraphs.watts_strogatz(N,K,beta)
end

@pyimport networkx as nx
function networkx_to_lightgraph(G)
    g = LightGraphs.Graph(length(nx.nodes(G)))
    nx_nodes = convert(Array{Int,1},nx.nodes(G))
    nx_to_lg = get_mapping(nx_nodes)
    for e in nx.edges(G)
        LightGraphs.add_edge!(g,nx_to_lg[e[1]],nx_to_lg[e[2]])
    end
    g
end


function get_mapping(nx_nodes::Array{Int,1})
    nx_to_lg = Dict{Int,Int}()
    for (lg_node,nx_node) in enumerate(nx_nodes)
        nx_to_lg[nx_node] = lg_node
    end
    return nx_to_lg
end




function powerlaw_cluster_graph(N::Int,k::Int,beta::Real)
    G = nx.powerlaw_cluster_graph(N,k,beta)
    return networkx_to_lightgraph(G)
end


function make_networkx_graph_from_lightgraph(G::LightGraphs.Graph)
    H = nx.empty_graph(length(LightGraphs.vertices(G)))
    for e in LightGraphs.edges(G)
        H[:add_edge](e[1]-1,e[2]-1)
    end
    H
end

function read_edgelist(filename)
    curr_edges = readdlm(filename,Int)
    curr_nodes = unique(curr_edges)
    g = LightGraphs.Graph(length(curr_nodes))
    to_lg = get_mapping(curr_nodes)
    for i = 1:size(curr_edges,1)
        LightGraphs.add_edge!(g,to_lg[curr_edges[i,1]],to_lg[curr_edges[i,2]])
    end
    g
end

function read_edgelist_nx(filename)
    G = nx.read_edgelist(filename,nodetype=int)
    return networkx_to_lightgraph(G)
end

end