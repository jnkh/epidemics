#########################Graph new type#####################3

module PayloadGraph

import LightGraphs,SharedArrays

export Graph,create_graph_from_value,fill_graph,set_payload,
get_payload,num_vertices,num_edges,get_average_degree,out_edges,
neighbors,vertices,add_edge!,add_vertex!,is_connected,set_array_with_payload,
shuffle_payload,shuffle_payload_by_cluster

struct Graph{P}
    g::LightGraphs.Graph
    payload::Array{P,1}
    
    # Graph(a::LightGraphs.Graph,b::Array{P,1}) =  LightGraphs.nv(a) == length(b) ? new(a,b) :  error("Incorrect array length")
    Graph{P}(a,b) where P =  LightGraphs.nv(a) == length(b) ? new(a,b) :  error("Incorrect array length")
end

Graph(a::LightGraphs.Graph,b::Array{P,1}) where P = Graph{P}(a,b); 


function shuffle_payload(g::PayloadGraph.Graph{P}) where {P}
    shuffle!(g.payload)
end

function shuffle_payload_by_cluster(g::PayloadGraph.Graph{P},clusters::Array{Array{Int,1},1}) where {P}
    for cluster in clusters
        g.payload[cluster] = shuffle(g.payload[cluster])
    end
end

function create_graph_from_value(g::LightGraphs.Graph,val::P) where {P}
    payload = fill(val,LightGraphs.nv(g)) 
    return Graph(g,payload)
end

function fill_graph(g::PayloadGraph.Graph{P},val::P) where {P}
    g.payload = fill(val,num_vertices(g))
end


function set_payload(g::PayloadGraph.Graph{P},val_array::Union{Array{P,1},SharedArrays.SharedArray{P,1}}) where {P}
    for i in 1:length(g.payload)
        g.payload[i] = val_array[i]
    end
end

function set_array_with_payload(g::PayloadGraph.Graph{P},val_array::Union{Array{P,1},SharedArrays.SharedArray{P,1}}) where {P}
    for i in 1:length(g.payload)
        val_array[i] = g.payload[i]
    end
end

function set_payload(g::PayloadGraph.Graph{P},v::Int,val::P) where {P}
    g.payload[v] = val
end

function set_payload_for_all(g::PayloadGraph.Graph{P},val::P) where {P}
    g.payload[:] = val
end

function get_payload(g::PayloadGraph.Graph{P}) where {P}
    return g.payload
end

function get_payload(g::PayloadGraph.Graph{P},v::Int) where {P}
    return g.payload[v]
end

num_vertices(g::PayloadGraph.Graph{P}) where {P} = LightGraphs.nv(g.g)
num_edges(g::PayloadGraph.Graph{P}) where {P} = LightGraphs.ne(g.g)
vertices(g::PayloadGraph.Graph{P}) where {P} = LightGraphs.vertices(g.g)
is_connected(g::Graph{P}) where {P} = LightGraphs.is_connected(g.g)

out_edges(g::PayloadGraph.Graph{P},v::Int) where {P} = LightGraphs.out_edges(g.g,v)
neighbors(g::PayloadGraph.Graph{P},v::Int) where {P} = LightGraphs.neighbors(g.g,v)

add_edge!(g::PayloadGraph.Graph{P},src::Int,dst::Int) where {P} = LightGraphs.add_edge!(g.g,src,dst)

function add_vertex!(g::PayloadGraph.Graph{P},val::P) where {P}
    push!(g.payload,val)
    LightGraphs.add_vertex!(g.g)
end

rem_edge!(g::Graph{P},src::Int,dst::Int) where {P} = LightGraphs.rem_edge!(g.g,src,dst)




function get_average_degree(g::PayloadGraph.Graph{P}) where {P}
    return 2*num_edges(g)/num_vertices(g)
end

end

#############################################################
