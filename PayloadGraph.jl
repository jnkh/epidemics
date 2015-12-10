#########################Graph new type#####################3

module PayloadGraph

import LightGraphs

export Graph,create_graph_from_value,fill_graph,set_payload,
get_payload,num_vertices,num_edges,get_average_degree,out_edges,
neighbors,vertices,add_edge!,add_vertex!,is_connected,set_array_with_payload

type Graph{P}
    g::LightGraphs.Graph
    payload::Array{P,1}
    
    Graph(a::LightGraphs.Graph,b::Array{P,1}) =  LightGraphs.nv(a) == length(b) ? new(a,b) :  error("Incorrect array length")
end

Graph{P}(a::LightGraphs.Graph,b::Array{P,1}) = Graph{P}(a,b) 


function create_graph_from_value{P}(g::LightGraphs.Graph,val::P)
    payload = fill(val,LightGraphs.nv(g)) 
    return Graph(g,payload)
end

function fill_graph{P}(g::PayloadGraph.Graph{P},val::P)
    g.payload = fill(val,num_vertices(g))
end


function set_payload{P}(g::PayloadGraph.Graph{P},val_array::Union{Array{P,1},SharedArray{P,1}})
    for i in 1:length(g.payload)
        g.payload[i] = val_array[i]
    end
end

function set_array_with_payload{P}(g::PayloadGraph.Graph{P},val_array::Union{Array{P,1},SharedArray{P,1}})
    for i in 1:length(g.payload)
        val_array[i] = g.payload[i]
    end
end

function set_payload{P}(g::PayloadGraph.Graph{P},v::Int,val::P)
    g.payload[v] = val
end

function set_payload_for_all{P}(g::PayloadGraph.Graph{P},val::P)
    g.payload[:] = val
end

function get_payload{P}(g::PayloadGraph.Graph{P})
    return g.payload
end

function get_payload{P}(g::PayloadGraph.Graph{P},v::Int)
    return g.payload[v]
end

num_vertices{P}(g::PayloadGraph.Graph{P}) = LightGraphs.nv(g.g)
num_edges{P}(g::PayloadGraph.Graph{P}) = LightGraphs.ne(g.g)
vertices{P}(g::PayloadGraph.Graph{P}) = LightGraphs.vertices(g.g)
is_connected{P}(g::Graph{P}) = LightGraphs.is_connected(g.g)

out_edges{P}(g::PayloadGraph.Graph{P},v::Int) = LightGraphs.out_edges(g.g,v)
neighbors{P}(g::PayloadGraph.Graph{P},v::Int) = LightGraphs.neighbors(g.g,v)

add_edge!{P}(g::PayloadGraph.Graph{P},src::Int,dst::Int) = LightGraphs.add_edge!(g.g,src,dst)

function add_vertex!{P}(g::PayloadGraph.Graph{P},val::P)
    push!(g.payload,val)
    LightGraphs.add_vertex!(g.g)
end

rem_edge!{P}(g::Graph{P},src::Int,dst::Int) = LightGraphs.rem_edge!(g.g,src,dst)




function get_average_degree{P}(g::PayloadGraph.Graph{P})
    return 2*num_edges(g)/num_vertices(g)
end

end

#############################################################
