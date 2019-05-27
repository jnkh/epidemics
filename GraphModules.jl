#########################TypedGraph new type#####################3

using Graphs, IM

struct TypedGraph{V,E}
    g::AbstractGraph{V,E}
    node_types::Array
    
    #TypedGraph(a,b) =  num_vertices(a) == length(b) ? new(a,b) :  error("Incorrect array length")

end


function get_neighbor_fraction_of_type{V,E}(t::TypedGraph{V,E},v::V,thistype)
    neighbors = out_neighbors(v,t.g)
    if length(neighbors) == 0 return 0.0 end
    count = 0
    for n in neighbors
        if get_type(t,n) == thistype
            count += 1
        end
    end
    return count/length(neighbors)
end

function get_fraction_of_type{V,E}(t::TypedGraph{V,E},thistype)
    count = 0
    vs = vertices(t.g)
    for v in vs
        if get_type(t,v) == thistype
            count += 1
        end
    end
    return count/length(vs)
end


function print_graph{V,E}(t::TypedGraph{V,E})
    for v in vertices(t.g)
        print(get_type(t,v))
    end
end

function update_graph{V,E}(t::TypedGraph{V,E},im::InfectionModel)

    new_types = copy(t.node_types)
    for v in vertices(t.g)
        update_node(t,v,im,new_types)
    end
    
    update_types(t,new_types)
end


function update_node{V,E}(t::TypedGraph{V,E},v::V,im::InfectionModel,new_types)
    if get_type(t,v) == INFECTED
        k = get_average_k(t) 
        #infect neighbors
        neighbors = out_neighbors(v,t.g)
        for w in neighbors
            if get_type(t,w) == SUSCEPTIBLE
                x = get_neighbor_fraction_of_type(t,w,INFECTED)
                p = p_birth(im,x)/k
                if rand() < p
                    new_types[vertex_index(w,t.g)] = INFECTED
                end
            end
        end
        
        #recover self
        x =get_neighbor_fraction_of_type(t,v,INFECTED)
        p = p_death(im,x)
        if rand() < p
            new_types[vertex_index(v,t.g)] = SUSCEPTIBLE
        end
    end

end

function update_types{V,E}(t::TypedGraph{V,E},new_types)
    for v in vertices(t.g)
        set_type(t,v,new_types[vertex_index(v,t.g)])
    end
end

function set_all_types{V,E}(t::TypedGraph{V,E},new_type)
    for v in vertices(t.g)
        set_type(t,v,new_type)
    end
end


function TypedGraph{V,E}(g::AbstractGraph{V,E})
    node_types = zeros(length(vertices(g)))
    return TypedGraph(g,node_types)
end

function get_type{V,E}(t::TypedGraph{V,E},v::V)
    return t.node_types[vertex_index(v,t.g)]
end

function set_type{V,E}(t::TypedGraph{V,E},v::V,node_type::Any)
    t.node_types[vertex_index(v,t.g)] = node_type
end

function get_average_k{V,E}(t::TypedGraph{V,E})
    return 2*num_edges(t.g)/num_vertices(t.g)
end






