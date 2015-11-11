##################################################
#### SIS Model captured in a module###############
##################################################


module SIS

using Graphs,IM,Distributions
export INFECTED,SUSCEPTIBLE,TypeGraph,set_type,get_type_graph,get_average_k,
get_fraction_of_type,print_graph,update_graph,set_all_types,get_neighbor_fraction_of_type,get_neighbor_fraction_of_type_new



    const INFECTED = 1
    const SUSCEPTIBLE = 0

    typealias TypeGraph{K}  AbstractGraph{KeyVertex{K},Edge{KeyVertex{K}}}

    function get_type_graph(g,initial_type=SUSCEPTIBLE)
        K = typeof(initial_type)

        #add vertices
        new_vs = Array(KeyVertex{K}, length(vertices(g)))
        i = 1
        for vertex in vertices(g)
            new_vs[i] = KeyVertex(i, initial_type)
            i += 1
        end

        g1 = inclist(new_vs,is_directed=false)
        #add_edges
        for vertex in vertices(g)
            for edge in out_edges(vertex,g)
                if (source(edge,g) <= target(edge,g))
                    add_edge!(g1, new_vs[source(edge,g)], new_vs[target(edge,g)])
                end
            end
        end
        return g1
    end

    function get_type{K}(v::KeyVertex{K})
        return v.key
    end

    function set_type{K}(v::KeyVertex{K},node_type::K)
        v.key = node_type
    end

    function get_average_k(g::AbstractGraph)
        return 2*num_edges(g)/num_vertices(g)
    end

    function get_neighbor_fraction_of_type{K}(g::TypeGraph{K},v::KeyVertex{K},thistype::K)
        neighbors = out_edges(v,g)
        if length(neighbors) == 0 return 0.0 end
        count = 0
        for e in neighbors
            if get_type(e.target) == thistype
                count += 1
            end
        end
        return count/length(neighbors)
    end

    function get_sample_of_types_from_neighbors{K}(g::TypeGraph{K},v::KeyVertex{K})
        neighbors = out_edges(v,g)
        if length(neighbors) == 0 return get_type(v) end
        neighbor_types = Array(K,length(neighbors))
        for (i,e) in enumerate(neighbors)
            neighbor_types[i] = get_type(e.target)
        end
        return sample(neighbor_types)
    end


    function get_fraction_of_type{K}(g::TypeGraph{K},thistype::K)
        count = 0
        vs = vertices(g)
        for v in vs
            if get_type(v) == thistype
                count += 1
            end
        end
        return count/length(vs)
    end


    function print_graph{K}(g::TypeGraph{K})
        for v in vertices(g)
            print(get_type(v))
        end
    end

    function update_graph{K}(g::TypeGraph{K},im::InfectionModel)

        new_types = get_types_array(g)
        for v in vertices(g)
            update_node(g,v,im,new_types)
        end

        update_types(g,new_types)
    end

    function get_types_array{K}(g::TypeGraph{K})
        types = K[]
        for v in vertices(g)
            push!(types,get_type(v))
        end
        types
    end


    function update_node{K}(g::TypeGraph{K},v::KeyVertex{K},im::InfectionModel,new_types::Array{K,1})
        if get_type(v) == INFECTED
            k = get_average_k(g) 
            #infect neighbors
            neighbors = out_neighbors(v,g)
            for w in neighbors
                if get_type(w) == SUSCEPTIBLE
                    x = get_neighbor_fraction_of_type(g,w,INFECTED)
                    p = p_birth(im,x)/k
                    if rand() < p
                        new_types[vertex_index(w,g)] = INFECTED
                    end
                end
            end

            #recover self
            x =get_neighbor_fraction_of_type(g,v,INFECTED)
            p = p_death(im,x)
            if rand() < p
                new_types[vertex_index(v,g)] = get_sample_of_types_from_neighbors(g,v)
            end
        end

    end

    function update_types{K}(g::TypeGraph{K},new_types::Array{K,1})
        for v in vertices(g)
            set_type(v,new_types[vertex_index(v,g)])
        end
    end

    function set_all_types{K}(g::TypeGraph{K},new_type::K)
        for v in vertices(g)
            set_type(v,new_type)
        end
    end

end
