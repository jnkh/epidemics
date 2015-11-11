##################################################
#### SIS Model captured in a module###############
##################################################

module SIS


using PayloadGraph,IM,Distributions

export INFECTED,SUSCEPTIBLE,get_average_degree,
get_fraction_of_type,print_graph,update_graph,set_all_types,get_neighbor_fraction_of_type,get_neighbor_fraction_of_type_new,get_parameters



const INFECTED = 1
const SUSCEPTIBLE = 0

function get_parameters(N,alpha,beta,verbose=false)
    critical_determinant = 4*alpha/(N*beta^2)
    y_n = beta/alpha
    if critical_determinant < 1
        y_minus = beta/(2*alpha)*(1 -  sqrt(1 - critical_determinant))
        y_plus = beta/(2*alpha)*(1 +  sqrt(1 - critical_determinant))
    else
        y_minus = -1
        y_plus = -1
    end
    y_p = beta/(2*alpha)*(1 +  sqrt(1 + critical_determinant))
    if verbose
        println("y_n = $y_n, y_- = $y_minus, y_+ = $y_plus, y_p = $y_p, critical determinant = $critical_determinant")
        println("'n_n = $(y_n*N)")
    end
    return y_n, y_minus, y_plus, y_p,critical_determinant
end

function get_neighbor_fraction_of_type{P}(g::Graph{P},v::Int,thistype::P)
    neighbors = PayloadGraph.neighbors(g,v)
    if length(neighbors) == 0 return 0.0 end
    count = 0
    for n in neighbors
        if get_payload(g,n) == thistype
            count += 1
        end
    end
    return count/length(neighbors)
end

function get_fraction_of_type{P}(g::Graph{P},thistype::P)
    count = 0
    vs = vertices(g)
    for v in vs
        if get_payload(g,v) == thistype
            count += 1
        end
    end
    return count/length(vs)
end



function get_sample_of_types_from_neighbors{P}(g::Graph{P},v::P)
    neighbors = PayloadGraph.neighbors(g,v)
    if length(neighbors) == 0
        error("Disconnected Graph")
        return get_payload(g,v)
    end
    neighbor_types = Array(P,length(neighbors))
    for (i,w) in enumerate(neighbors)
        neighbor_types[i] = get_payload(g,w)
    end
    return sample(neighbor_types)
end


function print_graph{P}(g::Graph{P})
    for v in vertices(g)
        print(get_payload(g,v))
    end
end

function update_graph{P}(g::Graph{P},im::InfectionModel)

    new_types = copy(get_payload(g))
    for v in vertices(g)
        update_node(g,v,im,new_types)
    end
    
    update_types(g,new_types)
end


function update_node{P}(g::Graph{P},v::Int,im::InfectionModel,new_types)
    if get_payload(g,v) == INFECTED
        k = get_average_degree(g) 
        #infect neighbors
        neighbors = PayloadGraph.neighbors(g,v)
        for w in neighbors
            if get_payload(g,w) == SUSCEPTIBLE
                x = get_neighbor_fraction_of_type(g,w,INFECTED)
                p = p_birth(im,x)/k
                if rand() < p
                    new_types[w] = INFECTED
                end
            end
        end
        
        #recover self
        x =get_neighbor_fraction_of_type(g,v,INFECTED)
        p = p_death(im,x)
        if rand() < p
            new_types[v] = get_sample_of_types_from_neighbors(g,v)
        end
    end

end


function update_types{P}(g::Graph{P},new_types::Array{P,1})
    for v in vertices(g)
        set_payload(g,v,new_types[v])
    end
end


end
