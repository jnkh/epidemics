##################################################
#### SIS Model captured in a module###############
##################################################

module SIS


using PayloadGraph,IM,Distributions, Base.Threads 

export INFECTED,SUSCEPTIBLE,get_average_degree,
get_fraction_of_type,print_graph,update_graph,set_all_types,get_neighbor_fraction_of_type,get_neighbor_fraction_of_type_new,get_parameters,update_graph_threads



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

function get_sample_of_types_from_neighbors_threadsafe{P}(g::Graph{P},v::P,rng)
    neighbors = PayloadGraph.neighbors(g,v)
    if length(neighbors) == 0
        error("Disconnected Graph")
        return get_payload(g,v)
    end
    return get_payload(g,sample_threadsafe(neighbors,rng))
end



function sample_threadsafe{T}(arr::Array{T,1},rng)
    len = length(arr)
    rand_idx = Int(floor(1 + rand(rng)*len))
    return arr[rand_idx]
end



function print_graph{P}(g::Graph{P})
    for v in vertices(g)
        print(get_payload(g,v))
    end
end

function println_safe(arg,m::Mutex)
    lock!(m)
    println(arg)
    unlock!(m)
end



#This function doesn't use the infection model yet since anonymous functions crash in the threading package!
function update_graph_threads{P}(g::Graph{P},im::InfectionModel,new_types::Union{Array{P,1},SharedArray{P,1}})
    rngs = [MersenneTwister(777+x) for x in 1:nthreads()]
    m = Mutex()
    set_array_with_payload(g,new_types)
    # @sync @parallel for v in vertices(g)
    @threads all for v = 1:length(vertices(g))
        if get_payload(g,v) == INFECTED

            neighbors = PayloadGraph.neighbors(g,v)
            k = get_average_degree(g) 
            for w in neighbors
                if get_payload(g,w) == SUSCEPTIBLE
                    x = get_neighbor_fraction_of_type(g,w,INFECTED)
                    # p = p_birth(im,k)/k
                    # p = p_birth(im,x)/k
                    p = pb(x)/k
                    if rand(rngs[threadid()]) < 0.05#p
                        # println_safe("birth at node $w",m)
                        lock!(m)
                        new_types[w] = INFECTED
                        unlock!(m)
                    end
                end
            end
                #infect neighbors
            #recover self
            x =get_neighbor_fraction_of_type(g,v,INFECTED)
            # p = p_death(im,x)
            # p = p_death(im,x)
            p = pd(x)
            if rand(rngs[threadid()]) < 0.2#p
            #     # println_safe("death at node $v",m)
            samp = Int(get_sample_of_types_from_neighbors_threadsafe(g,v,rngs[threadid()]))
                 lock!(m)
                new_types[v] = samp
                unlock!(m);
            end
        end
    end
    set_payload(g,new_types)
end

function pd(x)
    return 1 + 0.1
end

function pb(x)
    return 1 + 0.1*x
end




function update_graph{P}(g::Graph{P},im::InfectionModel,new_types::Array{P,1})

    set_array_with_payload(g,new_types)
    @sync @parallel for v in vertices(g)
    for v in vertices(g)
        update_node(g,v,im,new_types)
    end
    set_payload(g,new_types)
end

function update_node_threads{P}(g::Graph{P},v::Int,im::InfectionModel,new_types,rngs,m::Mutex)
    if get_payload(g,v) == INFECTED
        k = get_average_degree(g) 
        #infect neighbors
        neighbors::Array{Int,1} = PayloadGraph.neighbors(g,v)
        p = 0.0
        for w in neighbors
            if get_payload(g,w) == SUSCEPTIBLE
                x = get_neighbor_fraction_of_type(g,w,INFECTED)
                p::Float64 = p_birth(im,x)/k
                if rand(rngs[threadid()]) < p
                    lock!(m); new_types[w] = 2; unlock!(m);#INFECTED; unlock!(m)
                end
            end
        end
        
        #recover self
        x =get_neighbor_fraction_of_type(g,v,INFECTED)
        p = p_death(im,x)
        if rand(rngs[threadid()]) < p
            lock!(m); new_types[v] = 2; unlock!(m);#get_sample_of_types_from_neighbors(g,v); unlock!(m)
        end
    end

end


function update_node{P}(g::Graph{P},v::Int,im::InfectionModel,new_types)
    if get_payload(g,v) == INFECTED
        k = get_average_degree(g) 
        #infect neighbors
        neighbors::Array{Int,1} = PayloadGraph.neighbors(g,v)
        p = 0.0
        for w in neighbors
            if get_payload(g,w) == SUSCEPTIBLE
                x = get_neighbor_fraction_of_type(g,w,INFECTED)
                p::Float64 = p_birth(im,x)/k
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
