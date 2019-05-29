##################################################
#### SIS Model captured in a module###############
##################################################

module SIS

import LightGraphs,SharedArrays
using PayloadGraph,IM,Distributions,DegreeDistribution,StatsBase

export INFECTED,SUSCEPTIBLE,RateArray,
get_average_degree,
get_fraction_of_type,print_graph,update_graph,set_all_types,
get_neighbor_fraction_of_type,
get_neighbor_number_of_type,

get_parameters,

update_graph_threads,get_c_r,get_n_n,get_alpha_beta,
update_graph_experimental, update_graph_gillespie,
compute_all_rates,
compute_all_neighbor_numbers_of_type,
pick_update_and_time



const INFECTED = 1
const SUSCEPTIBLE = 0


function get_c_r(N,alpha,beta)
    return 4*alpha/(beta^2*N)
end

function get_n_n(N,alpha,beta)
    return beta/alpha*N
end

function get_alpha_beta(N,c_r,n_n)
    beta = 4.0/(c_r*n_n)
    alpha = (N*beta)/n_n
    return alpha,beta
end


f(y,alpha) = alpha.*y.^2
s(y,alpha,beta) = f(y,alpha)./y - beta
#get_y_eff(y,k) = y.*(1 + (1-y)./(y.*k))
#get_s_eff(y::Array,alpha,beta,k) = alpha*get_y_eff(y,k) - beta


#function according to when P_reach is neutral and when it is not
function get_parameters_exact(N,alpha,beta,verbose=false)
    critical_determinant = 2*alpha/(N*beta^2)
    y_n = 2*beta/alpha
    if critical_determinant < 1
        y_minus = beta/(alpha)*(1 -  sqrt(1 - critical_determinant))
        y_plus = beta/(alpha)*(1 +  sqrt(1 - critical_determinant))
    else
        y_minus = -1
        y_plus = -1
    end
    y_p = beta/(alpha)*(1 +  sqrt(1 + critical_determinant))
    if verbose
        println("y_n = $y_n, y_- = $y_minus, y_+ = $y_plus, y_p = $y_p, critical determinant = $critical_determinant")
        println("'n_n = $(y_n*N)")
    end
    return y_n, y_minus, y_plus, y_p,critical_determinant
end


#function according to a theory where s is assumed constant
function get_parameters(N,alpha,beta,verbose=false;exact=false)
    if exact
        alpha = alpha/2
    end
    critical_determinant = 4*alpha/(N*beta^2)
    y_n = beta/alpha
    pre_fac = beta/(2*alpha)

    if critical_determinant < 1
        y_minus = pre_fac*(1 -  sqrt(1 - critical_determinant))
        y_plus = pre_fac*(1 +  sqrt(1 - critical_determinant))
    else
        y_minus = -1
        y_plus = -1
    end
    y_p = pre_fac*(1 +  sqrt(1 + critical_determinant))
    if verbose
        println("y_n = $y_n, y_- = $y_minus, y_+ = $y_plus, y_p = $y_p, critical determinant = $critical_determinant")
        println("'n_n = $(y_n*N)")
    end
    return y_n, y_minus, y_plus, y_p,critical_determinant
end



#####################################################
##################Begin Experimental###################
#####################################################




function get_degree_experimental(g::Graph{P},v::Int) where {P}
    return length(PayloadGraph.neighbors(g,v))
end

# function get_fraction_of_type_experimental{P}(g::Graph{P},thistype::P)
#     count = 0
#     vs = vertices(g)
#     for v in vs
#         if get_payload(g,v) == thistype
#             count += 1
#         end
#     end
#     return count/length(vs)
# end



function get_sample_of_types_from_neighbors_experimental(g::Graph{P},v::P,p_k,p_k_n,n_k) where {P}
    neighbors = PayloadGraph.neighbors(g,v)
    # neighbor_types = sample(g.payload)
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

function random_node_of_degree_type(g,k)
    N = length(vertices(g))
    while true
        idx = rand(1:N)
        if LightGraphs.degree(g.g,idx) == k
            return get_payload(g,idx)
        end
    end
end

function random_node_of_same_degree(g,v,self_v)
    k = LightGraphs.degree(g.g,v)
    N = length(vertices(g))
    while true
        idx = rand(1:N)
        if idx != self_v && LightGraphs.degree(g.g,idx) == k
            return idx
        end
    end
end


function shuffle_graph_by_degrees(g,N,ks,ks_map,n_k)
    new_payload = fill(SUSCEPTIBLE,N)
    k_vec = LightGraphs.degree(g.g)
    for k in ks
        this_degree_indices = find(k_vec .== k)
        this_degree_infecteds = sample(this_degree_indices,n_k[ks_map[k]],replace=false)
        new_payload[this_degree_infecteds] = INFECTED 
    end
    set_payload(g,new_payload)
end




function get_neighbor_fraction_of_type_experimental(g::Graph{P},v::Int,thistype::P,N::Int,N_k,ks_map,p_k,p_k_n,n_k) where {P}
    neighbors = PayloadGraph.neighbors(g,v)
    k = length(neighbors)
    count = 0



    # k_vec = zeros(p_k)
    # for w in neighbors
    #     k_vec[ks_map[LightGraphs.degree(g.g,w)]] += 1
    # end
    # assert(sum(k_vec) == k)

    # # k_vec = rand(Multinomial(k,p_k_n[ks_map[k],:]))
    # for (k_idx,k_count) in enumerate(k_vec)
    #     if k_count > 0
    #         tot = N_k[k_idx]
    #         s = n_k[k_idx]
    #         f = tot - s 
    #         # count += rand(Binomial(Int(k_count),s/tot))
    #         count += rand(Hypergeometric(s,f,k_count))
    #     end
    # end


    for n in neighbors
        # pl = random_node_of_degree_type(g,LightGraphs.degree(g.g,n))
        # if pl == thistype
        m = random_node_of_same_degree(g,n,v)
        if get_payload(g,m) == thistype
        # if get_payload(g,n) == thistype
            count += 1
        end
    end


    # neighbors = sample(collect(1:length(g.payload)),k)
    # return rand(Binomial(k,get_fraction_of_type(g,thistype)))/k
    # if length(neighbors) == 0 return 0.0 end
    # count = 0
    # for n in neighbors
    #     if get_payload(g,n) == thistype
    #         count += 1
    #     end
    # end
    return count/k
end

function calculate_n_k(g,ks,ks_map)
    N = LightGraphs.nv(g.g)
    n_k = zeros(ks)
    for v in vertices(g)
        k = LightGraphs.degree(g.g,v)
        if get_payload(g,v) == INFECTED
            n_k[ks_map[k]] += 1
        end
    end
    @assert( sum(n_k)/N == get_fraction_of_type(g,INFECTED))
    n_k
end


function update_graph_experimental(g::Graph{P},im::Union{InfectionModel,InfectionModelLinear},new_types::Union{Array{P,1},SharedArrays.SharedArray{P,1}}) where {P}
    ks,ks_map,N_k,p_k,p_k_n = create_p_k_p_k_neighbor_from_graph(g.g)
    n_k = calculate_n_k(g,ks,ks_map)
    # p_k,p_k_n,n_k = 0,0,0
    N = length(vertices(g))
    shuffle_graph_by_degrees(g,N,ks,ks_map,n_k)

    set_array_with_payload(g,new_types)
    # @sync @parallel for v in vertices(g)
    for v in vertices(g)
        update_node_experimental(g,v,im,new_types,N,N_k,ks_map,p_k,p_k_n,n_k)
        # update_node_experimental(g,v,im,new_types)
    end
    set_payload(g,new_types)
end

function update_node_experimental(g::Graph{P},v::Int,im::Union{InfectionModel,InfectionModelLinear},new_types::Union{Array{P,1},SharedArrays.SharedArray{P,1}},N,N_k,ks_map,p_k,p_k_n,n_k) where {P}
    # if get_payload(g,v) == INFECTED
    #     # k = get_average_degree(g) 
    #     #infect neighbors
    #     neighbors::Array{Int,1} = PayloadGraph.neighbors(g,v)
    #     p = 0.0
    #     for w in neighbors
    #         if get_payload(g,w) == SUSCEPTIBLE
    #             # x = get_neighbor_fraction_of_type(g,w,INFECTED)
    #             x = get_neighbor_fraction_of_type_experimental(g,w,INFECTED,N,N_k,ks_map,p_k,p_k_n,n_k)
    #             k = get_degree(g,w)
    #             p::Float64 = p_birth(im,x)/k
    #             if rand() < p
    #                 new_types[w] = INFECTED
    #             end
    #         end
    #     end
        
    #     #recover self
    #     # x =get_neighbor_fraction_of_type(g,v,INFECTED)
    #     x = get_neighbor_fraction_of_type_experimental(g,v,INFECTED,N,N_k,ks_map,p_k,p_k_n,n_k)
    #     p = (1-x)*p_death(im,x)
    #     if rand() < p
    #         new_types[v] = SUSCEPTIBLE#get_sample_of_types_from_neighbors(g,v)
    #     end
    # end

#     y = get_fraction_of_type(g,INFECTED)
#     k = length(neighbors(g,v))
#     # y_sample = rand(Binomial(k,y))/k
    if get_payload(g,v) == SUSCEPTIBLE
        # x = get_neighbor_fraction_of_type_experimental(g,v,INFECTED,N,N_k,ks_map,p_k,p_k_n,n_k)
        x = get_neighbor_fraction_of_type(g,v,INFECTED)
        p = x*p_birth(im,x)
        #infect neighbors
        if rand() < p
            new_types[v] = INFECTED
        end
    elseif get_payload(g,v) == INFECTED 
        x = get_neighbor_fraction_of_type(g,v,INFECTED)
        #recover self
        p = (1-x)*p_death(im,x)
        if rand() < p
            new_types[v] = SUSCEPTIBLE#get_sample_of_types_from_neighbors_experimental(g,v)
        end
    end


end


# function update_node_experimental{P}(g::Graph{P},v::Int,im::Union{InfectionModel,InfectionModelLinear},new_types::Union{Array{P,1},SharedArray{P,1}})
#     y = get_fraction_of_type(g,INFECTED)
#     k = length(neighbors(g,v))
#     # y_sample = rand(Binomial(k,y))/k
#     y_sample = get_neighbor_fraction_of_type_experimental(g,v,INFECTED)
#     if get_payload(g,v) == SUSCEPTIBLE
#         p = y_sample*p_birth(im,y_sample)
#         # k = get_average_degree(g) 
#         #infect neighbors
#         # x = get_neighbor_fraction_of_type_experimental(g,v,INFECTED)
#         # p = x*p_birth(im,x)
#         if rand() < p
#             new_types[v] = INFECTED
#         end
#     elseif get_payload(g,v) == INFECTED 
#         #recover self
#         # x =get_neighbor_fraction_of_type_experimental(g,v,INFECTED)
#         p = (1-y_sample)*p_death(im,y_sample)
#         if rand() < p
#             new_types[v] = SUSCEPTIBLE#get_sample_of_types_from_neighbors_experimental(g,v)
#         end
#     end

# end



#####################################################
##################END Experimental###################
#####################################################





#####################################################
##################BEGIN  Gillespie###################
#####################################################
### Epidemic on a Graph ###

mutable struct RateArray
    arr::Array{Float64,1}
    s::Float64
end

function RateArray(v::Array{Float64,1})
    s = sum(v)
    return RateArray(v,s)
end

function update_sum(r::RateArray)
    r.s = sum(r.arr)
end

function compute_all_rates(g::LightGraphs.Graph,payload::Array{P,1},im::Union{InfectionModel,InfectionModelLinear}) where {P} 
    rates = fill(0.0,num_vertices(g))
    for v in LightGraphs.vertices(g)
        rates[v] = compute_rate(g,payload,v,im)
    end
    return rates
end



function compute_rate(g::Graph{P},v::Int,im::Union{InfectionModel,InfectionModelLinear}) where {P} 
    y = get_neighbor_fraction_of_type(g,v,INFECTED)
    rate::Float64 = 0.0
    if get_payload(g,v) == INFECTED
        rate = (1-y)*p_death(im,y)
    else 
        @assert(get_payload(g,v) == SUSCEPTIBLE )
        rate = y*p_birth(im,y)
    end
    return rate
end

function compute_rate(g::LightGraphs.Graph,payload::Array{P,1},v::Int,neighbor_numbers::Array{Int,1},im::Union{InfectionModel,InfectionModelLinear}) where {P} 
    y = get_neighbor_fraction_of_type(g,payload,v,neighbor_numbers,INFECTED)
    rate = 0.0
    if payload[v] == INFECTED
        rate = (1-y)*p_death(im,y)
    else 
        @assert(payload[v] == SUSCEPTIBLE )
        rate = y*p_birth(im,y)
    end
    return rate
end

function compute_all_rates(g::LightGraphs.Graph,payload::Array{P,1},im::Union{InfectionModel,InfectionModelLinear},rates::RateArray,neighbor_numbers::Array{Int,1}) where {P} 
    for v in LightGraphs.vertices(g)
        rates.arr[v] = compute_rate(g,payload,v,neighbor_numbers,im)
    end
    update_sum(rates)
end

function compute_all_neighbor_numbers_of_type(g::LightGraphs.Graph,payload::Array{P,1},neighbor_numbers::Array{Int,1},thistype::P) where {P} 
    for v in LightGraphs.vertices(g)
        neighbor_numbers[v] = get_neighbor_number_of_type(g,payload,v,thistype)
    end
end

# function pick_update_and_time(rates::RateArray)
    # v = sample(Weights(rates.arr,wsum=rates.s))
    # v = pfsample(rates)
    # t = rand(Exponential(1/rates.arr[v]))
    # return v,t
# end

function pfsample(w::RateArray)
    n::Int = length(w.arr)
    t::Float64 = rand() * w.s
    i::Int = 1
    cw::Float64 = w.arr[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w.arr[i]
    end
    return i
end

function update_nodes_and_rates(g::LightGraphs.Graph,payload::Array{P,1},v::Int,rates::RateArray,neighbor_numbers::Array{Int,1},im::Union{InfectionModel,InfectionModelLinear}) where {P} 
    curr_type = payload[v]
    infected_change::Int = 0
    if curr_type == INFECTED#set to SUSCEPTIBLE
        payload[v] = SUSCEPTIBLE
        infected_change = -1
    else
        payload[v] = INFECTED
        infected_change = +1
    end
    for w in LightGraphs.neighbors(g,v)
        neighbor_numbers[w] += infected_change
    end
    old::Float64 = 0.0
    new::Float64 = 0.0
    old = rates.arr[v]
    new = compute_rate(g,payload,v,neighbor_numbers,im)
    rates.arr[v] = new
    rates.s += (new - old)
    for w in LightGraphs.neighbors(g,v)
        old = rates.arr[w]
        new = compute_rate(g,payload,w,neighbor_numbers,im)
        rates.arr[w] = new
        rates.s += (new - old)
    end
    # if abs(rates.s - sum(rates.arr)) > 1e-8
    #     println(rates.s)
    #     println(sum(rates.arr))
    # end
    return infected_change 
end

function update_graph_gillespie(g::LightGraphs.Graph,payload::Array{P,1},im::Union{InfectionModel,InfectionModelLinear},rates::RateArray,neighbor_numbers::Array{Int,1},num_infected::Int) where {P} 
    # v::Int = pfsample(rates)
    v::Int = sample(Weights(rates.arr,rates.s))
    # t::Float64 = rand(Exponential(1/rates.arr[v]))
    dt::Float64 = rand(Exponential(1/rates.s))
    # v,t = pick_update_and_time(rates)
    infected_change = update_nodes_and_rates(g,payload,v,rates,neighbor_numbers,im)
    return dt,num_infected+infected_change
end





#####################################################
####################END  Gillespie###################
#####################################################

function get_neighbor_number_of_type(g::LightGraphs.Graph,payload::Array{P,1},v::Int,thistype::P) where {P} 
    count::Int = 0
    for n in LightGraphs.neighbors(g,v) 
        if payload[n] == thistype
            count = count + 1
        end
    end
    return count
end

function get_neighbor_number_of_type(g::Graph{P},v::Int,thistype::P) where {P} 
    count::Int = 0
    for n in PayloadGraph.neighbors(g,v) 
        if get_payload(g,n) == thistype
            count = count + 1
        end
    end
    return count
end

function get_neighbor_fraction_of_type(g::LightGraphs.Graph,payload::Array{P,1},v::Int,neighbor_numbers::Array{Int,1},thistype::P) where {P} 
    nn = length(LightGraphs.neighbors(g,v)) 
    if nn == 0 return 0.0 end
    return 1.0*neighbor_numbers[v]/nn
end

function get_neighbor_fraction_of_type(g::Graph{P},v::Int,thistype::P) where {P} 
    if length(PayloadGraph.neighbors(g,v)) == 0 return 0.0 end
    count = get_neighbor_number_of_type(g,v,thistype) 
    return count/length(PayloadGraph.neighbors(g,v))
end

function get_degree(g::Graph{P},v::Int) where {P} 
    return length(PayloadGraph.neighbors(g,v))
end

function get_fraction_of_type(g::Graph{P},thistype::P) where {P} 
    count = 0
    N = num_vertices(g)
    for v in 1:N
        if get_payload(g,v) == thistype
            count += 1
        end
    end
    return 1.0*count/N
end



function get_sample_of_types_from_neighbors(g::Graph{P},v::P) where {P} 
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

function print_graph(g::Graph{P}) where {P} 
    for v in vertices(g)
        print(get_payload(g,v))
    end
end


function update_graph(g::Graph{P},im::Union{InfectionModel,InfectionModelLinear},new_types::Union{Array{P,1},SharedArrays.SharedArray{P,1}}) where {P} 

    set_array_with_payload(g,new_types)
    # @sync @parallel for v in vertices(g)
    for v in vertices(g)
        update_node(g,v,im,new_types)
    end
    set_payload(g,new_types)
end



function update_node(g::Graph{P},v::Int,im::Union{InfectionModel,InfectionModelLinear},new_types::Union{Array{P,1},SharedArrays.SharedArray{P,1}}) where {P} 
    if get_payload(g,v) == INFECTED
        # k = get_average_degree(g) 
        #infect neighbors
        neighbors::Array{Int,1} = PayloadGraph.neighbors(g,v)
        p = 0.0
        for w in neighbors
            if get_payload(g,w) == SUSCEPTIBLE
                x = get_neighbor_fraction_of_type(g,w,INFECTED)
                k = get_degree(g,w)
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


function update_types(g::Graph{P},new_types::Array{P,1}) where {P} 
    for v in vertices(g)
        set_payload(g,v,new_types[v])
    end
end


end



###################THREADING########################
# using Base.Threads 

# function println_safe(arg,m::Mutex)
#     lock!(m)
#     println(arg)
#     unlock!(m)
# end


# function get_sample_of_types_from_neighbors_threadsafe(g::Graph{P},v::P,rng) where {P} 
#     neighbors = PayloadGraph.neighbors(g,v)
#     if length(neighbors) == 0
#         error("Disconnected Graph")
#         return get_payload(g,v)
#     end
#     return get_payload(g,sample_threadsafe(neighbors,rng))
# end



# function sample_threadsafe{T}(arr::Array{T,1},rng)
#     len = length(arr)
#     rand_idx = Int(floor(1 + rand(rng)*len))
#     return arr[rand_idx]
# end





# #This function doesn't use the infection model yet since anonymous functions crash in the threading package!
# function update_graph_threads{P}(g::Graph{P},im::Union{InfectionModel,InfectionModelLinear},new_types::Union{Array{P,1},SharedArrays.SharedArray{P,1}})
#     rngs = [MersenneTwister(777+x) for x in 1:nthreads()]
#     m = Mutex()
#     set_array_with_payload(g,new_types)
#     # @sync @parallel for v in vertices(g)
#     @threads all for v = 1:length(vertices(g))
#         if get_payload(g,v) == INFECTED

#             neighbors = PayloadGraph.neighbors(g,v)
#             k = get_average_degree(g) 
#             for w in neighbors
#                 if get_payload(g,w) == SUSCEPTIBLE
#                     x = get_neighbor_fraction_of_type(g,w,INFECTED)
#                     # p = p_birth(im,k)/k
#                     # p = p_birth(im,x)/k
#                     p = pb(x)/k
#                     if rand(rngs[threadid()]) < 0.05#p
#                         # println_safe("birth at node $w",m)
#                         lock!(m)
#                         new_types[w] = INFECTED
#                         unlock!(m)
#                     end
#                 end
#             end
#                 #infect neighbors
#             #recover self
#             x =get_neighbor_fraction_of_type(g,v,INFECTED)
#             # p = p_death(im,x)
#             q = pd(x)
#             if rand(rngs[threadid()]) < 0.2#p
#             #     # println_safe("death at node $v",m)
#             samp = Int(get_sample_of_types_from_neighbors_threadsafe(g,v,rngs[threadid()]))
#                  lock!(m)
#                 new_types[v] = samp
#                 unlock!(m);
#             end
#         end
#     end
#     set_payload(g,new_types)
# end

# function pd(x)
#     return 1 + 0.1
# end

# function pb(x)
#     return 1 + 0.1*x
# end

# function update_node_threads{P}(g::Graph{P},v::Int,im::Union{InfectionModel,InfectionModelLinear},new_types,rngs,m::Mutex)
#     if get_payload(g,v) == INFECTED
#         k = get_average_degree(g) 
#         #infect neighbors
#         neighbors::Array{Int,1} = PayloadGraph.neighbors(g,v)
#         p = 0.0
#         for w in neighbors
#             if get_payload(g,w) == SUSCEPTIBLE
#                 x = get_neighbor_fraction_of_type(g,w,INFECTED)
#                 p::Float64 = p_birth(im,x)/k
#                 if rand(rngs[threadid()]) < p
#                     lock!(m); new_types[w] = 2; unlock!(m);#INFECTED; unlock!(m)
#                 end
#             end
#         end
        
#         #recover self
#         x =get_neighbor_fraction_of_type(g,v,INFECTED)
#         p = p_death(im,x)
#         if rand(rngs[threadid()]) < p
#             lock!(m); new_types[v] = 2; unlock!(m);#get_sample_of_types_from_neighbors(g,v); unlock!(m)
#         end
#     end

# end



# # end
