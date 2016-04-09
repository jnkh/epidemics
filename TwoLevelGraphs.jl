module TwoLevelGraphs

using LightGraphs, Distributions, StatsBase

export TwoLevel, is_valid, get_num_infected, distribute_randomly, make_consistent, TwoLevelGraph, get_clusters, make_two_level_random_graph,
birth_fn,death_fn,adjust_infecteds,get_stationary_distribution,p_j_plus,p_j_minus,
compute_mean_y_local,compute_mean_y_squared_local,set_y, get_interpolations, get_stationary_distribution_nonlinear_theory,
generate_transition_matrix


type TwoLevel
    a::Array{Number,1} #number communities with [idx] infected nodes
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

function TwoLevel(t::TwoLevel)
  return TwoLevel(t.a,t.N,t.m,t.n,t.i,t.r,t.l)
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
    make_consistent(t)
end

function make_consistent(t::TwoLevel)
    t.n = Int(round(sum(t.a)))
    t.i = Int(round(get_num_infected(t)))
end

function set_y(t::TwoLevel,y_desired::AbstractFloat)
    adjust_infecteds(t,y_desired)
    make_consistent(t)
end    



#implement the MCMC exploration of the state space

##TODO: successive steps, not same time steps.

function run_mcmc_transition(t::TwoLevel,death_fn::Function,birth_fn::Function,plotting=false)
    death_probs = zeros(t.m)
    birth_probs = zeros(t.m)

    #birth
    for j = 1:t.m
        birth_probs[j] = birth_fn(t,j-1)
    end

    birth_idx = StatsBase.sample(WeightVec(birth_probs))
    birth_at(t,birth_idx)

    #death
    for j = 1:t.m
        death_probs[j] = death_fn(t,j)
    end
    death_idx = StatsBase.sample(WeightVec(death_probs))
    death_at(t,death_idx)


#     semilogy(birth_probs/death_probs)
#     println(birth_probs)
#     println(death_probs)
end

function birth_at(t::TwoLevel,idx::Int)
    t.a[idx] -= 1
    t.a[idx+1] += 1
end

function death_at(t::TwoLevel,idx::Int)
    t.a[idx+1] -= 1
    t.a[idx] += 1
end



function get_y_external(t::TwoLevel,j::Int)
  y_ext = (t.i - j)/(t.N - t.m)
  return y_ext

end

function get_y_internal(t::TwoLevel,j::Int,susceptible::Bool)
  if susceptible
    y_int = j/t.m
  else
    y_int = (j-1)/t.m
  end
  return y_int
end

function get_y_local(t::TwoLevel,j::Int,susceptible::Bool)
    y_ext = get_y_external(t,j)
    y_int = get_y_internal(t,j,susceptible)
    y_local = (t.l*y_int + t.r*y_ext)/(t.r + t.l)
  return y_local
end

function get_y_squared_local(t::TwoLevel,j::Int,susceptible::Bool)
  y_ext = get_y_external(t,j)
  y_int = get_y_internal(t,j,susceptible)
  y_squared_local = (t.r*y_ext*((t.r - 1)* y_ext + 1) +
t.l*y_int*((t.l - 1)*y_int + 1) + 2*t.l* t.r* y_ext*y_int)/(t.r + t.l)^2

  return y_squared_local
end


function birth_fn(t::TwoLevel,j::Int,alpha=1.0)
  return t.a[get_a_index(j)]*p_j_plus(t,j,alpha)
end


function death_fn(t::TwoLevel,j::Int,beta=0.1)
  return t.a[get_a_index(j)]* p_j_minus(t,j,beta)
end

function p_j_plus(t::TwoLevel,j::Int,alpha=1.0)
  y_local = get_y_local(t,j,true)
  y_squared_local = get_y_squared_local(t,j,true)
  return (t.m - j)*(y_local + alpha*y_squared_local)
end

function p_j_minus(t::TwoLevel,j::Int,beta=0.10)
  y_local = get_y_local(t,j,false)
  return j*(1 - y_local)*(1 + beta)

end

function get_a_index(j::Int)
  return j+1
end

function get_number_weight(t::TwoLevel,j::Int,susceptible::Bool)
  if susceptible
    return t.a[get_a_index(j)]*(t.m-j)
  else
    return t.a[get_a_index(j)]*(j)
  end
end

function compute_mean_y_local(t::TwoLevel,susceptible::Bool)
  y_local = 0.
  weight = 0.

  for j = 0:t.m
    curr_weight = get_number_weight(t,j,susceptible)
    weight += curr_weight
    y_local += curr_weight*get_y_local(t,j,susceptible)
  end
  if weight == 0.0 return 0.0 end
  return y_local/weight
end

function compute_mean_y_squared_local(t::TwoLevel,susceptible::Bool)
  y_local = 0.
  weight = 0.

  for j = 0:t.m
    curr_weight = get_number_weight(t,j,susceptible)
    weight += curr_weight
    y_local += curr_weight*get_y_squared_local(t,j,susceptible)
  end
  if weight == 0.0 return 0.0 end
  return y_local/weight
end


function compute_mean_y_local(t::TwoLevel)
  y_local = 0.
  weight = 0.

  for j = 0:t.m
    for susceptible = [true,false]
      curr_weight = get_number_weight(t,j,susceptible)
      weight += curr_weight
      y_local += curr_weight*get_y_local(t,j,susceptible)
    end
  end
  if weight == 0.0 return 0.0 end
  return y_local/weight
end

function compute_mean_y_squared_local(t::TwoLevel)
  y_local = 0.
  weight = 0.

  for j = 0:t.m
    for susceptible = [true,false]
      curr_weight = get_number_weight(t,j,susceptible)
      weight += curr_weight
      y_local += curr_weight*get_y_squared_local(t,j,susceptible)
    end
  end
  if weight == 0.0 return 0.0 end
  return y_local/weight
end


function adjust_infecteds(t::TwoLevel,y_desired::AbstractFloat)
    t.a[max(1,Int(round(y_desired*t.m)))] = t.n
    num_infected_desired = Int(round(y_desired*t.N))
    if get_num_infected(t) > num_infected_desired
        while true
            decrease_infecteds_by_one(t)
            if get_num_infected(t) <= num_infected_desired
                return
            end
        end
    end
    if get_num_infected(t) < num_infected_desired
        while true
            increase_infecteds_by_one(t)
            if get_num_infected(t) >= num_infected_desired
                return
            end
        end
    end
end

function decrease_infecteds_by_one(t::TwoLevel)
    largest_idx = findlast(x -> x > 0,t.a)
    if largest_idx > 1
        t.a[largest_idx] -= 1
        t.a[largest_idx - 1] += 1
    end
end

function increase_infecteds_by_one(t::TwoLevel)
    smallest_idx = findfirst(x -> x > 0,t.a)
    if smallest_idx < length(t.a)
        t.a[smallest_idx] -= 1
        t.a[smallest_idx+1] += 1
    end
end

function get_stationary_distribution(N::Int,m::Int,l::Int,r::Int,y_desired::AbstractFloat,death_fn::Function,birth_fn::Function,num_trials = 100_000)
    t = TwoLevel(N,m,l,r)
    #distribute_randomly(t,n)
    adjust_infecteds(t,y_desired)
    make_consistent(t)
    assert(is_valid(t))
    get_stationary_distribution(t,death_fn,birth_fn,num_trials)
end


function get_stationary_distribution(t::TwoLevel,death_fn::Function,birth_fn::Function,num_trials=100_000)
    burn_in = Int(round(num_trials/4))
    accum = zeros(t.a)
    dfn(x) = 1
    bfn(x) = 1
    for i = 1:burn_in
      run_mcmc_transition(t,death_fn, birth_fn,true)
    end
    for i = 1:num_trials
      run_mcmc_transition(t,death_fn, birth_fn,true)
      accum += t.a
    end
    accum /= num_trials
    return accum
end



############### TRANSITION MATRIX THEORY ###############

function get_stationary_distribution_theory(N::Int,m::Int,l::Int,r::Int,y_desired::AbstractFloat,alpha::AbstractFloat,beta::AbstractFloat)
    t = TwoLevel(N,m,l,r)
    #distribute_randomly(t,n)
    adjust_infecteds(t,y_desired)
    make_consistent(t)
    assert(is_valid(t))
    return get_stationary_distribution_theory(t,alpha,beta)
end

function get_stationary_distribution_nonlinear_theory(N::Int,m::Int,l::Int,r::Int,y_desired::AbstractFloat,alpha::AbstractFloat,beta::AbstractFloat)
    t = TwoLevel(N,m,l,r)
    #distribute_randomly(t,n)
    adjust_infecteds(t,y_desired)
    make_consistent(t)
    assert(is_valid(t))
    return get_stationary_distribution_nonlinear_theory(t,alpha,beta,y_desired)
end

function get_stationary_distribution_theory(t::TwoLevel,alpha::AbstractFloat,beta::AbstractFloat)
  transition_matrix = generate_transition_matrix(t,alpha,beta)
  equilibrium_distribution = nullspace(transition_matrix)
  if size(equilibrium_distribution)[2] > 1 
    println("PROBLEM: $(size(equilibrium_distribution)[2]) SOLUTIONS");
    println(equilibrium_distribution)
  end
  equilibrium_distribution = equilibrium_distribution[:,end]
  equilibrium_distribution *= t.n/sum(equilibrium_distribution)
  return equilibrium_distribution
end

function generate_transition_matrix(t::TwoLevel,alpha,beta)
    p_plus_arr = zeros(t.m+1)
    p_minus_arr = zeros(t.m+1)

    for idx = 1:(t.m + 1)
        j = idx-1
        p_plus_arr[idx] = p_j_plus(t,j,alpha)
        p_minus_arr[idx] = p_j_minus(t,j,beta)
    end

#     gamma = sum(p_plus_arr)/sum(p_minus_arr)
#     p_minus_arr *= gamma
    
    p_minus_arr /= sum(p_minus_arr)
    p_plus_arr /= sum(p_plus_arr)
    
    transition_matrix = zeros(t.m+1,t.m+1)
    for row = 1:(t.m+1)
        for col = 1:(t.m+1)
            if row == col
                transition_matrix[row,col] = -p_minus_arr[row] - p_plus_arr[row]
            elseif row+1 == col
                transition_matrix[row,col] += p_minus_arr[row+1]
            elseif row-1 == col
                transition_matrix[row,col] += p_plus_arr[row-1]
            end
        end
    end
    return transition_matrix
end

function generate_transition_matrix(t::TwoLevel,alpha,beta,gamma)
    p_plus_arr = zeros(t.m+1)
    p_minus_arr = zeros(t.m+1)

    for idx = 1:(t.m + 1)
        j = idx-1
        p_plus_arr[idx] = p_j_plus(t,j,alpha)
        p_minus_arr[idx] = p_j_minus(t,j,beta)
    end

#     gamma = sum(p_plus_arr)/sum(p_minus_arr)
#     p_minus_arr *= gamma
    
    p_plus_arr *= gamma 
    
    transition_matrix = zeros(t.m+1,t.m+1)
    for row = 1:(t.m+1)
        for col = 1:(t.m+1)
            if row == col
                transition_matrix[row,col] = -p_minus_arr[row] - p_plus_arr[row]
            elseif row+1 == col
                transition_matrix[row,col] += p_minus_arr[row+1]
            elseif row-1 == col
                transition_matrix[row,col] += p_plus_arr[row-1]
            end
        end
    end
    return transition_matrix
end

function propagate_in_time(transition,arr,time = 1)
    dt = 0.05
    timesteps = round(Int,time/dt)
    change = similar(arr)
    total_change = 0.0
    infected_begin = get_frac_infected(arr,t.N)
    for i = 1:timesteps
        total_change += dt*(get_flow_up(transition,arr) - get_flow_down(transition,arr))/t.N
        arr += dt*transition*arr
        if i % round(Int,timesteps/10) == 1
            infected_curr = get_frac_infected(arr,t.N)
            println(infected_curr)
            println("total_change: $(total_change)")
            println("ratio: $(total_change/(infected_curr-infected_begin))")
            
#             println("change up: $(get_flow_up(transition,arr))")
#             println("change down: $(get_flow_down(transition,arr))")
            
        end
    end
    return arr
end


function get_frac_infected(arr::Array{Any,1},N::Int)
    y = collect(0:(length(arr)-1))
    return sum(arr .* y)/N
end

function get_frac_infected(arr,t::TwoLevelGraphs.TwoLevel)
    y = collect(0:(length(arr)-1))
    return sum(arr .* y)/t.N
end

function get_frac_infected(t::TwoLevelGraphs.TwoLevel)
    y = collect(0:t.m)
    return sum(t.a .* y)/t.N
end


function get_upper_diagonal(arr::Array{Float64,2})
    assert(size(arr,1) == size(arr,2))
    ret = zeros(size(arr,1)-1)
    for i = 1:length(ret)
        ret[i] = arr[i,i+1]
    end
    ret
end

function get_lower_diagonal(arr::Array{Float64,2})
    assert(size(arr,1) == size(arr,2))
    ret = zeros(size(arr,1)-1)
    for i = 1:length(ret)
        ret[i] = arr[i+1,i]
    end
    ret
end

function get_flow_down(transition,arr)
    return sum(get_upper_diagonal(transition) .* arr[2:end])
end


function get_flow_up(transition,arr)
    return sum(get_lower_diagonal(transition) .* arr[1:end-1])
end

function get_flow_vecs(transition)
    flow_down_vec = [[0];get_upper_diagonal(transition)]
    flow_up_vec = [get_lower_diagonal(transition);[0]]
    return flow_down_vec,flow_up_vec
end

function net_flow_vec(transition)
    flow_down_vec,flow_up_vec = get_flow_vecs(transition)
    net_flow_vec = (flow_up_vec - flow_down_vec)
    return net_flow_vec
end

function net_infected_vec(t)
    return collect(0:t.m)/t.N
end

function net_total_vec(t)
    return ones(t.m + 1)
end

function compute_gamma(transition,arr)
    return get_flow_down(transition,arr)/get_flow_up(transition,arr)
end

function get_stationary_distribution_from_matrix(t::TwoLevel,transition_matrix)
  equilibrium_distribution = nullspace(transition_matrix)
  if size(equilibrium_distribution)[2] > 1
    nsolutions = size(equilibrium_distribution)[2]
    mask = Array(Bool, nsolutions)
    for i = 1:nsolutions
      mask[i] = all(equilibrium_distribution[:,i] .>= 0) && any(equilibrium_distribution[:,i] .> 0.0)
    end
    equilibrium_distribution = equilibrium_distribution[:,mask]
    if size(equilibrium_distribution)[2] > 1
     println("PROBLEM: $(size(equilibrium_distribution)[2]) SOLUTIONS");
   end
  end
  equilibrium_distribution = equilibrium_distribution[:,1]
  equilibrium_distribution *= t.n/sum(equilibrium_distribution)
  return equilibrium_distribution
end

function binary_search_transition_matrix(f_out,y_target,x_initial=1.0)
    tol = 1e-4
    max_iter = 1000
    x_upper = guarantee_upper_bound(f_out,y_target,x_initial)
    x_lower = guarantee_lower_bound(f_out,y_target,x_initial)
    y_upper = f_out(x_upper) 
    y_lower = f_out(x_lower) 
    
    iter = 1
    while y_upper - y_lower > tol && iter < max_iter
        x_mid = (x_upper + x_lower)/2
        y_mid = f_out(x_mid)
        if y_mid > y_target
            x_upper = x_mid
            y_upper = y_mid
        elseif y_mid < y_target
            x_lower = x_mid
            y_lower = y_mid
        else
            return x_mid
        end
        iter +=1 
    end
    if iter >= max_iter
        println("WARNING: no convergence in transition matrix method. Aborting binary search.")
    end
    return (x_upper+x_lower)/2
end

function guarantee_upper_bound(f_out,y_target,x_0 = 1.0)
    while f_out(x_0) <= y_target
      x_0 *= 1.5
    end
    x_0
end

function guarantee_lower_bound(f_out,y_target,x_0 = 1.0)
    while f_out(x_0) >= y_target
        x_0 /= 1.5
    end
    x_0
end

function get_initial_equalizing_gamma(t,alpha,beta)
  transition = generate_transition_matrix(t,alpha,beta,1.0)
  gamma = sum(get_upper_diagonal(transition))/sum(get_lower_diagonal(transition))
  gamma
end
    

function get_y_out(t,alpha,beta,gamma)
    transition = generate_transition_matrix(t,alpha,beta,gamma)
    arr = get_stationary_distribution_from_matrix(t,transition)
    gamma_out = compute_gamma(transition,arr)
    y_out = get_frac_infected(arr,t)
    y_out
end

function get_stationary_distribution_nonlinear_theory(t,alpha,beta,y_desired)
    if y_desired == 0.0 || t.i == 0
      arr = zeros(t.a)
      arr[1] = t.n
      return arr 
    end
    f_out(x) = get_y_out(t,alpha,beta,x)
    gamma_init = get_initial_equalizing_gamma(t,alpha,beta)
    gamma = binary_search_transition_matrix(f_out,y_desired,gamma_init)
    transition = generate_transition_matrix(t,alpha,beta,gamma)
    arr = get_stationary_distribution_from_matrix(t,transition)
    arr
end

##################### Develop Effective y and y squred ####################
using Dierckx

function get_interpolations(t::TwoLevel,alpha,beta)
    y_min = 1.0/t.N

    dy = clamp(y_min,0.01,0.1)
    y_range = collect(y_min:dy:(1.0-y_min))
    interpolation_order = 2
    y_real_range = zeros(y_range)

    y_eff_range_inf = zeros(y_range)
    y_sq_eff_range_inf = zeros(y_range)

    y_eff_range_susc = zeros(y_range)
    y_sq_eff_range_susc = zeros(y_range)


    for (i,y_desired) in enumerate(y_range)
      set_y(t,y_desired)
      #accum = get_stationary_distribution(t.N,t.m,t.l,t.r,y_desired,((x,y) -> death_fn(x,y,beta)),((x,y) -> birth_fn(x,y,alpha)),500_000)
      accum = get_stationary_distribution_nonlinear_theory(t.N,t.m,t.l,t.r,y_desired,alpha,beta)
      t.a = accum
      y_eff_range_inf[i] = compute_mean_y_local(t,false)
      y_sq_eff_range_inf[i] = compute_mean_y_squared_local(t,false)
      y_eff_range_susc[i] = compute_mean_y_local(t,true)
      y_sq_eff_range_susc[i] = compute_mean_y_squared_local(t,true)
      y_real_range[i] = get_frac_infected(t) 
      #y_real_range[i] = t.i/t.N 
      if y_desired == 1.0
        y_eff_range_inf[i] = 1.0
        y_sq_eff_range_inf[i] = 1.0
        y_eff_range_susc[i] = 1.0
        y_sq_eff_range_susc[i] = 1.0
        y_real_range[i] = 1.0
        println(y_real_range[i])
      # elseif y_desired == 0.0
      #   y_eff_range_inf[i] =0.0 
      #   y_sq_eff_range_inf[i] =0.0 
      #   y_eff_range_susc[i] =0.0 
      #   y_sq_eff_range_susc[i] =0.0 
      #   y_real_range[i] =0.0 
      end

  
    end
    

    #yy = collect(0:0.01:1)

    y_inf_interp = Spline1D(y_real_range,y_eff_range_inf,k=interpolation_order,bc="extrapolate")
    y_susc_interp = Spline1D(y_real_range,y_eff_range_susc,k=interpolation_order,bc="extrapolate")
    y_sq_inf_interp = Spline1D(y_real_range,y_sq_eff_range_inf,k=interpolation_order,bc="extrapolate")
    y_sq_susc_interp = Spline1D(y_real_range,y_sq_eff_range_susc,k=interpolation_order,bc="extrapolate")
    #plot(y_real_range,y_eff_range_susc)
    # #plot(yy,y_susc_interp[yy])semilogx
    # println(y_range)
    # println(y_eff_range_susc)
    # println(y_sq_eff_range_susc)
    # plot(y_real_range,y_eff_range_inf,"b")
    # plot(y_real_range,y_eff_range_susc,"r")
    # plot(y_real_range,y_sq_eff_range_inf,"--b")
    # plot(y_real_range,y_sq_eff_range_susc,"--r")
    return y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp
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



################    GRAPH CONSTRUCTION   ####################
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
    for (i,clust_from) in enumerate(clusters)
        for (j,clust_to) in enumerate(clusters)
            if j > i
                for v in clust_from
                    for w in clust_to
                        push!(possible_edges,Pair(v,w))
                    end
                end
            end
        end
    end
    return sample(possible_edges,num_edges,replace=false)
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
