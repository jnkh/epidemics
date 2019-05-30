__precompile__()
module TwoLevelGraphs

using LightGraphs, Distributions, StatsBase, IM, NLsolve#,PyPlot

export TwoLevel, is_valid, get_num_infected, distribute_randomly, make_consistent,
TwoLevelGraph, get_clusters, make_two_level_random_graph,birth_fn,death_fn,
adjust_infecteds,get_stationary_distribution,p_j_plus,p_j_minus,
compute_mean_y_local,compute_mean_y_squared_local,set_y, get_interpolations,
get_stationary_distribution_nonlinear_theory,generate_transition_matrix,
get_frac_infected,get_s_effective_two_level,get_splus_effective_two_level,
get_s_birth_effective_two_level,get_s_death_effective_two_level,
get_s_effective_two_level_interp,get_splus_effective_two_level_interp,
generate_regular_two_level_graph,same_cluster,get_p_reach_theory,get_sparse_j_mask,
get_stationary_distribution_nlsolve_finite_size,get_community_graph_fixation_ratio,
get_fixation_MC,phase_transition_condition,get_t

struct TwoLevel
    a::Array{Number,1} #number communities with [idx] infected nodes
    N::Int #total number of nodes
    m::Int #number of nodes per community
    n::Int #number of communities
    i::Number #total number of infecteds
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
    if t.m == 1
      println("Only one node per subgraph, setting l = 0.")
      t.l = 0
    elseif t.m == t.N
      println("Only one subgraph, setting r = 0.")
      t.r = 0
    end
end

function set_y(t::TwoLevel,y_desired::AbstractFloat)
    adjust_infecteds(t,y_desired)
    make_consistent(t)
end    

function get_t(N::Int,k,m,l)
    r = k -l
    t = TwoLevel(Int(ceil(N/m)*m),m,l,r)
    return t
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

    birth_idx = StatsBase.sample(Weights(birth_probs))
    birth_at(t,birth_idx)

    #death
    for j = 1:t.m
        death_probs[j] = death_fn(t,j)
    end
    death_idx = StatsBase.sample(Weights(death_probs))
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



function hypergeometric_mean(N,n,k)
  # n = max(n,0.0)
  # k = max(k,0.0)
  #N = population size, n = successes, k = trials
  return k * n/N
end

function hypergeometric_variance(N,n,k)
  # n = max(n,0.0)
  # k = max(k,0.0)
  return k*n*(N-n)*(N-k)/(N*N*(N-1))
end

function hypergeometric_second_moment(N,n,k)
  return hypergeometric_variance(N,n,k) + hypergeometric_mean(N,n,k)^2
end

function get_internal_infecteds_mean(t::TwoLevel,j::Int,susceptible::Bool)
  if ~susceptible
    j = j-1
  end
  l = t.l
  m = t.m
  if m == 1
    return 0
  end
  return hypergeometric_mean(m-1,j,l)
end

function get_external_infecteds_mean(t::TwoLevel,j::Int,susceptible::Bool)
  i = t.i
  m = t.m
  r = t.r
  N = t.N
  if N == m
    return 0
  end
  return hypergeometric_mean(N-m,i-j,r)
end

function get_internal_infecteds_second_moment(t::TwoLevel,j::Int,susceptible::Bool)
  if ~susceptible
    j = j-1
  end
  l = t.l
  m = t.m
  if m == 1
    return 0
  end
  return hypergeometric_second_moment(m-1,j,l)
end

function get_external_infecteds_second_moment(t::TwoLevel,j::Int,susceptible::Bool)
  i = t.i
  m = t.m
  r = t.r
  N = t.N
  if N == m
    return 0
  end
  return hypergeometric_second_moment(N-m,i-j,r)
end



function get_y_local(t::TwoLevel,j::Int,susceptible::Bool)
    i_int_mean = get_internal_infecteds_mean(t,j,susceptible)
    i_ext_mean = get_external_infecteds_mean(t,j,susceptible)
    l = t.l
    r = t.r
    if t.m == 1
      l = 0
    elseif t.m == t.N
      r = 0
    end
    y_local = (i_int_mean + i_ext_mean)/(r + l)
  return y_local
end

function get_y_squared_local(t::TwoLevel,j::Int,susceptible::Bool)
  i_int_mean = get_internal_infecteds_mean(t,j,susceptible)
  i_ext_mean = get_external_infecteds_mean(t,j,susceptible)
  i_int_second_moment = get_internal_infecteds_second_moment(t,j,susceptible)
  i_ext_second_moment = get_external_infecteds_second_moment(t,j,susceptible)
  l = t.l
  r = t.r
  if t.m == 1
    l = 0
  elseif t.m == t.N
    r = 0
  end
  y_squared_local = (i_int_second_moment + i_ext_second_moment + 2*i_int_mean*i_ext_mean)/(r + l)^2
  return y_squared_local
end


# function get_y_external(t::TwoLevel,j::Int)
#   y_ext = (t.i - j)/(t.N - t.m)
#   if t.N == t.m
#     y_ext = 0.0
#   end

#   # if j > t.i
#   #   y_ext = 0.0
#   # end
#   return y_ext

# end

# function get_y_internal(t::TwoLevel,j::Int,susceptible::Bool)
#   #-1 in the denominator because we can't count ourself in the average!
#   if susceptible
#     y_int = j/(t.m - 1)
#   else
#     y_int = (j-1)/(t.m - 1)
#   end
#   if t.m == 1
#     y_int = 0.0
#   end
#   # if j > t.i
#   #   y_int =  0.0
#   # end
#   return y_int
# end

# function get_y_local(t::TwoLevel,j::Int,susceptible::Bool)
#     y_ext = get_y_external(t,j)
#     y_int = get_y_internal(t,j,susceptible)
#     l = t.l
#     r = t.r
#     if t.m == 1
#       l = 0
#     elseif t.m == t.N
#       r = 0
#     end
#     y_local = (l*y_int + r*y_ext)/(r + l)
#   return y_local
# end

# function get_y_squared_local(t::TwoLevel,j::Int,susceptible::Bool)
#   y_ext = get_y_external(t,j)
#   y_int = get_y_internal(t,j,susceptible)
#   l = t.l
#   r = t.r
#   if t.m == 1
#     l = 0
#   elseif t.m == t.N
#     r = 0
#   end
#   y_squared_local = (r*y_ext*((r - 1)* y_ext + 1) +
# l*y_int*((l - 1)*y_int + 1) + 2*l* r* y_ext*y_int)/(r + l)^2

#   return y_squared_local
# end


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

#i_ext ~Hypergeometric(population=N-m,successes=yN-j,trials=r)
#i_int ~Hypergeometric(population=m-1,successes=j,trials=l) (j-1 if infected)

function get_y_times_theta_local(t::TwoLevel,j::Int,susceptible::Bool,y_n::Real)
  l = t.l
  r = t.r
  i = t.i
  if t.m == 1
    l = 0
  elseif t.m == t.N
    r = 0
  end
  d_i_ext = Hypergeometric(i - j,(N-m) - (i-j),r)#successes,failures,trials
  if susceptible
    d_i_int = Hypergeometric(j,m-1-j,l)
  else
    d_i_int = Hypergeometric((j-1),m-1-(j-1),l)
  end
  accum = 0
  p_i_ext = pdf(d_i_ext)
  p_i_int = pdf(d_i_int)
  for (i_r_idx,i_r) in enumerate(minimum(d_i_ext):maximum(d_i_ext))
    for (i_l_idx,i_l) in enumerate(minimum(d_i_int):maximum(d_i_int))
      y_curr = get_y_from_ir_il(i_r,i_l,r,l)
      if y_curr >= y_n
        accum += p_i_ext(i_r_idx)*p_i_int(i_l_idx)*y_curr
      end
    end
  end
  return accum
end


function get_y_from_ir_il(i_r,i_l,r,l)
  return (i_r + i_l)/(r + l)
end



function compute_mean_quantity_local(t::TwoLevel,susceptible::Bool,get_quantity::Function)
  y_local = 0.
  weight = 0.

  for j = 0:t.m
    curr_weight = get_number_weight(t,j,susceptible)
    weight += curr_weight
    y_local += curr_weight*get_quantity(t,j,susceptible)
  end
  if weight == 0.0 return 0.0 end
  return y_local/weight
end




function compute_mean_y_local(t::TwoLevel,susceptible::Bool)
  y_local = 0.
  weight = 0.

  weights = zeros(t.a)
  y_locals = zeros(t.a)

  for j = 0:t.m
    curr_weight = get_number_weight(t,j,susceptible)
    weight += curr_weight
    y_local += curr_weight*get_y_local(t,j,susceptible)

    weights[j+1] = curr_weight
    y_locals[j+1] = get_y_local(t,j,susceptible)

  end
  if weight == 0.0 return 0.0 end


  # if susceptible
  #  pygui(true)
  #  ion()
  #  figure(7)
  #  semilogy(t.a)
  #  figure(8)
  #  semilogy(weights)
  #  figure(9)
  #  plot(y_locals)
  # end
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
  if y_desired < 1/t.N
    t.a *= 0
    t.a[2] = y_desired*t.N
    t.a[1] = t.n - sum(t.a)
    return
  end

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
    @assert(is_valid(t))
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
    @assert(is_valid(t))
    return get_stationary_distribution_theory(t,alpha,beta)
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
    # println(transition_matrix)
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
    @assert(size(arr,1) == size(arr,2))
    ret = zeros(size(arr,1)-1)
    for i = 1:length(ret)
        ret[i] = arr[i,i+1]
    end
    ret
end

function get_lower_diagonal(arr::Array{Float64,2})
    @assert(size(arr,1) == size(arr,2))
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

#This function produces a mask that is true only if the value of j in
#question is actually "allowed". For example, if t.i = 10 (i.e. there
#are 10 total infecteds), but j = 11, then that value of j is not allowed.
#Similarly, if t.N - t.i = 10 and t.m - j = 11 (i.e. there are only
# 10 susceptibles on the graph but a subgraph would have 11, this is
# also not allowed)
function get_sparse_j_mask(t)
  eps = 1+1e-6
  js = collect(0:t.m)
  mask = falses(js)
  for j in js
    mask[j+1] = (j <= t.i+eps) && ((t.m - j) <= (t.N - t.i) + eps)
  end
  # if !any(mask[2:end])
  #   mask[2] = true
  # elseif !any(mask[1:end-1])
  #   mask[end-1] = true
  # end
  return mask
end



function apply_finite_size_effects(t,equilibrium_distribution)
  mask = get_sparse_j_mask(t)
  # if sum(mask) < length(mask)
    # println("reducing solution space from $(length(mask)) to $(sum(mask))")
    # println("infecteds: $(t.i), susceptibles: $(t.N - t.i)")
    # println("j: $j t.i: $(t.i), $(t.m -j)")
  # end
  equilibrium_distribution[.~mask] = 0.0
  equilibrium_distribution *= t.n/sum(equilibrium_distribution)
  return equilibrium_distribution 
end




function get_stationary_distribution_from_matrix(t::TwoLevel,transition_matrix)
  success = true
  eps = 1e-16
  equilibrium_distribution = nullspace(transition_matrix)
  # equilibrium_distribution = transition_matrix \ zeros(size(transition_matrix)[1])
  # equilibrium_distribution = reshape(equilibrium_distribution,(length(equilibrium_distribution),1))
  nsolutions = size(equilibrium_distribution)[2]
  for i = 1:nsolutions
    equilibrium_distribution[:,i] *= t.n/sum(equilibrium_distribution[:,i])
  end

  # println(equilibrium_distribution)
  equilibrium_distribution = clamp.(equilibrium_distribution,eps,Inf)

  if nsolutions > 1
    println("$(size(equilibrium_distribution)[2]) solutions")
    # println("PROBLEM: $(size(equilibrium_distribution)[2]) SOLUTIONS");
    mask = Array{Bool}(nsolutions)
    for i = 1:nsolutions
      mask[i] = all(equilibrium_distribution[:,i] .>= 0) && any(equilibrium_distribution[:,i] .> 0.0)
    end
    if sum(mask) != 1
      success = false
    else
     equilibrium_distribution = equilibrium_distribution[:,mask]
    end


    # if !any(mask)
    #  success = false 
    # else
    #  equilibrium_distribution = equilibrium_distribution[:,mask]
    # end

  end
  equilibrium_distribution = equilibrium_distribution[:,1]

  equilibrium_distribution *= t.n/sum(equilibrium_distribution)
  return equilibrium_distribution,success
end

function binary_search_transition_matrix(f_out,y_target,x_initial=1.0)
    tol = 1e-8
    max_iter = 1000
    # x_upper = guarantee_upper_bound(f_out,y_target,x_initial)
    # x_lower = guarantee_lower_bound(f_out,y_target,x_initial)

    x_lower,x_upper = find_valid_initial_value(f_out,y_target)
    y_upper = f_out(x_upper)[1]
    y_lower = f_out(x_lower)[1]

    
    iter = 1
    while y_upper - y_lower > tol && iter < max_iter
        x_mid = (x_upper + x_lower)/2
        y_mid,success = f_out(x_mid)
        if ! success
          println("Unstable transition matrix solution, aborting.")
          println("x_mid: $(x_mid), x_upper: $(x_lower), x_upper: $(x_upper)")
          return x_mid
        end
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
        println("WARNING: no convergence in transition matrix method. Aborting binary search with error $(y_upper - y_lower).")
    end
    # println("$(iter) binary search iterations.")
    return (x_upper+x_lower)/2
end

function guarantee_upper_bound(f_out,y_target,x_0 = 1.0)
  # while ! f_out(x_0)[2]
  #   println("initial x_0 too high for convergence")
  #   x_0 /= 2
  # end

  while f_out(x_0)[1] <= y_target
    x_0 *= 1.4
  end
  x_0
end

function guarantee_lower_bound(f_out,y_target,x_0 = 1.0)
  # while ! f_out(x_0)[2]
  #   println("initial x_0 too low for convergence")
  #   x_0 *= 2
  # end
  while f_out(x_0)[1] >= y_target
    x_0 /= 1.4
  end
  x_0
end



function find_valid_initial_value(f_out,y_desired)
  x_range = logspace(-6,6,100)
  y_range = zeros(x_range)
  success_range = falses(x_range)


  for (i,x) in enumerate(x_range)
    y,success = f_out(x)
    y_range[i] = y
    success_range[i] = success && y >= 0
  end

  # pygui(true)
  # ion()
  # figure(1)
  # loglog(x_range[success_range],y_range[success_range])
  # loglog(x_range[~success_range],y_range[~success_range],"o")
  # axhline(y_desired,linestyle="--",color="k")
  lower = (y_range .< y_desired) .& success_range
  higher = (y_range .> y_desired) .& success_range
  if !any(lower) || !any(higher)
    println("PROBLEM: No stable value for gamma found.")
    return 0.5,1.5
  end

  x_low_ind = findlast(lower)
  x_high_ind = findfirst(higher)

  if x_high_ind <= x_low_ind
    println("PROBLEM: unordered pair.")
    return 0.5,1.5
  end

  @assert(x_high_ind > x_low_ind)
  x_lower = x_range[x_low_ind]
  x_higher = x_range[x_high_ind]

  # println("Lower: $(x_lower) upper: $(x_higher). Mid: $((x_lower + x_higher)/2)")
  # println("f_out(x_mid): $(f_out((x_higher + x_lower)/2))")
  # println("success: $(sum(success_range))/$(length(success_range))")
  return x_lower,x_higher 
end





function get_initial_equalizing_gamma(t,alpha,beta)
  transition = generate_transition_matrix(t,alpha,beta,1.0)
  gamma = sum(get_upper_diagonal(transition))/sum(get_lower_diagonal(transition))
  gamma
end
    

function get_y_out(t,alpha,beta,gamma)
    transition = generate_transition_matrix(t,alpha,beta,gamma)
    arr,success = get_stationary_distribution_from_matrix(t,transition)
    gamma_out = compute_gamma(transition,arr)
    y_out = get_frac_infected(arr,t)
    y_out,success
end

function get_stationary_distribution_nonlinear_theory(t,alpha,beta,y_desired,apply_finite_size=true)
    if y_desired == 0.0 #|| t.i == 0
      arr = zeros(t.a)
      arr[1] = t.n
      return arr 
    end
    f_out(x) = get_y_out(t,alpha,beta,x)
    # gamma_init = get_initial_equalizing_gamma(t,alpha,beta)
    # gamma_init = find_valid_initial_value(f_out,y_desired)
    gamma_init = 1.0
    gamma = binary_search_transition_matrix(f_out,y_desired,gamma_init)
    transition = generate_transition_matrix(t,alpha,beta,gamma)
    arr,success = get_stationary_distribution_from_matrix(t,transition)
    if apply_finite_size
      arr = apply_finite_size_effects(t,arr)
    end
    # pygui(true)
    # ion()
    # figure(2)
    # semilogy(arr)
    # for j in 0:t.m
    #   if j > t.i
    #     arr[j] = 0.0
    #   end
    # end
    arr
end

function get_stationary_distribution_nonlinear_theory(N::Int,m::Int,l::Int,r::Int,y_desired::AbstractFloat,alpha::AbstractFloat,beta::AbstractFloat,apply_finite_size=true)
    t = TwoLevel(N,m,l,r)
    #distribute_randomly(t,n)
    adjust_infecteds(t,y_desired)
    t.i = y_desired*N
    # make_consistent(t)
    # @assert(is_valid(t))
    return get_stationary_distribution_nonlinear_theory(t,alpha,beta,y_desired,apply_finite_size)
end


################################
#######NLSOLVE APPROACH#########
################################

function f_finite_size!(fvec,x,t,alpha,beta)
    mask = get_sparse_j_mask(t) 
    gamma = x[end-1]
    a = x[1:end-2]
    m = generate_transition_matrix_finite_size(t,alpha,beta,gamma)
    fvec[1:end-2] = m*a
    arr = x[1]*zeros(length(mask)) #set type implicitly to x so autodiff works
    arr[mask] = a
    fvec[end-1] = get_frac_infected(arr,t)-t.i/t.N
    fvec[end] = sum(a) - t.n
end

function get_stationary_distribution_nlsolve_finite_size(N::Int,m::Int,l::Int,r::Int,y_desired::AbstractFloat,alpha::AbstractFloat,beta::AbstractFloat,apply_finite_size=true)
    if ~apply_finite_size
      println("Finite size effects always applied with nlsolve, ignoring 'apply_finite_size=false' option")
    end
    t = TwoLevel(N,m,l,r)
    #distribute_randomly(t,n)
    # adjust_infecteds(t,y_desired)
    t.i = y_desired*N
    return get_stationary_distribution_nlsolve_finite_size(t,alpha,beta)
end


function get_stationary_distribution_nlsolve_finite_size(t,alpha,beta)
    mask = get_sparse_j_mask(t) 
    dim = sum(mask)
    fn(x_res,x) = f_finite_size!(x_res,x,t,alpha,beta)
    res = nlsolve(fn,ones(dim+2), autodiff = :forward,ftol=1e-12,iterations=10_000)
    
    if ~res.f_converged
        println(res)
        println("Problem, didn't converge")
    end
    answer = zeros(length(mask))
    answer[mask] = res.zero[1:end-2]
    return answer#,res.zero[end-1] #this is gamma
end

function generate_transition_matrix_finite_size(t::TwoLevel,alpha,beta,gamma)
    p_plus_arr = zeros(t.m+1)
    p_minus_arr = zeros(t.m+1)
    mask = get_sparse_j_mask(t)

    for idx = 1:(t.m + 1)
        j = idx-1
        p_plus_arr[idx] = p_j_plus(t,j,alpha)
        p_minus_arr[idx] = p_j_minus(t,j,beta)
    end
    
    #can't allow up transitions there
    if findlast(mask) < length(mask)
        p_plus_arr[findlast(mask)] = 0
    end
   
    #can't allow down transitions
    if findfirst(mask) > 1 
        p_minus_arr[findfirst(mask)] = 0
    end
    
    p_plus_arr = p_plus_arr[mask]
    p_minus_arr = p_minus_arr[mask]
    dim = sum(mask)

#     gamma = sum(p_plus_arr)/sum(p_minus_arr)
#     p_minus_arr *= gamma
    
    p_plus_arr *= gamma 
    
    transition_matrix = p_plus_arr[1]*zeros(dim,dim) #set type to same as p_plus_arr so autodiff works
    for row = 1:(dim)
        for col = 1:(dim)
            if row == col
                transition_matrix[row,col] = -p_minus_arr[row] - p_plus_arr[row]
            elseif row+1 == col
                transition_matrix[row,col] += p_minus_arr[row+1]
            elseif row-1 == col
                transition_matrix[row,col] += p_plus_arr[row-1]
            end
        end
    end
    # println(transition_matrix)
    return transition_matrix
end


##################### Develop Effective y and y squred ####################

#if this quantity is > 1, the fixation probability should not decrease with large N
function phase_transition_condition(t,alpha,beta,apply_finite_size=true)
    y_s = t.m/t.N
    y_n = beta/alpha
    y_desired = y_s
    accum = get_stationary_distribution_nlsolve_finite_size(t.N,t.m,t.l,t.r,y_desired,alpha,beta,apply_finite_size)
    t.a = accum
    y_real = get_frac_infected(t)
    t.i = y_real*t.N
    y_susc_sq = compute_mean_y_squared_local(t,true)
    y_inf = compute_mean_y_local(t,false)
    return y_susc_sq*(1 - y_s)/(y_s*y_n*(1-y_inf))
end
    
using Dierckx

function logitspace(eps,num_points)
  @assert( 0<eps<1)
  logit(x) = log(x/(1-x))
  logist(x) = 1/(e^(-x) + 1)
  return logist.(linspace(logit(eps),logit(1-eps),num_points))
end


function get_interpolations(t::TwoLevel,alpha,beta,apply_finite_size=true,num_points=200)
    dy = 1.0/t.N
    y_min = dy/10



    # dy0 = clamp(y_min,1e-5,0.01)
    # dy2 = clamp(y_min,0.01,0.1)
    # y_range = vcat( collect(y_min:y_min:4*dy),collect(5*dy:dy:0.1) , collect(0.1+dy:dy2:(1.0-dy)) )
    # y_range = logspace(log10(y_min),log10(1-y_min),200)
    y_range = linspace(y_min,1-y_min,num_points)
    if t.N > num_points #distribute points logistically between 0 and 1
      y_range = logitspace(y_min,num_points)
    end
    interpolation_order = 1
    y_real_range = zeros(y_range)

    y_eff_range_inf = zeros(y_range)
    y_sq_eff_range_inf = zeros(y_range)

    y_eff_range_susc = zeros(y_range)
    y_sq_eff_range_susc = zeros(y_range)

    for (i,y_desired) in enumerate(y_range)
      # set_y(t,y_desired)
      #accum = get_stationary_distribution(t.N,t.m,t.l,t.r,y_desired,((x,y) -> death_fn(x,y,beta)),((x,y) -> birth_fn(x,y,alpha)),500_000)
      # accum = get_stationary_distribution_nonlinear_theory(t.N,t.m,t.l,t.r,y_desired,alpha,beta,apply_finite_size)
      accum = get_stationary_distribution_nlsolve_finite_size(t.N,t.m,t.l,t.r,y_desired,alpha,beta,apply_finite_size)
      t.a = accum
      y_real = get_frac_infected(t)
      # y_real = y_desired
      t.i = y_real*t.N
      y_eff_range_inf[i] = compute_mean_y_local(t,false)
      y_sq_eff_range_inf[i] = compute_mean_y_squared_local(t,false)
      y_eff_range_susc[i] = compute_mean_y_local(t,true)
      y_sq_eff_range_susc[i] = compute_mean_y_squared_local(t,true)
      y_real_range[i] = y_real 
      # y_real_range[i] = t.i/t.N 
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
    

    s_eff_range = get_s_effective_two_level(y_real_range,y_eff_range_susc,y_sq_eff_range_susc,y_eff_range_inf,y_sq_eff_range_inf,alpha,beta)
    splus_eff_range = get_splus_effective_two_level(y_real_range,y_eff_range_susc,y_sq_eff_range_susc,y_eff_range_inf,y_sq_eff_range_inf,alpha,beta)

    s_birth_range = get_s_birth_effective_two_level(y_real_range,y_eff_range_susc,y_sq_eff_range_susc,alpha) 
    s_death_range = get_s_death_effective_two_level(y_real_range,y_eff_range_inf,beta) 
    #yy = collect(0:0.01:1)

    y_inf_interp = Spline1D(y_real_range,y_eff_range_inf,k=interpolation_order,bc="extrapolate")
    y_susc_interp = Spline1D(y_real_range,y_eff_range_susc,k=interpolation_order,bc="extrapolate")
    y_sq_inf_interp = Spline1D(y_real_range,y_sq_eff_range_inf,k=interpolation_order,bc="extrapolate")
    y_sq_susc_interp = Spline1D(y_real_range,y_sq_eff_range_susc,k=interpolation_order,bc="extrapolate")

    s_birth_interp = Spline1D(y_real_range,s_birth_range,k=interpolation_order,bc="extrapolate")
    s_death_interp = Spline1D(y_real_range,s_death_range,k=interpolation_order,bc="extrapolate")
    s_interp = Spline1D(y_real_range,s_eff_range,k=interpolation_order,bc="extrapolate")
    splus_interp = Spline1D(y_real_range,splus_eff_range,k=interpolation_order,bc="extrapolate")


    # pygui(true)
    # ion()
    # figure(4)
    # plot(y_real_range,y_real_range,"-k")
    # plot(y_real_range,y_real_range.^2,"--k")
    # plot(y_real_range,y_eff_range_inf,"b")
    # plot(y_real_range,y_eff_range_susc,"r")
    # plot(y_real_range,y_sq_eff_range_inf,"--b")
    # plot(y_real_range,y_sq_eff_range_susc,"--r")

    # figure(5)
    # plot(y_real_range,y_birth_range-y_death_range,"-k")
    # plot(y_real_range,(y_real_range*alpha-beta),"--k")

    # figure(6)
    # plot(y_real_range,y_real_range.*s_birth_range,"-r")
    # plot(y_real_range,(1 - y_real_range).*s_death_range,"-b")
    # # #plot(yy,y_susc_interp[yy])semilogx
    # # println(y_range)
    # # println(y_eff_range_susc)
    # # println(y_sq_eff_range_susc)

    return y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp,s_birth_interp,s_death_interp,s_interp,splus_interp
end



struct TwoLevelGraph
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


function get_s_effective_two_level(y,y_susc,y_sq_susc,y_inf,y_sq_inf,alpha::Float64,beta::Float64)
    return get_s_birth_effective_two_level(y,y_susc,y_sq_susc,alpha) - get_s_death_effective_two_level(y,y_inf,beta)
end

function get_splus_effective_two_level(y,y_susc,y_sq_susc,y_inf,y_sq_inf,alpha::Float64,beta::Float64)
    return get_s_birth_effective_two_level(y,y_susc,y_sq_susc,alpha) + get_s_death_effective_two_level(y,y_inf,beta)
end

function get_s_birth_effective_two_level(y::Array,y_susc,y_sq_susc,alpha::Float64)
    ret = 1.0/y.*(y_susc + alpha .* y_sq_susc)
    ret[y .== 0] = 1.0
    return ret
end

function get_s_birth_effective_two_level(y::Number,y_susc,y_sq_susc,alpha::Float64)
    if y == 0 return 1.0 end
    return 1.0/y.*(y_susc + alpha .* y_sq_susc)
end

function get_s_death_effective_two_level(y::Array,y_inf,beta::Float64)
  ret = 1.0 ./ (1-y).*(1 - y_inf).*(1 + beta)
  ret[y .== 1.0] = (1 + beta)
  return ret
end

function get_s_death_effective_two_level(y::Number,y_inf,beta::Float64)
  if y == 1.0 return (1 + beta) end
  return 1.0 ./ (1-y).*(1 - y_inf).*(1 + beta)
end


    
function get_splus_effective_two_level_interp(yy,alpha::Float64,beta::Float64,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)
    y_inf = evaluate(y_inf_interp,yy)
    y_susc = evaluate(y_susc_interp,yy)
    y_sq_inf = evaluate(y_sq_inf_interp,yy)
    y_sq_susc = evaluate(y_sq_susc_interp,yy)
    return get_splus_effective_two_level(yy,y_susc,y_sq_susc,y_inf,y_sq_inf,alpha,beta)
end


function get_s_effective_two_level_interp(yy,alpha::Float64,beta::Float64,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)
    y_inf = evaluate(y_inf_interp,yy)
    y_susc = evaluate(y_susc_interp,yy)
    y_sq_inf = evaluate(y_sq_inf_interp,yy)
    y_sq_susc = evaluate(y_sq_susc_interp,yy)
    return get_s_effective_two_level(yy,y_susc,y_sq_susc,y_inf,y_sq_inf,alpha,beta)
end


function get_s_birth_effective_two_level_interp(yy,alpha::Float64,y_susc_interp,y_sq_susc_interp)
    y_susc = evaluate(y_susc_interp,yy)
    y_sq_susc = evaluate(y_sq_susc_interp,yy)
    return get_s_birth_effective_two_level(yy,y_susc,y_sq_susc,alpha)
end

function get_s_death_effective_two_level_interp(yy,beta::Float64,y_inf_interp)
    y_inf = evaluate(y_inf_interp,yy)
    return get_s_death_effective_two_level(yy,y_inf,beta)
end


function get_p_reach_theory(t,alpha,beta,N,apply_finite_size=true,num_points=1000)
    y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp,
    s_birth_interp,s_death_interp,s_interp,splus_interp = get_interpolations(t,alpha,beta,apply_finite_size,num_points)

    # s(x) = get_s_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)
    # splus(x) = get_splus_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)

    s(x) = evaluate(s_interp,x) 
    splus(x) = evaluate(splus_interp,x) 

    s_actual(x) = 2*s(x)./splus(x)

    xx = logspace(log10(1/N),0,num_points) 
    pp = P_reach_fast(s,splus,N,1/N,xx)
    return xx,pp,s_actual
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
                        push!(possible_edges,Edge(v,w))
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
#             push!(possible_edges,Edge(i,j))
#         end
#     end
#     super_edges = sample(possible_edges,num_edges)
#     edges = []
#     for se in super_edges
#         e = Edge(sample(clusters[se[1]]),sample(clusters[se[2]]))
#         push!(edges,e)
#     end
#    return edges
#end


function num_internal_edges(cluster::Array{Int,1},t::TwoLevel)
    num_desired = Int(length(cluster)*t.l/2)
    num_trials = Int(length(cluster)*(length(cluster) - 1)/2)
    if num_trials == 0 || num_desired == 0
      return 0
    end
    total_edges = rand(Binomial(num_trials,num_desired/num_trials))
    #total_edges = num_desired #size of cluster times number of internal edges per node
    return total_edges
end

function num_external_edges(clusters::Array{Array{Int,1},1},t::TwoLevel)
    num_desired = Int(t.N*t.r/2)
    num_trials = Int(t.N*(t.N-1)/2)
    if num_trials == 0 || num_desired == 0
      return 0
    end
    total_edges = rand(Binomial(num_trials,num_desired/num_trials))
    #total_edges = num_desired
    return total_edges
end

function get_edges_for_subgraph(cluster::Array{Int,1},num_edges::Int)
    g = LightGraphs.erdos_renyi(length(cluster),num_edges)
    edges = collect(LightGraphs.edges(g))
    new_edges = copy(edges)
    for (i,e) in enumerate(edges)
        new_edges[i] = Edge(cluster[e.src],cluster[e.dst])
    end
    return new_edges
end

function get_edges_for_subgraph(cluster::Array{Int,1},p_edges::Float64)
    g = LightGraphs.erdos_renyi(length(cluster),p_edges)
    edges = collect(LightGraphs.edges(g))
    new_edges = copy(edges)
    for (i,e) in enumerate(edges)
        new_edges[i] = Edge(cluster[e.src],cluster[e.dst])
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


##############Create Regular Two-Level Graph############

function generate_single_regular_two_level_graph(t::TwoLevel)
    sub_edges = get_regular_edges_for_subgraph(t)
    super_edges = get_regular_edges_for_supergraph(t)
    all_edges = vcat(sub_edges,super_edges)

    @assert(length(sub_edges) == t.N*t.l/2)
    @assert(length(super_edges) == t.N*t.r/2)
    clusters = get_clusters(t)

    nodes = vcat(clusters...)
    N = length(nodes)
    G = LightGraphs.Graph(N)
    for e in all_edges
        add_edge!(G,e)
    end
    return G
end

function generate_regular_two_level_graph(t::TwoLevel)
    max_resample = 100
    resample = 0
    while true
        G = generate_single_regular_two_level_graph(t)
        if (length(collect(LightGraphs.edges(G))) == t.N*(t.l+t.r)/2)
            if resample > 0
                println("resampled graph $resample times.")
            end
            return G
        end
        resample += 1
    end
end

function remove_duplicate_superedges(edges,clusters)
    duplicates = get_duplicates(edges)
    # println("removing $(length(duplicates)) duplicate edges")
    unique_edges = unique(edges)
    for dup_edge in duplicates
        unique_edges = rewire_edges(unique_edges,dup_edge,clusters)
    end
    @assert(length(get_duplicates(unique_edges))==0)
    return unique_edges
end

function rewire_edges(unique_edges,dup_edge,clusters)
    edge = dup_edge
    rand_idx = 0
    while true
        rand_idx = rand(1:length(unique_edges))
        edge = unique_edges[rand_idx]
        if valid_swap(dup_edge,edge,unique_edges,clusters)
            break
        end
    end
    e1,e2 = swap_edges(edge,dup_edge)
    unique_edges[rand_idx] = e1
    push!(unique_edges,e2)
    return unique_edges
end

function swap_edges(e1,e2)
    e3 = Edge(e1.src,e2.dst)
    e4 = Edge(e2.src,e1.dst)
    return e3,e4
end


function valid_swap(e1,e2,unique_edges,clusters)
    if e1.src == e2.src || e1.dst == e2.dst return false end
    if e1.src == e2.dst || e1.dst == e2.src return false end
    if same_cluster(e1.src,e2.dst,clusters) || same_cluster(e1.dst,e2.src,clusters) return false end
    e3,e4 = swap_edges(e1,e2)
    if e3 in unique_edges || e4 in unique_edges || reverse(e3) in unique_edges || reverse(e4) in unique_edges
      return false
    end
    return true
end


function get_duplicates(arr)
    duplicates = []
    arr = sort(arr,by=x -> Pair(x.src,x.dst))
    for i in 1:length(arr)-1
        curr = arr[i]
        next = arr[i+1]
        if next == curr
            push!(duplicates,next)
        end
    end
    duplicates
end   

function get_regular_edges_for_subgraph(t)
    clusters = get_clusters(t)
    edges = []
    for cluster in clusters
        edges = vcat(edges,get_regular_edges_for_subgraph(cluster,t.l))
    end
    return edges
end

function get_regular_edges_for_subgraph(cluster,l)
    N = length(cluster)
    temp_edges = get_regular_edges(N,l)
    return remap_edges(temp_edges,cluster)
end

function remap_edges(edges,cluster)
    new_edges = copy(edges)
    for (i,e) in enumerate(edges)
        new_edges[i] = Edge(cluster[e.src],cluster[e.dst])
    end
    return new_edges
end

#There are N total nodes, divided onto n communities. Each community as m = N/m nodes. Each node has exactly r edges.
#Thus each community has exactly m*r edges incoming. 
function get_regular_super_edges_for_supergraph(t::TwoLevel)
    N = t.N
    m = t.m
    n = t.n
    r = t.r
    @assert(n*m == N)
    max_increment = min(n-1,m*r)
    tot_neighbors = m*r
    curr_neighbors = 0
    edges = []
    while true
        new_curr_neighbors = curr_neighbors + max_increment
        if new_curr_neighbors > tot_neighbors
            break
        end
        edges = vcat(edges,get_regular_edges(n,max_increment))
        curr_neighbors = new_curr_neighbors
    end
    if curr_neighbors < tot_neighbors
        edges = vcat(edges,get_regular_edges(n,tot_neighbors - curr_neighbors))
    end
    @assert(length(edges) == N*r/2)
    return edges
end

function get_regular_edges_for_supergraph(t::TwoLevel)
    super_edges = get_regular_super_edges_for_supergraph(t)
    clusters = get_clusters(t)
    edges = distribute_super_edges(super_edges,clusters,t.r)
end

function get_regular_edges(N,k)
    temp_graph = LightGraphs.random_regular_graph(N,k)
    temp_edges = collect(LightGraphs.edges(temp_graph))
    return temp_edges
end


#each cluster should have exactly m*r edges incoming. The idea is to distribute the subnodes randomly among these edges
#duplicate the indeces of the subnodes r times. shuffle them. Then replace each cluster index with the next available subnode from that cluster
function distribute_super_edges(super_edges,clusters,r)
    super_edges = shuffle(super_edges)
    new_edges = copy(super_edges)
    for (clust_idx,cluster) in enumerate(clusters)
        cluster_indeces = shuffle(repmat(cluster,r))
        for (edge_idx,edge) in enumerate(super_edges)
            if edge.src == clust_idx
                new_edge = Edge(pop!(cluster_indeces),new_edges[edge_idx].dst)
                new_edges[edge_idx] = new_edge
            elseif edge.dst == clust_idx
                new_edge = Edge(new_edges[edge_idx].src,pop!(cluster_indeces))
                new_edges[edge_idx] = new_edge
            end
        end
        @assert(length(cluster_indeces) == 0)
    end
    new_edges = remove_duplicate_superedges(new_edges,clusters)
    return new_edges
end
        
        
function same_cluster(n1,n2,clusters)
    for cluster in clusters
        if n1 in cluster
            if n2 in cluster
                return true
            else
                return false
            end
        end
    end
end 




####Forward/Backward fixation ratio


function get_alpha_beta(N,c_r,y_n)
    n_n = Int(N*y_n)#y_n*N
    beta = get_beta(N,c_r,n_n)#4.0/(c_r*n_n)
    alpha = get_alpha(N,c_r,n_n)#(N*beta)/n_n
    return alpha,beta
end
    

function get_t(N,k,m,l)
    r = k -l
    t = TwoLevel(Int(ceil(N/m)*m),m,l,r)
    return t
end

function get_community_graph_fixation_ratio(t,alpha,beta)
    return get_fixation_MC(t,alpha,beta)/get_fixation_MC(t,alpha,beta,true)
#     yy,pp,pp_reverse = get_single_community_fixation(t,alpha,beta,false)
#     return pp[end]/pp_reverse[end]
end

function get_theory_sim_fixation_ratio(t,alpha,beta,num_trials,use_model = false)
    success = true
    if use_model
        yys,pps,_ = get_p_reach_theory(t,alpha,beta,N,true,200)
    else
        yys,pps = get_simulation_yy_pp(t,alpha,beta,num_trials)
    end
#     yy,pp,pp_reverse = get_single_community_fixation(t,alpha,beta)
#     pfixth = pp[end]
    pfixth = get_fixation_MC(t,alpha,beta)
    pfix = pps[end]
    if yys[end] < 0.99
        println("simulation didn't reach 1.0")
        println("theory suggests at least $(1/pfixth)")
        success = false
        pfix = 0
#         return -1
    end
    figure(1)
    loglog(yys,pps)
    return pfix/pfixth, success,pfix,pfixth
end


using Dierckx

function get_simulation_yy_pp(t,alpha,beta,num_trials=100)
    im_normal = InfectionModel(x -> 1 + alpha*x , x -> 1 + beta);
    fixation_threshold = 1.0

    verbose = false
    ###Set to true if we want by-node information on infecteds (much more data!)
    carry_by_node_information = false
    graph_model = true
    in_parallel = true
    T = generate_regular_two_level_graph(t)
    graph_fn = () -> T 
    graph_data = TwoLevelGraph(LightGraphs.Graph(),t,get_clusters(t))
    graph_information = GraphInformation(graph_fn,Graph(),carry_by_node_information,graph_data)

    # @time runs = run_epidemics_parallel(num_trials,() -> run_epidemic_graph(N,im_normal,graph_information,fixation_threshold),in_parallel);
    # yy,pp = get_p_reach(runs)
    # yy /= N
    @time runs = run_epidemics_parallel(num_trials,() -> run_epidemic_graph_gillespie(t.N,im_normal,graph_information,fixation_threshold),in_parallel);
    yys,pps = get_p_reach(runs)
    yys /= N;
    return yys,pps
end



function get_dmp_dmm(t,j,alpha,beta)
    i_orig = t.i
    t.i = i_orig + j 
    z_s = TwoLevelGraphs.get_y_local(t,j,true)
    z_sq_s = TwoLevelGraphs.get_y_squared_local(t,j,true)
    z_i = TwoLevelGraphs.get_y_local(t,j,false)
#     println("j: $j")
#     println("z_s: $(z_s)")
#     println("z_i: $(z_i)")
#     println()

    y_i = j/t.m
    
    
    dmp = (1-y_i)*(z_s + alpha*z_sq_s)
    dmm = (y_i)*(1-z_i)*(1 + beta)
    
    if j == 0 || j == t.m
        dmm = 0 #j == 0
        dmp = 0 #j ==t.m
    end

    t.i = i_orig
    return dmp,dmm
end

function get_fixation_MC(t,alpha,beta,reverse=false)
    y = 0.0
    t.i = t.N*y
    m = t.m

    j_range = collect(0:t.m)
    y_range = j_range/t.m
    s_arr = zeros(Float64,size(j_range))
    dmp_arr = zeros(Float64,size(j_range))
    dmm_arr = zeros(Float64,size(j_range))
    s_plus_arr = zeros(Float64,size(j_range))
    for (idx,j) in enumerate(j_range)
        dmp,dmm = get_dmp_dmm(t,j,alpha,beta)
        dmp_arr[idx] = dmp
        dmm_arr[idx] = dmm
    end
    if reverse
        tmp = dmp_arr
        dmp_arr = dmm_arr
        dmm_arr = tmp
        dmp_arr = dmp_arr[end:-1:1]
        dmm_arr = dmm_arr[end:-1:1]
    end
        
    M = zeros(Float64,m+1,m+1)
    Q = zeros(Float64,m-1,m-1)
#     for (idx,j) in enumerate(j_range)
    for idx in 2:m
        M[idx,idx+1] = dmp_arr[idx]
        M[idx,idx-1] = dmm_arr[idx]
        M[idx,idx] = -dmp_arr[idx]-dmm_arr[idx]
    end
    Q = M[2:m,2:m]
    W1 = M[2:m,1]
    W2 = M[2:m,m+1]
    return(- inv(Q)*W2)[1]
    
#     sanity check
#     eps = 0.5
#     P = eye(m+1) + eps*M
#     s0 = zeros(m+1)
#     s0[2] = 1.0
#     println((s0'*(P^100000))[end])
    
#     tot = 0
#     prod_vec = zeros(m)
#     lambdas = dmp_arr[2:m]
#     mus = dmm_arr[2:m]
#     for i in 1:m
#         prod_vec[i] = prod(lambdas[1:i-1])*prod(mus[i:end])
#     end
# #     prod_vec[1] = prod(mus)
# #     prod_vec[end] = prod(lambdas)
#     return prod(lambdas)/sum(prod_vec)
        
#     M[1,1] = 1
#     M[end,end] = 1
#     pi = eig(M')
    
#     return pi
end

function get_single_community_fixation(t,alpha,beta,plotting=false)
    y = 0.0
    t.i = t.N*y

    j_range = collect(0:t.m)
    y_range = j_range/t.m
    s_arr = zeros(Float64,size(j_range))
    dmp_arr = zeros(Float64,size(j_range))
    dmm_arr = zeros(Float64,size(j_range))
    s_plus_arr = zeros(Float64,size(j_range))
    for (idx,j) in enumerate(j_range)
        dmp,dmm = get_dmp_dmm(t,j,alpha,beta)
        s_arr[idx] = (dmp-dmm)
        s_plus_arr[idx] = (dmp+dmm)
        dmp_arr[idx] = dmp
        dmm_arr[idx] = dmm
    end

    # plot(j_range,s_arr./s_plus_arr)
    # plot(j_range,s_arr,"-")
    # if plotting
    #     plot(j_range,dmm_arr,"--")
    #     plot(j_range,dmp_arr,"-")
    # end


    interpolation_order = 3
    s_fn(x) = evaluate(Spline1D(y_range,s_arr,k=interpolation_order,bc="extrapolate"),x)
    splus_fn(x) = evaluate(Spline1D(y_range,s_plus_arr,k=interpolation_order,bc="extrapolate"),x)

    s_m_fn(x) = evaluate(Spline1D(1-y_range[end:-1:1],-s_arr[end:-1:1],k=interpolation_order,bc="extrapolate"),x)

    
    # if plotting
    #     figure()
    #     plot(y_range,s_fn(y_range))
    #     plot(y_range,s_m_fn(y_range))
    # end
    yy = y_range[1:end]
    pp_reverse = P_reach_fast(s_m_fn,splus_fn,t.m,1.0/t.m,yy)
    pp = P_reach_fast(s_fn,splus_fn,t.m,1.0/t.m,yy)
    return yy,pp,pp_reverse
end




end
