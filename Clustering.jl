module Clustering

using Epidemics,
NLsolve,Distributions,StatsBase,LightGraphs

export

run_epidemics_with_clustering,run_epidemic_well_mixed_with_clustering,
get_p_reach_well_mixed_with_clustering, 

get_num_infecteds,EdgeCounts,copy



### Mean Field Simulation


function compute_p_reach(a,b,u_0,dt,p_k,num_trials=100)
    max_reaches = []
    for i = 1:num_trials
        y_k_arr,y_tot_arr = step_in_time(a,b,u_0,dt,p_k);
        max_y = maximum(y_tot_arr)
        push!(max_reaches,max_y)
    end
    return get_p_reach_from_max_reaches(max_reaches)
end
    
    
    
function get_p_reach_from_max_reaches(max_reaches)
    sorted_max_reaches = sort(max_reaches)
    xvals = unique(sorted_max_reaches)
    yvals = zeros(length(xvals))
    for r in sorted_max_reaches
        idx = findfirst(xvals,r)
        yvals[1:idx] += 1
    end

    return xvals,yvals/length(max_reaches)
end



function get_clustering_from_graph(g)
    return mean(local_clustering_coefficient(g))
end

type EdgeCounts
    mii::Int
    mis::Int
    msi::Int
    mss::Int
    m::Int
    k::Int
end

function copy(ms::EdgeCounts)
    return EdgeCounts(ms.mii,ms.mis,ms.msi,ms.mss,ms.m,ms.k)
end

function EdgeCounts(mii::Int,mis::Int,m::Int,k::Int)
    msi = mis
    mss = m - mis - msi - mii
    return EdgeCounts(mii,mis,msi,mss,m,k)
end

function update_mis_mii(ms::EdgeCounts,mis_new::Int,mii_new::Int)
    assert(0 <= mis_new <= ms.m)
    assert(0 <= mii_new <= ms.m)
    mss = ms.m - 2*mis_new - mii_new
    if mss < 0
        println(mis_new," ", mii_new)
    end
    assert(mss >= 0)
    ms.mis = mis_new
    ms.msi = mis_new
    ms.mii = mii_new
    ms.mss = mss
end

function get_infected_indicator(attached_infected::Bool)
    return attached_infected ? 1 : 0
end

function get_y(ms::EdgeCounts,I,z)
    k = ms.k
    return 1.0/k*(I + (k-1)*z)
end

function get_y_sq(ms::EdgeCounts,y,z)
    k = ms.k
    return y^2 + (k-1)/(k^2) * z*(1-z)
end

function get_y_y_sq(C::Float64,ms::EdgeCounts,central_infected::Bool,attached_infected::Bool)
    z = get_z_a_b_mean(C,ms,central_infected,attached_infected)
    I = get_infected_indicator(attached_infected)
    y = get_y(ms,I,z)
    y_sq = get_y_sq(ms,y,z)
    return y,y_sq 
end

function p_i_to_s(ms::EdgeCounts,C::Float64,central_infected::Bool,attached_infected::Bool,alpha,beta)
    y,y_sq = get_y_y_sq(C,ms,central_infected,attached_infected)
    ret = p_i_to_s(y,y_sq,alpha,beta)
    if 0.1*ret < 0 || 0.1*ret > 1 || ms.mii < 0 || ms.mis < 0 || ms.msi < 0 || ms.mss < 0
        println(ms)
        println(y," ",y_sq)
    end
    return p_i_to_s(y,y_sq,alpha,beta)
end

function p_s_to_i(ms::EdgeCounts,C::Float64,central_infected::Bool,attached_infected::Bool,alpha,beta)
    y,y_sq = get_y_y_sq(C,ms,central_infected,attached_infected)
    ret = p_s_to_i(y,y_sq,alpha,beta)
    if 0.1*ret < 0 || 0.1*ret > 1 || ms.mii < 0 || ms.mis < 0 || ms.msi < 0 || ms.mss < 0
        println(ms)
        println(y," ",y_sq)
    end
    return p_s_to_i(y,y_sq,alpha,beta)
end


function p_i_to_s(y,y_sq,alpha,beta)
    return (1-y)*(1+beta)
end

function p_s_to_i(y,y_sq,alpha,beta)
    return y + alpha*y_sq 
end

function get_pair_count(ms::EdgeCounts,first_infected::Bool,second_infected::Bool)
    if first_infected && second_infected
        return ms.mii
    elseif first_infected && ~second_infected
        return ms.mis
    elseif ~first_infected && second_infected
        return ms.msi
    else
        return ms.mss
    end
end

function get_pair_count_approximation(ms::EdgeCounts,first_infected::Bool,second_infected::Bool)
    fac1 = get_node_count(ms,first_infected)
    fac2 = get_node_count(ms,second_infected)
    return fac1*fac2*ms.k/get_N(ms)
end

function get_pair_count_division_safe(ms::EdgeCounts,first_infected::Bool,second_infected::Bool)
    ret = get_pair_count(ms,first_infected,second_infected)
    ret_alt = get_pair_count_approximation(ms,first_infected,second_infected)
    # if (~first_infected && ~second_infected) || (ms.mii > 0 && ms.mis > 0)
    if (ms.mis == 0 || ms.mii == 0) && (first_infected || second_infected)
        return ret_alt
    else
        # println(ms)
        # println(first_infected, " ", second_infected)
        # println("n: $ret, alt : $(ret_alt)")
        return ret 
    end
end

function get_node_count(ms::EdgeCounts,infected::Bool)
    n = get_num_infecteds(ms)
    if infected
        return n
    else
        return get_N(ms) - n
    end
end

function get_N(ms)
    return ms.m/ms.k
end

function get_frac_infecteds(ms::EdgeCounts)
    y = (ms.mii + ms.mis)/ms.m
    return y
end

function get_num_infecteds(ms::EdgeCounts)
    n = (ms.mii + ms.mis)/ms.k
    return n
end

function get_z_a_b(ms::EdgeCounts,central_infected::Bool,attached_infected::Bool)
    term_numerator = 
    get_pair_count_division_safe(ms,central_infected,false)*get_pair_count_division_safe(ms,attached_infected,false)*get_node_count(ms,true)
    term_denominator = 
    get_pair_count_division_safe(ms,central_infected,true)*get_pair_count_division_safe(ms,attached_infected,true)*get_node_count(ms,false)

    return 1/(1 + (term_numerator/term_denominator) )
end

function get_z_a(ms::EdgeCounts,central_infected::Bool)
    return get_pair_count_division_safe(ms,central_infected,true)/(ms.k * get_node_count(ms,central_infected))
end


function get_z_a_b_mean(C::Float64,ms::EdgeCounts,central_infected::Bool,attached_infected::Bool)
    z_a_b = get_z_a_b(ms,central_infected,attached_infected)
    z_a = get_z_a(ms,central_infected)
    # if ms.mis == 0 || ms.mii == 0
        # println(ms)
        # println("$(central_infected), $(attached_infected)")
        # println("z_a_b: $(z_a_b)")
        # println("z_a: $(z_a)")
    # end
    return C*z_a_b + (1-C)*z_a
end


function update_ms_clustering(ms::EdgeCounts,C::Float64,N::Int,alpha,beta,dt)
    k = ms.k
    mii_delta_minus_raw = rand(Binomial(ms.mii,dt*p_i_to_s(ms,C,true,true,alpha,beta))) 
    mii_delta_plus_raw = rand(Binomial(ms.msi,dt*p_s_to_i(ms,C,false,true,alpha,beta))) 

    mii_delta_minus = 2*mii_delta_minus_raw
    mii_delta_plus = 2*mii_delta_plus_raw


    mis_delta_minus1 = mii_delta_plus_raw
    mis_delta_plus1 = mii_delta_minus_raw
    mis_delta_minus2 = rand(Binomial(ms.mis,dt*p_i_to_s(ms,C,true,false,alpha,beta))) 
    mis_delta_plus2 = rand(Binomial(ms.mss,dt*p_s_to_i(ms,C,false,false,alpha,beta)))


    mis_curr = ms.mis + k* ((mis_delta_plus1 + mis_delta_plus2) - (mis_delta_minus1+mis_delta_minus2))
    mii_curr = ms.mii + k* (mii_delta_plus - mii_delta_minus)
    mis_curr = clamp(mis_curr,0,Int(ms.m/2))
    mii_curr = clamp(mii_curr,0,ms.m)

    while (mii_curr + 2*mis_curr) > ms.m
        if mii_curr >= 2*mis_curr
            mii_curr -= 2*k
        else
            mis_curr -= 1*k
        end
    end
    if mii_curr + 2*mis_curr > ms.m
        println(ms)
    end

    update_mis_mii(ms,mis_curr,mii_curr)
end

function is_fixed(ms::EdgeCounts,N,fixation_threshold)
    n = get_num_infecteds(ms)
    return n >= N || n >= fixation_threshold*N || n <= 0# ms.mis == 0
end


function run_epidemic_well_mixed_with_clustering(N,k,C,alpha,beta,fixation_threshold=1.0)
    m = N*k

    mii = 0
    mis = k
    ms = EdgeCounts(mii,mis,m,k)
    
    dt = 0.1/(2*k)
    max_y = -1.0
#     infecteds::Array{Float64,1} = []
    infecteds::Array{EdgeCounts,1} = []
    
    n = get_num_infecteds(ms)
    max_y = n/N 
    fixed=false
    push!(infecteds,copy(ms))

    step = 0
    max_steps = Inf
#     push!(y_k_vec,n_vec)

    while n > 0
        if is_fixed(ms,N,fixation_threshold) || step > max_steps
            fixed = true
            break
        end
   
        update_ms_clustering(ms,C,N,alpha,beta,dt)

        n = get_num_infecteds(ms)
        max_y = max(max_y,n/N)
        push!(infecteds,copy(ms))

        step += 1
        #println(ms)
        # infecteds[step] = n
        # push!(infecteds,n)
        # n_vec_actual = zeros(get_k_range(N))
        # n_vec_actual[k_idx_array] = n_vec
#         push!(y_k_vec,n_vec_actual)
    end

    run_size = 0#dt*sum(infecteds)
    if fixed
        run_size = Inf
    end

    return infecteds,max_y
end





function get_p_reach_well_mixed_with_clustering(N,k,C,alpha,beta,num_runs=10)
    runs,maxs = run_epidemics_with_clustering(num_runs, () -> run_epidemic_well_mixed_with_clustering(N,k,C,alpha,beta));
#     yy,pp = get_p_reach(runs,N)
    yy,pp = get_p_reach_from_max_reaches(maxs)
    return yy,pp,runs
end

function run_epidemics_with_clustering(num_runs::Int,run_epidemic_fn)
    runs = Array(Array{EdgeCounts,1},num_runs)
    maxs = Array(Float64,num_runs)

    for i in 1:num_runs
        run,max_y = run_epidemic_fn()
        runs[i] = run
        maxs[i] = max_y
#         push!(y_k_vec_arr,y_k_vec)
    end
    #get rid of fixed oneus
    # runs = vcat(runs...)

    return runs,maxs
end



end