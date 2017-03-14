module DegreeDistribution

using Epidemics,
NLsolve,Distributions,StatsBase

export get_mean_k, get_k_range,
get_y_k_equilibrium, get_y_k_branching_process,
get_p_k_as_vec,
get_mean_y_k,
get_delta_y_plus,get_delta_y_minus,get_y_bar,

run_epidemics_by_degree,run_epidemic_well_mixed_by_degree,
get_p_reach_well_mixed_by_degree_simulation


function get_k_range(N)
    return collect(1:N-1)
end

function get_mean_k(p_k_fn::Function,N::Int)
    mean_k = 0
    k_range = get_k_range(N)
    for k in k_range
        mean_k += k*p_k_fn(k) 
    end
    return mean_k
end

function get_mean_k(p_k::Array,N::Int)
    return dot(p_k,get_k_range(N))
end

function neighbor_degree_distribution(k,degree_distribution,mean_k)
    return k * degree_distribution(k)/mean_k
end

function get_p_k_as_vec(degr_distr,N)
    thresh = 1e-3*1/N
    k_range = get_k_range(N)
    dd(x) = degr_distr(x) #(x) = 1/N
    mean_k = get_mean_k(dd,N)
    ndd(x) = neighbor_degree_distribution(x,degr_distr,mean_k)
    p_k = []
    p_k_neighbor = []
    for x in k_range
        push!(p_k,dd(x))
        push!(p_k_neighbor,ndd(x))
    end
    mask = p_k .> thresh
    #set irrelevant entries to zero.
    p_k[~mask] = 0
    p_k_neighbor[~mask] = 0
    p_k /= sum(p_k)
    p_k_neighbor /= sum(p_k_neighbor)
    return p_k, p_k_neighbor,mean_k
end


function get_mean_y_k(y_k,p_k_neighbor,N)
    mean_y_k = 0
    for k in 1:length(p_k_neighbor)
        mean_y_k += y_k[k]*p_k_neighbor[k]
    end
    return mean_y_k
end


function get_mean_N_k(p_k,p_k_neighbor,N)
    mean_N_k = 0
    for k in 1:length(p_k)
        mean_N_k += N*p_k[k]*p_k_neighbor[k]
    end
    return mean_N_k
end


function get_mean_y_k_squared(mean_y_k,k)
    return mean_y_k^2 + (1 - mean_y_k)* mean_y_k/k
end

function get_mean_y_k_squared_hg(mean_y_k,k,mean_N_k)
    return mean_y_k^2 + (1 - mean_y_k)* mean_y_k*(mean_N_k - k)/(k*mean_N_k)
end


function y_k_dot_given_y_k!(y_k,y_k_dot,y_desired,N,p_k,p_k_neighbor,alpha,beta)
    k_range = get_k_range(N)
    mean_y_k = get_mean_y_k(y_k,p_k_neighbor,N)
    mean_N_k = get_mean_N_k(p_k,p_k_neighbor,N)
    for k in k_range
        # mean_y_k_squared = get_mean_y_k_squared(mean_y_k,k)
        mean_y_k_squared = get_mean_y_k_squared_hg(mean_y_k,k,mean_N_k)
        y_k_dot[k] = y_k[end]*get_delta_y_k_plus(y_k,k,mean_y_k,mean_y_k_squared,alpha) - get_delta_y_k_minus(y_k,k,mean_y_k,beta) 
    end
    y_k_dot[end] = y_desired - get_y_bar(p_k,y_k) 
end

function get_delta_y_k_plus(y_k,k,mean_y_k,mean_y_k_squared,alpha)
    return (1 - y_k[k])*(mean_y_k + alpha*mean_y_k_squared)
end

function get_delta_y_k_minus(y_k,k,mean_y_k,beta)
    return y_k[k]* (1 - mean_y_k)*(1 + beta)
end

function get_delta_y_plus(y_k,p_k,mean_y_k,N,alpha)
    k_range = get_k_range(N)
    ret = 0
    for k in k_range
        mean_y_k_squared = get_mean_y_k_squared(mean_y_k,k)
        ret += get_delta_y_k_plus(y_k,k,mean_y_k,mean_y_k_squared,alpha)*p_k[k]
    end
    return ret
end

function get_delta_y_minus(y_k,p_k,mean_y_k,N,beta)
    k_range = get_k_range(N)
    ret = 0
    for k in k_range
        ret += get_delta_y_k_minus(y_k,k,mean_y_k,beta)*p_k[k]
    end
    return ret
end
 
function get_y_bar(p_k,y_k)
    return dot(p_k,y_k[1:end-1])
end
    

function mock_degree_distribution(mean_k,sigma_k)
    function p_k(k)
        if k == mean_k + sigma_k
            return 0.5
        elseif k == mean_k - sigma_k
            return 0.5
        else
            return 0
        end
    end
    return p_k
end

function get_y_k_equilibrium(y_desired,N,p_k,p_k_neighbor,alpha,beta)
    f!(x,out) = y_k_dot_given_y_k!(x,out,y_desired,N,p_k,p_k_neighbor,alpha,beta)
    x_init = 1/N*zeros(N)
    ret = nlsolve(f!,x_init,autodiff=true,ftol=1e-12)
    y_k = ret.zero
    return y_k
end



function get_y_k_branching_process(y_desired,p_k,alpha,N)
    y_k = ones(N)
    for k = 1:N-1
        y_k[k] = (1 + alpha/k)
    end
    return y_desired*y_k/dot(y_k[1:N-1],p_k)
end



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


function get_a_k_b_k(y_k,p_k,p_k_neighbor,alpha,beta,N,ks)
    z = get_z(y_k,p_k_neighbor)
    N_k = N*p_k
    y_k_tilde = get_y_k_tilde(z)
    y_k_sq_tilde = get_y_k_sq_tilde(z,ks)
    a_k = get_a_k(y_k,y_k_tilde,y_k_sq_tilde,alpha,beta)
    b_k = get_b_k(y_k,y_k_tilde,y_k_sq_tilde,alpha,beta,N_k)
    return a_k,b_k
end

function get_a_k_only(y_k,p_k,p_k_neighbor,alpha,beta,N,ks)
    z = get_z(y_k,p_k_neighbor)
    N_k = N*p_k
    y_k_tilde = get_y_k_tilde(z)
    y_k_sq_tilde = get_y_k_sq_tilde(z,ks)
    a_k = get_a_k(y_k,y_k_tilde,y_k_sq_tilde,alpha,beta)
    return a_k
end

function get_b_k_only(y_k,p_k,p_k_neighbor,alpha,beta,N,ks)
    z = get_z(y_k,p_k_neighbor)
    N_k = N*p_k
    y_k_tilde = get_y_k_tilde(z)
    y_k_sq_tilde = get_y_k_sq_tilde(z,ks)
    b_k = get_b_k(y_k,y_k_tilde,y_k_sq_tilde,alpha,beta,N_k)
    return b_k
end


#forcing function
function get_a_k(y_k,y_k_tilde,y_k_sq_tilde,alpha,beta)
    return (1 - y_k).*(y_k_tilde + alpha.*y_k_sq_tilde) - y_k.*(1 - y_k_tilde)*(1 + beta)
end

#drift function
function get_b_k(y_k,y_k_tilde,y_k_sq_tilde,alpha,beta,N_k)
    vec = 1./N_k.*((1 - y_k).*(y_k_tilde + alpha.*y_k_sq_tilde) + y_k.*(1 - y_k_tilde).*(1 + beta))
    vec[N_k .== 0] = 0
    return vec 
end

function get_x_k(y_k,p_k_neighbor)
    return y_k .* p_k_neighbor
end

function get_z(y_k,p_k_neighbor)
    return dot(y_k,p_k_neighbor)
end

function get_y_k_tilde(z::Float64)
    return z
end

function get_y_k_sq_tilde(z::Float64,k::Real)
    return z^2 + 1.0/k*z*(1-z)
end

function run_epidemic_well_mixed_by_degree(N,alpha,beta,p_k,p_k_neighbor,fixation_threshold=1.0;hypergeometric=true)
    ks = get_k_range(N)
    
    #sample N_k (could also do this deterministically!)
    N_k = rand(Multinomial(N,p_k))
    #need to adjust p_k for this specific application:
    p_k = N_k ./ N
    p_k_neighbor = p_k .*ks
    p_k /= sum(p_k)
    p_k_neighbor /= sum(p_k_neighbor)
    
    
    n_vec = zeros(ks)
    idx = sample(get_k_range(N),WeightVec(N_k))
    n_vec[idx] = 1
    dt = 0.1
    max_y = -1.0
#     infecteds::Array{Float64,1} = []
    infecteds::Array{Float64,1} = []
    y_k_vec::Array{Array{Float64,1},1} = []
    y_k = 1.0*zeros(ks)
    
    n = sum(n_vec)
    max_y = n/N
    fixed=false
#     push!(infecteds,n)
#     push!(y_k_vec,n_vec)

    while n > 0
        if n >= N || n >= fixation_threshold*N 
            fixed = true
            break
        end
        update_n_vec_by_degree(n_vec,y_k,ks,
        N,N_k,p_k,p_k_neighbor,alpha,beta,dt,hypergeometric)
        n = sum(n_vec)
        max_y = max(max_y,n/N)
        # infecteds[step] = n
        # push!(infecteds,n)
#         push!(y_k_vec,n_vec)
    end

    size = dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,fixed),y_k_vec,max_y
end

function get_y_k_from_n_k!(y_k,n_k,N_k)
    for i = 1:length(y_k)
        y_k[i] = n_k[i]/N_k[i]
        if N_k[i] == 0
            y_k[i] = 0
        end
    end
end

function get_delta_k(k,N,mean_k)
    # return  1 - k/(N*mean_k)
    return  1 - k/(N)
    # return  1
    # return 0.0
end


function get_hg_correction_1(y_k,p_k_neighbor,N_k)
    fac1 = 0.0
    fac2 = 0.0
    fac3 = 0.0
    for k = 1:length(N_k)
        if N_k[k] > 1
            fac1 += y_k[k]*(1-y_k[k])*p_k_neighbor[k]^2/(N_k[k])
            fac2 += y_k[k]*p_k_neighbor[k]*(1-p_k_neighbor[k])*(1-y_k[k])/N_k[k]
        end
        fac3 += y_k[k]^2*p_k_neighbor[k]
    end
    return fac1,fac2,fac3
end

function get_hg_correction(y_k,p_k_neighbor,N_k)
    fac1 = 0.0
    fac2 = 0.0
    for k = 1:length(N_k)
        if N_k[k] > 1
            fac1 += y_k[k]*(1-y_k[k])*p_k_neighbor[k]/(N_k[k])
            fac2 += y_k[k]*(1-y_k[k])*p_k_neighbor[k]*(1-p_k_neighbor[k])/(N_k[k])
        end
    end
    return fac1,fac2
end

function get_y_k_sq_tilde_hg(z::Float64,k::Real,N::Float64)
    return z^2 + 1.0*((N-k)/(N*k))*z*(1-z)
end

function get_y_k_sq_tilde_hg_2(z,k,fac1,fac2)
    return z^2 + (1.0/k)*z*(1-z) - fac1 - (1.0/k)*fac2
end

function get_y_k_sq_tilde_hg_3(z,k,N,mean_k,fac1,fac2,fac3)
    delta_k = get_delta_k(k,N,mean_k)
    return z^2 + (1.0/k)*z*(1-delta_k*z) - fac1 - (1.0/k)*(delta_k*fac2 + (1 - delta_k)*fac3)
end


function update_n_vec_by_degree(n_vec::Array{Int,1},y_k::Array{Float64,1},
    ks::Array{Int,1},N::Int,N_k,p_k,p_k_neighbor,alpha,beta,dt,hypergeometric=true)
    
    N_ave = dot(p_k_neighbor,N_k)
    get_y_k_from_n_k!(y_k,n_vec,N_k)
    z = get_z(y_k,p_k_neighbor)
    y_k_tilde = 0.0
    y_k_sq_tilde = 0.0
    n_plus_trials = 0.0
    n_minus_trials = 0.0
    n_plus_prob = 0.0
    n_minus_prob = 0.0

    hg_fac1 = 0.0
    hg_fac2 = 0.0
    hg_fac3 = 0.0
    mean_k = 0.0

    if hypergeometric
        # hg_fac1,hg_fac2 = get_hg_correction(y_k,p_k_neighbor,N_k)
        mean_k = dot(ks,p_k)
        hg_fac1,hg_fac2,hg_fac3 = get_hg_correction_1(y_k,p_k_neighbor,N_k)
    end
    
    
    for k in ks 
        y_k_tilde = get_y_k_tilde(z)
#         y_k_sq_tilde = get_y_k_sq_tilde(z,k)
        if hypergeometric
            # y_k_sq_tilde = get_y_k_sq_tilde_hg(z,k,N_ave)
            # y_k_sq_tilde = get_y_k_sq_tilde_hg_2(z,k,hg_fac1,hg_fac2)
            # y_k_sq_tilde = get_y_k_sq_tilde_hg_2(z,k,hg_fac1,hg_fac2)
            y_k_sq_tilde = get_y_k_sq_tilde_hg_3(z,k,N,mean_k,hg_fac1,hg_fac2,hg_fac3)
        else
            y_k_sq_tilde = get_y_k_sq_tilde(z,k)
        end
        n_plus_trials = N_k[k] - n_vec[k]
        n_minus_trials = n_vec[k]
        n_plus_prob = dt*(y_k_tilde + alpha*y_k_sq_tilde)
        n_minus_prob = dt*(1-y_k_tilde)*(1 + beta)
        if n_plus_trials > 0
            delta_n_k_plus = rand(Binomial(n_plus_trials,n_plus_prob))
        else
            delta_n_k_plus = 0
        end
        if n_minus_trials > 0
            delta_n_k_minus = rand(Binomial(n_minus_trials,n_minus_prob))
        else
            delta_n_k_minus = 0
        end
        n_vec[k] = n_vec[k] + delta_n_k_plus - delta_n_k_minus
        if n_vec[k] < 0 || n_vec[k] > N_k[k]
            println(n_vec[k], " y:", sum(n_vec)/N)
        end
        n_vec[k] = clamp(n_vec[k],0,N_k[k])
    end
end

function adjust_binomial_factor_to_integer(n_float)
    n_int = ceil(n_float)
    fac = n_int/n_float
    return Int(n_int),fac
end


function get_p_reach_well_mixed_by_degree_simulation(N,alpha,beta,p_k,p_k_neighbor,num_runs=10,hypergeometric=true)
    runs,tuples,maxs = run_epidemics_by_degree(num_runs, () -> run_epidemic_well_mixed_by_degree(N,alpha,beta,p_k,p_k_neighbor,hypergeometric=hypergeometric));
#     yy,pp = get_p_reach(runs,N)
    yy,pp = get_p_reach_from_max_reaches(maxs)
    return yy,pp,tuples
end

function run_epidemics_by_degree(num_runs::Int,run_epidemic_fn)
    runs = Array(EpidemicRun,num_runs)
    maxs = Array(Float64,num_runs)
    y_k_vec_arr::Array{Array{Array{Float64,1},1}} = []

    for i in 1:num_runs
        _,y_k_vec,max_y = run_epidemic_fn()
#         runs[i] = run
        maxs[i] = max_y
#         push!(y_k_vec_arr,y_k_vec)
    end
    all_y_k_vec = vcat(y_k_vec_arr...)
    #get rid of fixed oneus

    return runs,hcat(all_y_k_vec),maxs
end



end