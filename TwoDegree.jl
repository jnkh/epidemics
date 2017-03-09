module TwoDegree

using SIS,Distributions, IM, LightGraphs,PayloadGraph, Dierckx,GraphGeneration, Epidemics

export 

run_epidemic_well_mixed_two_degree,compute_y1_y2_vs_y,
get_p_reach_well_mixed_two_degree_simulation

### Well Mixed Case ###
function run_epidemic_well_mixed_two_degree(N,alpha,beta,tdp,fixation_threshold=1.0)
    n1 = 0
    n2 = 0
    dt = 0.01
    if rand() < tdp.p1
        n1 = 1
    else
        n2 = 1
    end
    infecteds::Array{Float64,1} = []
    tuples::Array{Tuple{Float64,Float64,Float64},1} = []
    n = n1 + n2
    fixed=false
    push!(infecteds,n)
    push!(tuples,create_tuple(n,n1,n2,N,tdp))


    while n > 0
        if !(n < N && n < N*fixation_threshold)
            fixed = true
            break
        end
        n1,n2 = update_n_two_degree(n1,n2,N,alpha,beta,tdp,dt)
        n = n1 + n2
        push!(infecteds,n)
        push!(tuples,create_tuple(n,n1,n2,N,tdp))
    end

    size = dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,fixed),tuples
end

function create_tuple(n,n1,n2,N,tdp)
    N1 = Int(round(tdp.p1*N))
    N2 = N-N1
    return (n/N,n1/N1,n2/N2)
end



function get_p_neighbor(tdp)
    p1 = tdp.p1
    p2 = tdp.p2
    k1 = tdp.k1
    k2 = tdp.k2
    p1k = k1*p1/(k1*p1 + k2*p2)
    p2k = k2*p2/(k1*p1 + k2*p2)
    return p1k,p2k
end


function update_n_two_degree(n1::Int,n2::Int,N::Int,alpha,beta,tdp,dt)
    p1 = tdp.p1
    p2 = tdp.p2
    k1 = tdp.k1
    k2 = tdp.k2
    N1 = Int(round(tdp.p1*N))
    N2 = N-N1
    n = n1 + n2
    y = n/N
    y_tilde = get_mean_y_tilde(n1,n2,N1,N2,tdp)
    y_tilde_sq1 = get_mean_y_tilde_sq(k1,n1,n2,N1,N2,tdp)
    y_tilde_sq2 = get_mean_y_tilde_sq(k2,n1,n2,N1,N2,tdp)
    
    
    delta_n_plus1 = rand(Binomial(N1-n1,dt*(y_tilde + alpha*y_tilde_sq1)))
    delta_n_minus1 = rand(Binomial(n1,dt*(1-y_tilde)*(1 + beta)))
    
    delta_n_plus2 = rand(Binomial(N2-n2,dt*(y_tilde + alpha*y_tilde_sq2)))
    delta_n_minus2 = rand(Binomial(n2,dt*(1-y_tilde)*(1 + beta)))
    
    n1 = n1 + delta_n_plus1 - delta_n_minus1
    n2 = n2 + delta_n_plus2 - delta_n_minus2
    
    n1 = clamp(n1,0,N1)
    n2 = clamp(n2,0,N2)
#     y_desired = (n1+n2)/N
#     if y_desired > 0.08
#         n1 = Int(round(y_desired*N1))
#         n2 = Int(round(y_desired*N2))
#     end
    return n1,n2
end

function get_mean_y_tilde(n1,n2,N1,N2,tdp)
    y1 = n1/N1
    y2 = n2/N2
    p1k,p2k = get_p_neighbor(tdp)
    return p1k*y1 + p2k*y2
end

function get_mean_y_tilde_sq(k,n1,n2,N1,N2,tdp)
    y1 = n1/N1
    y2 = n2/N2
    p1k,p2k = get_p_neighbor(tdp)
    mean_k1sq = k^2*p1k^2 + k*p1k*(1-p1k)
    mean_k1 = k*p1k
    denom = mean_k1sq*y1^2 + mean_k1*y1*(1-y1) +
    2*(mean_k1*k - mean_k1sq)*y1*y2 +
    (k^2-2*mean_k1*k + mean_k1sq)*y2^2+
    (k-mean_k1)*y2*(1-y2)
    return denom/k^2
end
    
function get_p_reach_well_mixed_two_degree_simulation(alpha,beta,N,tdp,num_runs=10000)
    runs,tuples = run_epidemics_two_degree(num_runs, () -> run_epidemic_well_mixed_two_degree(N,alpha,beta,tdp,1.0));
    yy,pp = get_p_reach(runs,N)
    return yy,pp,tuples
end

function run_epidemics_two_degree(num_runs::Int,run_epidemic_fn)
    runs = EpidemicRun[]
    tuples_arr::Array{Array{Tuple{Float64,Float64,Float64},1}} = []

    for i in 1:num_runs
        run,tuples = run_epidemic_fn()
        push!(runs,run)
        push!(tuples_arr,tuples)
    end
    all_tuples = vcat(tuples_arr...)
    #get rid of fixed oneus

    return runs,all_tuples
end

######Data Analysis########

function compute_y1_y2_vs_y(tuples,N)
    y_arr = collect(0:1/N:1)
    y_count = zeros(N+1)
    y1_arr = zeros(N+1) 
    y2_arr = zeros(N+1) 
    for tuple in tuples
        y = tuple[1]
        y1 = tuple[2]
        y2 = tuple[3]
        idx = 1+Int(round(y*N))
        y_count[idx] += 1
        y1_arr[idx] += y1
        y2_arr[idx] += y2
    end
    y_count[y_count .== 0] = 1
    return y_arr, y1_arr ./ y_count, y2_arr ./y_count
end

function compute_y1_y2_vs_y(y_0,tuples,N,tol =0)
    if tol == 0
        tol = 1/N
    end
    count = 0
    y1_tot= 0
    y2_tot= 0
    for tuple in tuples
        y = tuple[1]
        y1 = tuple[2]
        y2 = tuple[3]
        if y_0 - tol <= y <= y_0 + tol
            y1_tot += y1
            y2_tot += y2
            count += 1
        end
    end
    return count,y1_tot/count,y2_tot/count 
end


end