module Clustering

using Epidemics,SIS,
NLsolve,Distributions,StatsBase,LightGraphs,GSL

export

run_epidemics_with_clustering,run_epidemic_well_mixed_with_clustering,
get_p_reach_well_mixed_with_clustering, 

get_num_infecteds,EdgeCounts,copy,

get_z_a_b_mean,get_z_a,get_z_a_b



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
    mii::Real
    mis::Real
    msi::Real
    mss::Real
    m::Real
    k::Real
end

function copy(ms::EdgeCounts)
    return EdgeCounts(ms.mii,ms.mis,ms.msi,ms.mss,ms.m,ms.k)
end

function EdgeCounts(mii::Real,mis::Real,m::Real,k::Real)
    msi = mis
    mss = m - mis - msi - mii
    return EdgeCounts(mii,mis,msi,mss,m,k)
end


function get_minimum_implied_mis(mii::Int,k::Int)
    if mii == 0 return 0 end
    n_min = Int(ceil(0.5*(1 + sqrt(1 + 4*mii))))
    return n_min*k-n_min*(n_min-1)
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

function adjust_pair_count_for_self(ms::EdgeCounts,central_infected::Bool,attached_infected::Bool)
    ms_new = copy(ms) 
    ms_new.m -= 2
    if central_infected &&  attached_infected
        ms_new.mii -= 2
    elseif (central_infected && ~ attached_infected) || (~central_infected &&  attached_infected)
        ms_new.mis -= 1
        ms_new.msi -= 1
    else
        ms_new.mss -= 2
    end
    return ms_new
end




function get_pair_count_approximation(ms::EdgeCounts,first_infected::Bool,second_infected::Bool)
    fac1 = get_node_count(ms,first_infected)
    fac2 = get_node_count(ms,second_infected)
    return fac1*fac2*ms.k/get_N(ms)
end

function get_pair_count_division_safe(ms::EdgeCounts,first_infected::Bool,second_infected::Bool)
    ret = get_pair_count(ms,first_infected,second_infected)
    ret_alt = get_pair_count_approximation(ms,first_infected,second_infected)
    return ret
    # if (~first_infected && ~second_infected) || (ms.mii > 0 && ms.mis > 0)
    # if (ms.mis == 0) && (first_infected || second_infected)
    # if (get_num_infecteds(ms) < 3) return ret_alt end
    # if (ms.mis <= ms.k) && (first_infected $ second_infected)
    # if (ms.mis == 0 || ms.mii == 0) && (first_infected || second_infected)
    # if get_num_infecteds(ms) < get_N(ms) && (first_infected $ second_infected)
        # return max(ret_alt,ret)
    # else
        # println(ms)
        # println(first_infected, " ", second_infected)
        # println("n: $ret, alt : $(ret_alt)")
        # return ret 
    # end
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

    ret = 1/(1 + (term_numerator/term_denominator) )
    if ~(0 <= ret <= 1)
        println(ms)
        println(term_numerator)
        println(term_denominator)
        println(ret)
    end
    return guarantee_between_a_b(ret,0,1)
end

function get_z_a(ms::EdgeCounts,central_infected::Bool)
    ret =  get_pair_count_division_safe(ms,central_infected,true)/(ms.k * get_node_count(ms,central_infected))
    return guarantee_between_a_b(ret,0,1)
end

function guarantee_between_a_b(val::Float64,a::Real,b::Real,eps::Float64=1e-6)
    assert(a-eps <= val <= b+eps)
    return clamp(val,a,b)
end


function get_z_a_b_mean(C::Float64,ms::EdgeCounts,central_infected::Bool,attached_infected::Bool)
    z_a_b = get_z_a_b(ms,central_infected,attached_infected)
    z_a = get_z_a(ms,central_infected)
    # println("a: $(z_a), ab: $(z_a_b)")
    # if ms.mis == 0 || ms.mii == 0
        # println(ms)
        # println("$(central_infected), $(attached_infected)")
        # println("z_a_b: $(z_a_b)")
        # println("z_a: $(z_a)")
    # end
    return C*z_a_b + (1-C)*z_a
end


function update_ms_clustering(ms::EdgeCounts,C::Float64,N::Int,alpha,beta,dt)
    mis_curr = 0
    mii_curr = 0
    # success = false
    # resampled = 0
    # while ~success

    delta_mis,delta_mii = get_delta_mis_mii_gillespie(ms,C,N,alpha,beta,dt)
    # delta_mis,delta_mii = get_delta_mis_mii_complex(ms,C,N,alpha,beta,dt)
    # delta_mis,delta_mii = get_delta_mis_mii_sampling(ms,C,N,alpha,beta,dt)
    mis_curr = ms.mis + delta_mis 
    mii_curr = ms.mii + delta_mii 
    mis_curr = clamp(mis_curr,0,Int(ms.m/2))
    mii_curr = clamp(mii_curr,0,ms.m)

    while (mii_curr + 2*mis_curr) > ms.m
        if mii_curr >= 2*mis_curr
            mii_curr -= 2
        else
            mis_curr -= 1
        end
    end
    if mii_curr + 2*mis_curr > ms.m
        println(ms)
    end
    update_mis_mii(ms,mis_curr,mii_curr)
    
        # mis_new_min = get_minimum_implied_mis(mii_curr,ms.k)
        # success = mis_curr >= mis_new_min
        # if ~success 
            # resampled += 1 
            # println("resampling ($resampled), mis_new: $(mis_curr), min: $(mis_new_min)")
            # println(ms)
        # end
    # end


end


function get_delta_mis_mii(ms::EdgeCounts,C::Float64,N::Int,alpha,beta,dt)
    k = ms.k

    fac = 1#k


    mii_delta_minus_raw = ms.mii > 0 ? fac*rand(Binomial(ms.mii,dt*p_i_to_s(ms,C,true,true,alpha,beta)))  : 0
    mii_delta_plus_raw = ms.msi > 0 ? fac*rand(Binomial(ms.msi,dt*p_s_to_i(ms,C,false,true,alpha,beta))) : 0

    mii_delta_minus = 2*mii_delta_minus_raw
    mii_delta_plus = 2*mii_delta_plus_raw


    mis_delta_minus1 = mii_delta_plus_raw
    mis_delta_plus1 = mii_delta_minus_raw
    # mis_delta_plus1 = fac*rand(Binomial(ms.mii,dt*p_i_to_s(ms,C,true,true,alpha,beta))) 
    # mis_delta_minus1 = fac*rand(Binomial(ms.msi,dt*p_s_to_i(ms,C,false,true,alpha,beta))) 
    mis_delta_minus2 = ms.mis > 0 ? fac*rand(Binomial(ms.mis,dt*p_i_to_s(ms,C,true,false,alpha,beta)))  : 0
    mis_delta_plus2 = ms.mss > 0 ? fac*rand(Binomial(ms.mss,dt*p_s_to_i(ms,C,false,false,alpha,beta))) : 0


    delta_mis =  1* ((mis_delta_plus1 + mis_delta_plus2) - (mis_delta_minus1+mis_delta_minus2))
    delta_mii =  1* (mii_delta_plus - mii_delta_minus)
    return delta_mis,delta_mii
end


function get_delta_mis_mii_gillespie(ms::EdgeCounts,C::Float64,N::Int,alpha,beta,dt)
    k = ms.k

    fac = k


    delta_mis = 0
    delta_mii = 0
    posterior = true
    k_i,k_s = 0,0
    #ss -> is 
    rates = Float64[] 
    infected_configuration = []
    ms_effectives = [] 
    for central_infected in [true,false]
        for attached_infected in [true,false]
            m_count = get_pair_count(ms,central_infected,attached_infected)
            ms_effective = adjust_pair_count_for_self(ms,central_infected,attached_infected)
            transition_fn = central_infected ? p_i_to_s : p_s_to_i
            rate = m_count > 0 ? fac*dt*transition_fn(ms_effective,C,central_infected,attached_infected,alpha,beta) : 0.0
            push!(rates,m_count*rate)
            push!(infected_configuration,(central_infected,attached_infected))
            push!(ms_effectives,ms_effective)
        end
    end
    change_idx,t = pick_update_and_time(rates)
    central_infected,attached_infected = infected_configuration[change_idx]
    ms_effective = ms_effectives[change_idx]
    delta_curr = 1
    if posterior
        k_i,k_s = sample_posterior_k_i(C,ms_effective,central_infected,attached_infected,alpha,beta)
    else
        k_i,k_s = draw_k_i(ms_effective,k,C,central_infected,attached_infected)
    end

    I_a = get_infected_indicator(attached_infected)
    k_s_tot = (1-I_a) + k_s
    k_i_tot = I_a + k_i

    delta_mii_loc = 2*k_i_tot*delta_curr
    delta_mis_loc = k_i_tot*delta_curr - k_s_tot*delta_curr

    if central_infected
        delta_mii -= delta_mii_loc
        delta_mis += delta_mis_loc
    else
        delta_mii += delta_mii_loc
        delta_mis -= delta_mis_loc
    end
    return delta_mis,delta_mii
end




function get_delta_mis_mii_complex(ms::EdgeCounts,C::Float64,N::Int,alpha,beta,dt)
    k = ms.k

    fac = k


    delta_mis = 0
    delta_mii = 0
    posterior = true
    k_i,k_s = 0,0
    #ss -> is 
    for central_infected in [true,false]
        for attached_infected in [true,false]
            m_count = get_pair_count(ms,central_infected,attached_infected)
            if m_count > 0
                ms_effective = adjust_pair_count_for_self(ms,central_infected,attached_infected)
                transition_fn = central_infected ? p_i_to_s : p_s_to_i
                delta_curr_ = rand(Binomial(m_count,1/fac*dt*transition_fn(ms_effective,C,central_infected,attached_infected,alpha,beta)))
                for i =1:delta_curr_
                    delta_curr = 1
                    if posterior
                        k_i,k_s = sample_posterior_k_i(C,ms_effective,central_infected,attached_infected,alpha,beta)
                    else
                        k_i,k_s = draw_k_i(ms_effective,k,C,central_infected,attached_infected)
                    end

                    I_a = get_infected_indicator(attached_infected)
                    k_s_tot = (1-I_a) + k_s
                    k_i_tot = I_a + k_i

                    delta_mii_loc = 2*k_i_tot*delta_curr
                    delta_mis_loc = k_i_tot*delta_curr - k_s_tot*delta_curr

                    if central_infected
                        delta_mii -= delta_mii_loc
                        delta_mis += delta_mis_loc
                    else
                        delta_mii += delta_mii_loc
                        delta_mis -= delta_mis_loc
                    end
                end
            end
        end
    end
    return delta_mis,delta_mii
end



function get_delta_mis_mii_sampling(ms::EdgeCounts,C::Float64,N::Int,alpha,beta,dt)
    k = ms.k

    fac = k


    delta_mis = 0
    delta_mii = 0
    posterior = false
    k_i,k_s = 0,0
    #ss -> is 
    for central_infected in [true,false]
        for attached_infected in [true,false]
            ms_effective = ms#adjust_pair_count_for_self(ms,central_infected,attached_infected)
            m_count = get_pair_count(ms,central_infected,attached_infected)
            transition_fn = central_infected ? p_i_to_s : p_s_to_i
            I_a = get_infected_indicator(attached_infected)
            for i = 1:m_count
                k_i,k_s = draw_k_i(ms_effective,k,C,central_infected,attached_infected)
                y = (I_a + k_i)/k
                p_transition = 1/fac*dt*transition_fn(y,y^2,alpha,beta)
                transitioned = rand() < p_transition
                if transitioned
                    delta_curr = 1
                    k_s_tot = (1-I_a) + k_s
                    k_i_tot = I_a + k_i

                    delta_mii_loc = 2*k_i_tot*delta_curr
                    delta_mis_loc = k_i_tot*delta_curr - k_s_tot*delta_curr

                    if central_infected
                        delta_mii -= delta_mii_loc
                        delta_mis += delta_mis_loc
                    else
                        delta_mii += delta_mii_loc
                        delta_mis -= delta_mis_loc
                    end
                end
            end
        end
    end
    return delta_mis,delta_mii
end


function draw_k_i(ms::EdgeCounts,k::Int,C,central_infected::Bool,attached_infected::Bool)
    k_prime = k-1

    k_c = rand(Binomial(k_prime,C))
    k_c_bar = k_prime - k_c 

    z_a = get_z_a(ms,central_infected)
    z_a_b = get_z_a_b(ms,central_infected,attached_infected)
    k_i = rand(Binomial(k_c,z_a_b)) + rand(Binomial(k_c_bar,z_a))

    # sab,fab = success_failure_pair_count_z_a_b(ms,central_infected,attached_infected)
    # sa,fa = success_failure_pair_count_z_a(ms,central_infected,attached_infected)
    # if !(0 <= k_c <= sab+fab)
    #     println("AB")
    #     println("k: $(k_c), s: $(sab), f: $(fab)")
    #     println(ms)
    #     println(central_infected, " ", attached_infected)
    # end
    # if !(0 <= k_c_bar <= sa+fa)
    #     println("A")
    #     println("k: $(k_c_bar), s: $(sa), f: $(fa)")
    #     println(ms)
    # end
    # k_i = rand(Hypergeometric(sab,fab,k_c)) + rand(Hypergeometric(sa,fa,k_c_bar))

    k_s = k_prime - k_i
    return k_i,k_s
end


function success_failure_pair_count_z_a_b(ms::EdgeCounts,central_infected::Bool,attached_infected::Bool)
    # failures = get_pair_count_division_safe(ms,central_infected,false)*get_pair_count_division_safe(ms,attached_infected,false)/get_node_count(ms,false)
    # successes = get_pair_count_division_safe(ms,central_infected,true)*get_pair_count_division_safe(ms,attached_infected,true)/get_node_count(ms,true)
    failures = triple_count_c(ms,central_infected,attached_infected,false)
    successes = triple_count_c(ms,central_infected,attached_infected,true)
    # get_pair_count_division_safe(ms,central_infected,false)*get_pair_count_division_safe(ms,attached_infected,false)/get_node_count(ms,false)
    # successes = get_pair_count_division_safe(ms,central_infected,true)*get_pair_count_division_safe(ms,attached_infected,true)/get_node_count(ms,true)
    # if (failures != round(failures) ) || ( successes != round(successes))
        # println("f: $failures, s: $successes")
        # println(ms)
    # end
    return Int(round(successes)),Int(round(failures))
end


function success_failure_pair_count_z_a(ms::EdgeCounts,central_infected::Bool,attached_infected)

    failures = triple_count_noc(ms,central_infected,attached_infected,false)
    successes = triple_count_noc(ms,central_infected,attached_infected,true)


    # successes = get_pair_count_division_safe(ms,central_infected,true)
    # failures = (ms.k * get_node_count(ms,central_infected)) - successes
    return Int(round(successes)),Int(round(failures))
end

function triple_count_noc(ms::EdgeCounts,b::Bool,a::Bool,c::Bool)
    return get_pair_count(ms,a,b)*get_pair_count(ms,b,c)/get_node_count(ms,b)
end


function triple_count_c(ms::EdgeCounts,b::Bool,a::Bool,c::Bool)
    num = get_pair_count(ms,a,b)*get_pair_count(ms,b,c)*get_pair_count(ms,a,c)
    denom = get_node_count(ms,b)*get_node_count(ms,a)*get_node_count(ms,c)
    return (get_N(ms)/ms.k)*(num/denom)
end



function sample_posterior_k_i(C::Float64,ms::EdgeCounts,central_infected::Bool,attached_infected::Bool,alpha,beta)
    k = ms.k
    z_a_b = get_z_a_b(ms,central_infected,attached_infected)
    z_a = get_z_a(ms,central_infected)
    k_i_vec = collect(0:k-1)
    prob_vec = zeros(Float64,k_i_vec)
    update_fn = central_infected ? Clustering.p_i_to_s : Clustering.p_s_to_i
    for (i,k_i) in enumerate(k_i_vec)
        y = (k_i + Clustering.get_infected_indicator(attached_infected) )/ k
        prob_vec[i] = prior_ki_fast(k_i,k-1,z_a,z_a_b,C)*update_fn(y,y^2,alpha,beta)
    end
    prob_vec /= sum(prob_vec)
    ki = sample(k_i_vec, WeightVec(prob_vec))
    return ki,(k-1)-ki
end
    
function prior_ki(ki::Int,kprime::Int,z_a::Float64,z_ab::Float64,C::Float64)
    tot = 0.0
    term = 0.0
    term0 = 0.0
    for j = 0:kprime
        term0 = 0.0
        for i = 0:ki
            term0 += binomial_pdf(j,i,z_ab)*binomial_pdf(kprime-j,ki-i,z_a)
        end
        tot += term0*binomial_pdf(kprime,j,C)
    end
    return tot
end

function prior_ki_fast(ki::Int,kprime::Int,z_a::Float64,z_ab::Float64,C::Float64)
    z_a_b_m = C*z_ab + (1-C)*z_a
    return binomial_pdf(kprime,ki,z_a_b_m)
end

            
function binomial_pdf(n::Int,k::Int,p::Float64)
    return binomial(n,k)*p^k*(1-p)^(n-k)
end   


function is_fixed(ms::EdgeCounts,N,fixation_threshold)
    n = get_num_infecteds(ms)
    return n >= N || n >= fixation_threshold*N || n < 1# ms.mis == 0
end


function run_epidemic_well_mixed_with_clustering(N,k,C,alpha,beta,fixation_threshold=1.0)
    m = N*k

    mii = 0
    mis = k
    ms = EdgeCounts(mii,mis,m,k)
    
    # dt = 0.1/(2*k)
    dt = 0.05#100/(N)
    max_y = -1.0
#     infecteds::Array{Float64,1} = []
    infecteds::Array{EdgeCounts,1} = []
    
    n = get_num_infecteds(ms)
    max_y = n/N 
    fixed=false
    push!(infecteds,copy(ms))

    step = 0
    max_steps = Int(N/dt)
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





function get_p_reach_well_mixed_with_clustering(N,k,C,alpha,beta,num_runs=10,fixation_threshold=1.0)
    runs,maxs = run_epidemics_with_clustering(num_runs, () -> run_epidemic_well_mixed_with_clustering(N,k,C,alpha,beta,fixation_threshold));
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