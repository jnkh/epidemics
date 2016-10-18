module Epidemics

using SIS,Distributions, IM, LightGraphs,PayloadGraph, Dierckx

export run_epidemic_graph,run_epidemic_well_mixed,run_epidemics_parallel,
run_epidemics,s,get_s_eff,normed_distribution, P_w_th,get_y_eff,
EpidemicRun, get_sizes, get_num_fixed,GraphInformation,
get_dt_two_level,run_epidemic_well_mixed_two_level, update_n_two_level,
get_p_reach, CompactEpidemicRuns

function graph_is_connected(g::LightGraphs.Graph)
    parents = LightGraphs.dijkstra_shortest_paths(g,1).parents[2:end]
    return size(parents) == 0 || minimum(parents) > 0
end

function guarantee_connected(graph_fn)
    g = graph_fn()
    resampled = 0
    while(!graph_is_connected(g))
        g = graph_fn()
        resampled += 1
    end
    if resampled > 0
        println("Resampled Graph $resampled times.")
    end
    return g
end

type GraphInformation
    graph_fn::Function
    graph::LightGraphs.Graph
    carry_by_node_info::Bool
    data
end

function GraphInformation()
    return GraphInformation(_ -> _,LightGraphs.Graph(),false,nothing)
end

type EpidemicRun
    infecteds_vs_time::Array{Float64,1}
    size::Float64
    fixed::Bool
    infecteds_by_nodes_vs_time::Array{Array{Int,1},1}
    graph_information::GraphInformation
end

type CompactEpidemicRuns
    sizes::Array{Float64,1}
    y_reach::Array{Float64,1}
    p_reach::Array{Float64,1}
end

function CompactEpidemicRuns(runs::Array{EpidemicRun,1},N::Int)
    sizes = get_sizes(runs)
    y_reach,p_reach = get_p_reach(runs,N)
    return CompactEpidemicRuns(sizes,y_reach,p_reach)
end

function EpidemicRun(infecteds_vs_time::Array{Float64,1},size::Float64,fixed::Bool)
    return EpidemicRun(infecteds_vs_time,size,fixed,[],GraphInformation())
end

function get_sizes(runs::Array{EpidemicRun,1})
    return filter(x -> x < Inf,[e.size for e in runs])
end

function get_num_fixed(runs::Array{EpidemicRun,1})
    sum = 0
    for r in runs
        if r.fixed
            sum += 1
        end
    end
    return sum
end

function get_max_reach(run::EpidemicRun)
    return maximum(run.infecteds_vs_time)
end

function get_max_reaches(runs::Array{EpidemicRun, 1})
    return map(get_max_reach,runs)
end

function get_p_reach(runs::Array{EpidemicRun, 1})
    max_reaches = get_max_reaches(runs)
    sorted_max_reaches = sort(max_reaches)
    xvals = unique(sorted_max_reaches)
    yvals = zeros(xvals)
    for r in sorted_max_reaches
        idx = findfirst(xvals,r)
        yvals[1:idx] += 1
    end

    return xvals,yvals/length(max_reaches)
end


function get_p_reach(runs::Array{EpidemicRun, 1},N::Real)
    xvals,yvals = get_p_reach(runs)
    return xvals/N,yvals
end




### Epidemic on a Graph ###
function run_epidemic_graph(N::Int,im::InfectionModel,graph_information::GraphInformation,fixation_threshold=1.0)
    fixed=false
    #construct graph
    g = guarantee_connected(graph_information.graph_fn)
#     graph_information.graph = g
    carry_by_node_info::Bool = graph_information.carry_by_node_info

    #create payload graph
    p = create_graph_from_value(g,SUSCEPTIBLE)
    infecteds::Array{Float64,1} = []
    infecteds_by_nodes::Array{Array{Int,1},1} = []


    set_payload(p,1,INFECTED)
    frac = get_fraction_of_type(p,INFECTED)
    push!(infecteds,N*frac)
    if carry_by_node_info
        push!(infecteds_by_nodes,copy(get_payload(p)))
    end

    new_types = fill(SUSCEPTIBLE,N)# convert(SharedArray,fill(SUSCEPTIBLE,N))

    while frac > 0
        if !(frac < 1 && frac < fixation_threshold)
            fixed = true
            break
        end
        update_graph(p,im,new_types)
        frac = get_fraction_of_type(p,INFECTED)
        push!(infecteds,N*frac)
        if carry_by_node_info
            push!(infecteds_by_nodes,copy(get_payload(p)))
        end
    end

    size = im.dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,fixed,infecteds_by_nodes,graph_information)
end


### Well Mixed Case ###
function run_epidemic_well_mixed(N,im,fixation_threshold=1.0)
    infecteds::Array{Float64,1} = []
    n = 1
    fixed=false
    push!(infecteds,n)


    while n > 0
        if !(n < N && n < N*fixation_threshold)
            fixed = true
            break
        end
        n = update_n(n,N,im)
        push!(infecteds,n)
    end

    size = im.dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,fixed)
end

function update_n(n::Int,N::Int,im::InfectionModel)
    y = n/N
    delta_n_plus = rand(Binomial(n,(1-y)*p_birth(im,n/N)))
    delta_n_minus = rand(Binomial(n,(1-y)*p_death(im,n/N)))
    return n + delta_n_plus - delta_n_minus
end



### Well Mixed Case Two Level###
### Well Mixed Case Two Level###
function run_epidemic_well_mixed_two_level(dt::AbstractFloat,N::Int,y_susc_fn,y_sq_susc_fn,y_inf_fn,y_sq_inf_fn,alpha::AbstractFloat,beta::AbstractFloat,fixation_threshold=1.0)
    infecteds::Array{Float64,1} = []
    n = 1
    fixed=false
    push!(infecteds,n)

    while n > 0
        if !(n < N && n < N*fixation_threshold)
            fixed = true
            break
        end
        n = update_n_two_level(dt,n,N,y_susc_fn,y_sq_susc_fn,y_inf_fn,y_sq_inf_fn,alpha,beta)
        push!(infecteds,n)
    end

    size = dt*sum(infecteds)
    if fixed
        size = Inf
    end

    return EpidemicRun(infecteds,size,fixed)
end

function get_dt_two_level(alpha::AbstractFloat,beta::AbstractFloat)
    desired_rate = 0.1
    max_rate = maximum([(1 + alpha),(1 + beta)])
    dt = desired_rate/max_rate
    return float(dt)
end

function update_n_two_level(dt::AbstractFloat,n::Int,N::Int,y_susc_fn,y_sq_susc_fn,y_inf_fn,y_sq_inf_fn,alpha::AbstractFloat,beta::AbstractFloat)
    y = n/N
    y_susc = evaluate(y_susc_fn,y)
    y_inf = evaluate(y_inf_fn,y)
    y_sq_susc = evaluate(y_sq_susc_fn,y)
    y_sq_inf = evaluate(y_sq_inf_fn,y)

    prob_birth = (y_susc + y_sq_susc*alpha)*dt
    prob_death = (1 - y_inf)*(1 + beta)*dt

    if prob_death < 0
        prob_death = 0
    end

    if prob_birth < 0
        prob_birth = 0
    end

    if prob_birth == 0
        delta_n_plus = 0
    else
       delta_n_plus = rand(Binomial(N-n,prob_birth))
    end

    if prob_death == 0
        delta_n_minus = 0
    else
        delta_n_minus = rand(Binomial(n,prob_death))
    end

    return n + delta_n_plus - delta_n_minus

end




###performing many runs###

function run_epidemics(num_runs::Int,run_epidemic_fn)
    runs = EpidemicRun[]

    for i in 1:num_runs
        run = run_epidemic_fn()
        push!(runs,run)
    end
    #get rid of fixed oneus

    return runs
end


function run_epidemics_parallel(num_runs::Int,run_epidemic_fn,parallel=true)

    mapfn = parallel ? pmap : map
    ret = mapfn( _ -> run_epidemic_fn() ,1:num_runs)
    for val in ret
        if isa(val,RemoteException)
            throw(val)
        end
    end
    runs::Array{EpidemicRun,1} = ret
    return runs
end


########### Utility Functions ##################


function unzip{A,B}(zipped::Array{Tuple{A,B},1})
    l = length(zipped)
    a = Array{A,1}(l)
    b = Array{B,1}(l)
    for (i,(a_el,b_el)) in enumerate(zipped)
        a[i] = a_el
        b[i] = b_el
    end
    return a,b
end

function unzip(zipped)
    l = length(zipped)
    a = []
    b = []
    for (i,(a_el,b_el)) in enumerate(zipped)
        push!(a,a_el)
        push!(b,b_el)
    end
    return a,b
end






function normed_distribution(x,px)
    return px./sum(diff(x).*px[1:end-1])
end

P_w_th(w,s) = exp(-s.^2.*w./4).*w.^(-1.5)./(2*sqrt(pi).*(1 .+ s))

function s(y,alpha::Int,beta::Int)
    return alpha*y - beta
end

function get_y_eff(y,k::Int)
    return y.*(1 + (1-y)./(y*k))
end

function get_s_eff(y,alpha::Float64,beta::Float64,k::Int)
    return alpha*get_y_eff(y,k) - beta
end

function get_c_r(N,alpha,beta)
    return 4*alpha/(beta^2*N)
end

function get_n_n(N,alpha,beta)
    return beta/alpha*N
end

end
