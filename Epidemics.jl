module Epidemics

using SIS,Distributions, IM, LightGraphs,PayloadGraph

export run_epidemic_graph,run_epidemic_well_mixed,get_s_eff,run_epidemics_parallel,run_epidemics


function graph_is_connected(g::LightGraphs.Graph)
    parents = LightGraphs.dijkstra_shortest_paths(g,1).parents[2:end]
    return size(parents) == 0 || minimum(parents) > 0
end

function guarantee_connected(graph_fn)
    g = graph_fn()
    while(!graph_is_connected(g))
        g = graph_fn()
    end
    return g
end
    
function run_epidemic_graph(N::Int,k::Int,im::InfectionModel,regular=false,fixation_threshold=1.0)
    fixed=false
    if regular
        g = guarantee_connected( () -> LightGraphs.random_regular_graph(N,k))
    else
        g = guarantee_connected( () -> LightGraphs.erdos_renyi(N,1.0*k/(N-1)))
    end
    p = create_graph_from_value(g,SUSCEPTIBLE)
    infecteds::Array{Float64,1} = []
    set_payload(p,1,INFECTED)
    frac = get_fraction_of_type(p,INFECTED)
    push!(infecteds,N*frac)

    while frac > 0
        if !(frac < 1 && frac < fixation_threshold)
            fixed = true
            break
        end
        update_graph(p,im)
        frac = get_fraction_of_type(p,INFECTED)
        push!(infecteds,N*frac)

    end
    return infecteds,fixed
end

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
    return infecteds,fixed
end

function update_n(n::Int,N::Int,im::InfectionModel)
    y = n/N
    delta_n_plus = rand(Binomial(n,(1-y)*p_birth(im,n/N)))
    delta_n_minus = rand(Binomial(n,(1-y)*p_death(im,n/N)))
    return n + delta_n_plus - delta_n_minus
end
    
function run_epidemics(N::Int,num_runs::Int,im::InfectionModel,run_epidemic_fn)  
    runs = []
    num_fixed = 0
    sizes = zeros(num_runs)
    total_length =0
    
    for i in 1:num_runs
        infecteds,fixed = run_epidemic_fn(N,im)
        push!(runs,infecteds)
        if fixed
            num_fixed += 1
            sizes[i] = Inf
        else
            sizes[i]= im.dt*sum(infecteds)
            total_length += size(infecteds)[1]
        end
        
    end
    #get rid of fixed ones
    sizes = sizes[sizes .< Inf]

    return sizes,num_fixed,total_length,runs
end


function run_epidemics_parallel(N::Int,num_runs::Int,im::InfectionModel,run_epidemic_fn,parallel=true)  
    num_fixed = 0
    sizes = zeros(num_runs)
    fixed_array = Array{Bool,1}(num_runs)
    total_length =0
    
    mapfn = parallel ? pmap : map 
    runs = mapfn(_ -> run_epidemic_fn(N,im),1:num_runs)
    runs,fixed_array = unzip(runs)
    
    for i in 1:num_runs
        if fixed_array[i]
            num_fixed += 1
            sizes[i] = Inf
        else
            sizes[i]= im.dt*sum(runs[i])
            total_length += size(runs[i])[1]
        end
    end
    
    #get rid of fixed ones
    sizes = sizes[sizes .< Inf]

    return sizes,num_fixed,total_length,runs
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

s(y,alpha,beta) = alpha*y - beta

get_y_eff(y,k) = y.*(1 + (1-y)./(y*k))

get_s_eff(y,alpha,beta,k) = alpha*get_y_eff(y,k) - beta

end