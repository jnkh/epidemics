module GraphGeneration

using LightGraphs, Distributions

export get_gamma_params, graph_from_gamma_distribution,
graph_from_degree_distribution

function get_gamma_params(mu,sigma)
    k = mu^2/sigma^2
    theta = sigma^2/mu
    return k,theta
end

function graph_from_gamma_distribution(N::Int,mu_k,sigma_k)
	k,theta = get_gamma_params(mu_k,sigma_k)
	d = Gamma(k,theta)
	return graph_from_degree_distribution(d,N)
end





#############Creates a graph from a prescribed degree distribution using 
#############The stub-connect algorithm with random rewiring to remove
#############Random edges (such as self-edges and duplicate edges).
function graph_from_degree_distribution(d::UnivariateDistribution,N::Int,min_degree=1)
    g = Graph(N)
    
    degrees = sample_degrees(d,N,min_degree)
    stubs = get_stubs(degrees)
    edges = make_edges(stubs)
    edges = remove_invalid_edges(edges)
   
    #produce graph
    while(!isempty(edges))
        edge = pop!(edges)
        add_edge!(g,edge)
    end
    g
end


function get_invalid_edges(edges)
    duplicates = get_duplicates(edges)
    self_edges = get_self_edges(edges)
    return vcat(duplicates,self_edges)
end

function get_valid_edges(edges)
    valid_edges = []
    for edge in edges
        if ~ is_self_edge(edge)
            push!(valid_edges,edge)
        end
    end
    return unique(valid_edges)
end

function get_duplicates(arr)
    duplicates = []
    arr = sort(arr)
    for i in 1:length(arr)-1
        curr = arr[i]
        next = arr[i+1]
        if next == curr
            if ~is_self_edge(next)
                push!(duplicates,next)
            end
        end
    end
    duplicates
end 

function is_self_edge(edge)
    return edge[1] == edge[2]
end

function get_self_edges(edges)
    self_edges = []
    for edge in edges 
        if is_self_edge(edge)
            push!(self_edges,edge)
        end
    end
    self_edges   
end
    

function remove_invalid_edges(edges)
    invalid_edges = get_invalid_edges(edges)
    unique_edges = get_valid_edges(edges)
    # println("invalid: $(length(invalid_edges)), unique: $(length(unique_edges)), total: $(length(edges))")
#     println("removing $(length(invalid_edges)) / $(length(unique_edges)) invalid edges")
    for dup_edge in invalid_edges
        unique_edges = rewire_edges(unique_edges,dup_edge)
    end
    assert(length(get_invalid_edges(unique_edges))==0)
    return unique_edges
end

function rewire_edges(unique_edges,dup_edge)
    edge = dup_edge
    rand_idx = 0
    while true
        rand_idx = rand(1:length(unique_edges))
        edge = unique_edges[rand_idx]
        if valid_swap(dup_edge,edge,unique_edges)
            break
        end
    end
    e1,e2 = swap_edges(edge,dup_edge)
    unique_edges[rand_idx] = e1
    push!(unique_edges,e2)
    return unique_edges
end

function swap_edges(e1,e2)
    e3 = Pair(e1[1],e2[2])
    e4 = Pair(e2[1],e1[2])
    return e3,e4
end


function valid_swap(e_invalid,e_valid,unique_edges)
    
    if e_valid[1] == e_invalid[1] || e_valid[2] == e_invalid[2] return false end
    if e_valid[1] == e_invalid[2] || e_valid[2] == e_invalid[1] return false end
    e3,e4 = swap_edges(e_invalid,e_valid)
    assert((~ is_self_edge(e3)) && (~is_self_edge(e4)))
    if e3 in unique_edges || e4 in unique_edges || reverse(e3) in unique_edges || reverse(e4) in unique_edges
      return false
    end
    return true
end


function sample_integers(d::UnivariateDistribution,N::Int,min_degree=1)
    samples = rand(d,N)
    return [Int(max(el,min_degree)) for el in round(samples)]
end

function sample_degrees(d::UnivariateDistribution,N::Int,min_degree=1)
    degrees = sample_integers(d,N)
    #make sure sum is even
    while sum(degrees) % 2 != 0
        degrees = sample_integers(d,N)
    end
    degrees
end

function get_stubs(degrees)
    stubs = Array(Int,sum(degrees))
    idx = 1
    for i = 1:length(degrees)
        for j = 0:degrees[i]-1
            stubs[idx + j] = i
        end
        idx = idx + degrees[i]
    end
    return shuffle(stubs)
end

function make_ordered_pair(first::Int,second::Int)
    first <= second ? Pair(first,second) : Pair(second,first)
end

function make_edges(stubs)
    edges = [] 
    for i = 1:Int(length(stubs)/2)
        p = make_ordered_pair(stubs[2*i],stubs[2*i-1])
        push!(edges,p)
    end
    return edges
end

end