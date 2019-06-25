__precompile__()
module GraphGeneration

using LightGraphs, Distributions,Munkres,Random,LinearAlgebra

export get_gamma_params, graph_from_gamma_distribution,
graph_from_degree_distribution,
graph_from_two_degree_distribution,
graph_given_degrees,
compute_two_degree_params,
compute_k_sigma_k,
TwoDegreeParams,
get_p_k_two_degree,
regular_clustering_graph,
change_num_triangles!,
change_clustering_by_swapping,
get_mutual_neighbors,
community_overlap,
randomize_labels



function get_gamma_params(mu,sigma)
    k = mu^2/sigma^2
    theta = sigma^2/mu
    return k,theta
end

function graph_from_gamma_distribution(N::Int,mu_k,sigma_k,min_degree=3)
	k,theta = get_gamma_params(mu_k,sigma_k)
	d = Gamma(k,theta)
	return graph_from_degree_distribution(d,N,min_degree)
end


####################################################
##############Two Degree Distribution###############
####################################################
#Create a graph with a degree distribution that 
#allows only 2 degrees
mutable struct TwoDegreeParams
    k1::Int
    k2::Int
    p1::Float64
    p2::Float64
end

function graph_from_two_degree_distribution(N::Int,kbar,sigmak,min_degree=3)
    tdp = compute_two_degree_params(kbar,sigmak)
    if tdp.k2 > N-1 || tdp.k1 > N -1
        println("-----------Error--------------")
        println("Invalid 2 degree params:")
        println(tdp)
        println("k_bar: $kbar, sigma_k: $sigmak")
        println("------------------------------") 
        if tdp.k1 == tdp.k2
            println("ABORTING: kbar too high!")
            return 0
        end
    end
    if tdp.k1 != tdp.k2
        tdp.k2 = min(tdp.k2,N-1)
        tdp.k1 = min(tdp.k1,N-1)
        tdp = TwoDegreeParams(kbar,tdp.k1,tdp.k2)
    end
    return graph_from_two_degree_distribution(tdp,N)
end
    

function graph_from_two_degree_distribution(tdp::TwoDegreeParams,N::Int)
    degrees = get_degrees_from_p_k(get_p_k_two_degree(tdp),N)
    return graph_given_degrees(degrees,N)
end
    

function get_p_k_two_degree(tdp::TwoDegreeParams)
    p_1 = tdp.p1
    p_2 = tdp.p2
    k_1 = tdp.k1
    k_2 = tdp.k2
    k_bar = p_1*k_1 + p_2*k_2
    sigma_k = (p_1*k_1^2 + p_2*k_2^2 - k_bar^2)^(0.5)
    # println("k_bar: $(k_bar), sigma_k: $(sigma_k)")
    function p_k(x)
        val = 0
        if k_1 - 0.5 < x < k_1 + 0.5
            val += p_1
        end
        if k_2 - 0.5 < x < k_2 + 0.5
            val += p_2
        end
        return val
    end
    return p_k
end

function get_degrees_from_p_k(p_k,N)
    probs = zeros(N-1)
    for i = 1:length(probs)
        probs[i] = p_k(i)
    end
    d = Multinomial(N,probs)
    degree_counts = rand(d)
    while dot(degree_counts,collect(1:length(degree_counts))) %2 != 0
        degree_counts = rand(d)
    end
    degrees = zeros(Int,N)
    curr_idx = 1
    for (i,count) in enumerate(degree_counts)
        while degree_counts[i] > 0
            degrees[curr_idx] = i
            degree_counts[i] -= 1
            curr_idx += 1
        end
    end
    return degrees
end


function compute_two_degree_params(kbar,sigmak,min_degree = 3)
    sigmak = Int(round(sigmak))
    k1 = kbar - sigmak
    k2 = kbar + sigmak
    p1 = 0.5
    p2 = 0.5
    tdp = TwoDegreeParams(k1,k2,p1,p2)
    if k1 < min_degree
        k1 = min_degree
        tdp = compute_params_given_k1(k1,kbar,sigmak)
    end
    return tdp
end

function compute_params_given_k1(k1,kbar,sigmak)
    k2bar = kbar^2 + sigmak^2
    a = kbar - k1
    b = k1^2 - k2bar
    c = k1*k2bar - kbar*k1^2
    k2raw = (-b + (b^2 - 4*a*c)^0.5)/(2*a)
    k2 = Int(round(k2raw))
    return TwoDegreeParams(kbar,k1,k2)
end

function TwoDegreeParams(kbar::Int,k1::Int,k2::Int)
    p1 = (k2 - kbar)/(k2 - k1)
    p2 = (kbar - k1)/(k2 - k1)
    return TwoDegreeParams(k1,k2,p1,p2)
end

function compute_k_sigma_k(tdp::TwoDegreeParams)
    p1 = tdp.p1
    p2 = tdp.p2
    k1 = tdp.k1
    k2 = tdp.k2
    
    k_bar = p1*k1 + p2*k2
    sigma_k = (p1*k1^2 + p2*k2^2 - k_bar^2)^(0.5)
    println("k_bar: $(k_bar), sigma_k: $(sigma_k)")
end



############################################
########Edge Rewiring For Clustering########
############################################


function swap_edges_c(e1,e2)
    e3 = Edge(e1.src,e2.dst)
    e4 = Edge(e2.src,e1.dst)
    return e3,e4
end


function valid_swap_c(G,v1,v2,w1,w2)
    if w1 == w2 return false end
    if length(unique([v1,v2,w1,w2])) != 4 return false end
        
    e1 = Edge(v1,w1)
    e2 = Edge(v2,w2)
    e1s,e2s = swap_edges_c(e1,e2)
    
    if has_edge(G,e1s) || has_edge(G,e2s) return false end
    return true
end
 



function form_triangle!(G)
    success = false
    while !success
        v0 = sample(vertices(G))
        v1,v2 = sample(neighbors(G,v0),2,replace=false)
        w1 = v0
        w2 = v0
        while w1 == v0 || w2 == v0
            w1 = sample(neighbors(G,v1))
            w2 = sample(neighbors(G,v2))
        end
        if valid_swap_c(G,v1,v2,w1,w2)
            e1 = Edge(v1,w1)
            e2 = Edge(v2,w2)
            e1s,e2s = swap_edges_c(e1,e2)
            @assert(has_edge(G,e1))
            @assert(has_edge(G,e2))
            @assert(!has_edge(G,e1s))
            @assert(!has_edge(G,e2s))
            @assert(v1 != v2)
            
            C_before = mean([local_clustering_coefficient(G,vloc) for vloc in [v1,v2,w1,w2]])
            rem_edge!(G,e1)
            rem_edge!(G,e2)
            add_edge!(G,e1s)
            add_edge!(G,e2s)
            C_after = mean([local_clustering_coefficient(G,vloc) for vloc in [v1,v2,w1,w2]])
            #undo if bad change
            if C_after <= C_before
                rem_edge!(G,e1s)
                rem_edge!(G,e2s)
                add_edge!(G,e1)
                add_edge!(G,e2)
            else
                success = true
            end
        end
    end
end


function form_triangle_simple!(G)
    change_num_triangles!(G,true)
end

function get_valid_set_of_edges(G)
    while true
        v1,v2 = sample(vertices(G),2,replace=false)
        n1s = collect(neighbors(G,v1))
        n2s = collect(neighbors(G,v2))
        max_trials = 2*(length(n1s) + length(n2s))
        if length(n1s) == 0 || length(n2s) == 0
            continue
        end
        w1 = v2
        w2 = v1
        trial = 0
        while (w1 == v2 || w2 == v1) && trial <= max_trials
            # println("trial: $trial / $(max_trials), w1: $w1, w2: $w2")
            w1 = sample(n1s)
            w2 = sample(n2s)
            trial += 1
        end
        if trial <= max_trials
            # print("valid set after $(trial) trials ($(max_trials) max)")
            @assert(w1 != v2 && w2 != v1)
            return v1,v2,w1,w2
        # else
            # pass
            # print("exceeded $(max_trials) trials. $(length(n1s)), $(length(n2s))")
        end
    end
end



function get_mutual_neighbors(g::LightGraphs.Graph,v1::Int,v2::Int)
    neighbors1 = LightGraphs.neighbors(g,v1)
    neighbors2 = LightGraphs.neighbors(g,v2)
    return intersect(neighbors1,neighbors2)
end

get_mutual_neighbors(g::LightGraphs.Graph,e::Edge) = get_mutual_neighbors(g,e.src,e.dst)


function get_nodes_affected_by_swap(g::LightGraphs.Graph,e1::Edge,e2::Edge)
    return union([e1.src,e1.dst,e2.src,e2.dst],get_mutual_neighbors(g,e1.src,e1.dst),get_mutual_neighbors(g,e2.src,e2.dst),
        get_mutual_neighbors(g,e1.src,e2.dst),get_mutual_neighbors(g,e2.src,e1.dst))
end


function change_num_triangles!(G,increase=true)
    success = false
    N = nv(G)
    max_trials = 1e5#nv(G)^2
    trial = 0
    epsilon = 0.1/length(edges(G))^2
    C_diff = 0.0
    while !success && trial < max_trials
        v1,v2,w1,w2 = get_valid_set_of_edges(G)
        if valid_swap_c(G,v1,v2,w1,w2)
            e1 = Edge(v1,w1)
            e2 = Edge(v2,w2)
            involved_vertices = get_nodes_affected_by_swap(G,e1,e2) #[v1,v2,w1,w2]
            e1s,e2s = swap_edges_c(e1,e2)
            
            C_before = mean(local_clustering_coefficient(G,involved_vertices))
            rem_edge!(G,e1)
            rem_edge!(G,e2)
            add_edge!(G,e1s)
            add_edge!(G,e2s)
            C_after = mean(local_clustering_coefficient(G,involved_vertices))
            #undo if bad change
            if (increase && C_after <= C_before+epsilon) || (~increase && C_after >= C_before-epsilon) 
                rem_edge!(G,e1s)
                rem_edge!(G,e2s)
                add_edge!(G,e1)
                add_edge!(G,e2)
            else
                success = true
                C_diff = (C_after - C_before)*length(involved_vertices)/N
                # println("C: $(mean(local_clustering_coefficient(G))) ")
            end
            # println("success: $(success)")
            # println(" ",mean(local_clustering_coefficient(G)))
        end
        trial += 1
    end
    # println("success $(success) after $(trial) trials")
    return success,C_diff
end

function change_clustering_by_swapping(G,C)
    C_curr = mean(local_clustering_coefficient(G))
    k = mean(degree(G))
    N = nv(G)
    increase = C_curr < C
    if C_curr == C
        return G
    end
    iters = 1
    tot_iters = 0
    while (increase && C_curr < C) || (~increase && C_curr > C)
        # iters = Int(ceil(0.5*abs(C_curr - C)/(1.0/(N*k)))) #expected absolute change
        # println(iters)
        # println(C_curr)
        # for i = 1:iters
        succ,C_diff = change_num_triangles!(G,increase)
        C_curr += C_diff

        if !succ
            print("Warning: Clustering adjustment didn't converge")
            return G
        end
        tot_iters += 1
        # end
        # C_curr = mean(local_clustering_coefficient(G))
    end
    @assert(abs(C_curr - mean(local_clustering_coefficient(G))) < 1e-5)
    return G
end

function regular_clustering_graph(N,k,C)
    G = LightGraphs.random_regular_graph(N,k)
    return change_clustering_by_swapping(G,C)
end

function community_overlap(ls1::Array{Int,1},ls2::Array{Int,1})
    ls1 = randomize_labels(ls1)
    ls2 = randomize_labels(ls2)
    @assert(maximum(ls1) <= length(ls1))
    @assert(maximum(ls2) <= length(ls2))
    cs1 = unique(ls1)
    cs2 = unique(ls2)
    mat = construct_overlap_matrix(ls1,ls2,cs1,cs2)
#     mat_modified[mat .== 0]
    matching = munkres(-1*mat)
    total_overlap = sum([mat[i,matching[i]] for i in 1:length(matching) if ~isnothing(matching[i])])
#     o,ls1,ls2,d = community_overlap(ls1,ls2)
#     total_overlap = sum([mat[i,d[i]] for i in 1:length(matching)])
    return 1.0*total_overlap/length(ls1),[matching[x] for x in ls1],ls2,matching
end

function construct_overlap_matrix(ls1,ls2,cs1,cs2)
#     n = maximum([length(cs1),length(cs2)])
    n1,n2 = length(cs1),length(cs2)
    mat = zeros(Int,n1,n2)
    for i = 1:n1
        for j = 1:n2
#             if i > length(cs1) || j > length(cs2)
#                 mat[i,j] = -Inf
#             end
            mat[i,j] = overlap(ls1,i,ls2,j)
        end
    end
    return mat
end

function community_overlap(g::Graph,g1::Graph,num_trials = 20)
    max_overlap = 0.0
    max_ls1 = []
    max_ls2 = []
    for i in 1:num_trials
        ls1 = label_propagation(g)[1]
        ls2 = label_propagation(g1)[1]
        o,ls1_new,ls2_new = community_overlap(ls1,ls2)
        if o > max_overlap
            max_overlap = o
            max_ls1 = ls1_new
            max_ls2 = ls2_new
        end
    end
    return max_overlap,max_ls1,max_ls2
end


function overlap(ls1,idx1,ls2,idx2)
    return sum((ls1 .== idx1) .& (ls2 .== idx2))
end

function community_size(ls,idx)
    return sum(ls .== idx)
end

function randomize_labels(ls_vec)
    unique_labels = sort(unique(ls_vec))
    rand_labels = shuffle(unique_labels)
    d = Dict(unique_labels[i] => rand_labels[i] for i in 1:length(unique_labels))
    ls_vec_new = [d[el] for el in ls_vec]
    return ls_vec_new
end


#############Creates a graph from a prescribed degree distribution using 
#############The stub-connect algorithm with random rewiring to remove
#############Random edges (such as self-edges and duplicate edges).


function graph_given_degrees(degrees::Array{Int,1},N::Int)
    g = Graph(N)
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


function graph_from_degree_distribution(d::UnivariateDistribution,N::Int,min_degree=1)
    degrees = sample_degrees(d,N,min_degree)
    return graph_given_degrees(degrees,N)  

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

import Base.isless,Base.isequal,Base.==
function isless(e1::Edge,e2::Edge)
    isless(edge_to_tuple(e1),edge_to_tuple(e2))
end

function isequal(e1::Edge,e2::Edge)
    isequal(edge_to_tuple(e1),edge_to_tuple(e2))
end

function ==(e1::Edge,e2::Edge)
    ==(edge_to_tuple(e1),edge_to_tuple(e2))
end


function edge_to_tuple(e::Edge)
    if e.src < e.dst
        return (e.src,e.dst)
    else
        return (e.dst,e.src)
    end
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
    return edge.src == edge.dst
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
    @assert(length(get_invalid_edges(unique_edges))==0)
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
    e3 = Edge(e1.src,e2.dst)
    e4 = Edge(e2.src,e1.dst)
    return e3,e4
end


function valid_swap(e_invalid,e_valid,unique_edges)
    
    if e_valid.src == e_invalid.src || e_valid.dst == e_invalid.dst return false end
    if e_valid.src == e_invalid.dst || e_valid.dst == e_invalid.src return false end
    e3,e4 = swap_edges(e_invalid,e_valid)
    @assert((~ is_self_edge(e3)) && (~is_self_edge(e4)))
    if e3 in unique_edges || e4 in unique_edges || reverse(e3) in unique_edges || reverse(e4) in unique_edges
      return false
    end
    return true
end


function sample_integers(d::UnivariateDistribution,N::Int,min_degree=1)
    samples = rand(d,N)
    return [Int(max(el,min_degree)) for el in round.(samples)]
end

function sample_degrees(d::UnivariateDistribution,N::Int,min_degree=1)
    degrees = sample_integers(d,N,min_degree)
    #make sure sum is even
    while sum(degrees) % 2 != 0
        degrees = sample_integers(d,N,min_degree)
    end
    degrees
end

function get_stubs(degrees)
    stubs = zeros(Int,sum(degrees))
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
    first <= second ? Edge(first,second) : Edge(second,first)
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