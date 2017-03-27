#= This module uses code by Erik Volz, free under licence, as described in the paper
<p>Title: Random Clustering Network</p>
 * <p>Description: Constructs a random network as described in:<br>
 * Erik Volz, "Random graphs with tunable degree distribution and clustering .",
 * cond-mat/0405381. </p>

 http://www.erikvolz.info/home/clustering

 It uses an executable of this code to generate a graph of arbitrary degree distribution and 
 average clustering coefficient.

 @Author Julian Kates-Harbeck, 2016
 =#


module RandomClusteringGraph

using GraphCreation
using Distributions, LightGraphs

export random_clustering_graph, generate_degree_sequence

function generate_degree_sequence(d::Distribution,N::Int)
    initial_degrees = rand(d,N)
    degrees = zeros(Int,length(initial_degrees))
    for i in 1:length(degrees)
        deg = initial_degrees[i]
        #make sure degree is not greater than number of nodes
        while deg > N-1
            deg = rand(d)
        end
        #make sure degree is nonnegative
        deg = max(0,deg)
        #make sure degree is integer
        deg = Int(round(Int,deg))
        degrees[i] = deg
    end
    degrees
end

function random_clustering_graph(degree_dist::Distribution,N::Int,C::AbstractFloat,delete_out=true,out_filename=nothing)
    degs = generate_degree_sequence(degree_dist,N)
    return random_clustering_graph(degs,N,C,delete_out,out_filename)
end

function random_clustering_graph(degs::Array{Int,1},N::Int,C::AbstractFloat,delete_out = true,out_filename=nothing)
    deg_seq_filename = "temp_degree_sequence_$(now()).dat"
    writedlm(deg_seq_filename,degs,'\t')
    if out_filename == nothing
        out_filename = "tgf_out_$(now()).dat"
    end
    run(`java -jar RandomClusteringNetwork.jar $deg_seq_filename $N $C $out_filename`)
    if isfile(deg_seq_filename)
        run(`rm $(deg_seq_filename)`) 
    end
    G =  read_edgelist(out_filename)
    if delete_out && isfile(out_filename)
        run(`rm $(out_filename)`) 
    end
    return G

end

end 