module SlurmNodes

export get_list_of_nodes

#returns a zipped list of tuples of the form (nodename<:AbstractString,num_procs_on_node::Int). This can be fed directly to addprocs like addprocs(get_list_of_nodes()).
function get_list_of_nodes()
	println("nodelist: $(ENV["SLURM_NODELIST"])")
	println("cpus_per_node: $(ENV["SLURM_JOB_CPUS_PER_NODE"])")
	s = ENV["SLURM_NODELIST"]
	cpus_per_node = reduce(vcat,map(convert_cpus_strings_to_ints,split(ENV["SLURM_JOB_CPUS_PER_NODE"],',')))
	idx = searchindex(s,"[")
	if idx > 0
		numlist = split(s[idx+1:end-1],",")
		node_numbers = generate_list_of_node_numbers(numlist)
		node_str = s[1:idx-1]
		node_list = [node_str*"$n" for n in node_numbers]
	else
		node_list = [s]
	end
	return collect(zip(node_list,cpus_per_node))
end

function get_integer_list_from_string(s::AbstractString)
	idx = searchindex(s,"(")



function convert_cpus_strings_to_ints(s::AbstractString)
    idx = searchindex(s,"(")
    if idx == 0
        return [parse(Int,s)]
    else
#        println(s[1:idx-1]," ",s[idx+2:searchindex(s,")")-1])
        return repmat([parse(Int,s[1:idx-1])],parse(Int,s[idx+2:searchindex(s,")")-1]))
    end
end


function generate_list_of_node_numbers{S <: AbstractString}(numlist::Array{S,1})
	list_of_nodes = []
 for s in numlist
 	if is_node_range(s)
 	list_of_nodes = vcat(list_of_nodes,get_list_of_nodes_from_range(s))
 	else
 	push!(list_of_nodes,parse(Int,s))
 	end
 end
 return list_of_nodes
end

function is_node_range(s)
  return searchindex(s,"-") > 0
end

function get_list_of_nodes_from_range(node_range)
       lower,upper = map(x -> parse(Int,x),split(node_range,"-"))
       return collect(lower:upper)
end

end