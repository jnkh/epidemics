module SlurmNodes

export get_list_of_nodes,get_partial_list_of_nodes

#returns a zipped list of tuples of the form (nodename<:AbstractString,num_procs_on_node::Int). This can be fed directly to addprocs like addprocs(get_list_of_nodes()).

function get_partial_list_of_nodes(num_nodes::Int,)
    hostnames,nums = unzip(get_list_of_nodes())
	total_procs = sum(nums)
    if num_nodes >= total_procs return zip(hostnames,nums) end
    
    nums_to_use = []
    i = 1
    while num_nodes > 0
        if num_nodes >= nums[i]
            push!(nums_to_use,nums[i])
            num_nodes -= nums[i]
        else
            push!(nums_to_use,num_nodes)
            num_nodes = 0
        end
        i+= 1
    end
#     while length(nums_to_use) < length(nums)
#         push!(nums_to_use,0)
#     end
    return zip(hostnames[1:length(nums_to_use)],nums_to_use)
end



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



function unzip(input::Vector)
    n = length(input)
    types  = map(typeof, first(input))
    output = map(T->Vector{T}(n), types)

    for i = 1:n
       @inbounds for (j, x) in enumerate(input[i])
           (output[j])[i] = x
       end
    end
    return (output...)
end

end