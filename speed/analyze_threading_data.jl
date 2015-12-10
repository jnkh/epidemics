using JLD

num_threads_range = [1,2,3,4,6,8,12,16,32,64]
d = load("thread_data/1_threads.jld")

N_range =  d["N_range"]

total_elapsed_range = zeros((length(num_threads_range),length(N_range))) 

for (i,num_threads) in enumerate(num_threads_range)
	d = load("thread_data/$(num_threads)_threads.jld")

	N_range =  d["N_range"]
	total_elapsed_range[i,:] =  d["elapsed_range"]
	num_threads = d["num_threads"]
end


println(N_range)

println(total_elapsed_range)