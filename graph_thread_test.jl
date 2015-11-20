using SIS,Base.Threads,JLD


function get_timing_threads(N)
	alpha = 0.1
	beta= 0.1

	const f1 = x -> 1 + alpha::Float64*x
	const f2 = x -> 1 + beta::Float64
	im = IM.InfectionModel(f1,f2)

	p = PayloadGraph.Graph(LightGraphs.Graph(N,Int(round(N*(N-1)/2))),zeros(Int,N))

	p.payload[1] = 1
	# p.payload[2] = 1
	# p.payload[5] = 1
	# p.payload[15] = 1
	# p.payload[10] = 1
	new_types = zeros(Int,N)
	num_trials = 100

	println("running on $(nthreads()) threads.")

	tic()
	for i = 1:num_trials
		num = 0
		new_types = zeros(Int,N)
		p.payload *= 0
		p.payload[1] = 1
		while sum(p.payload) > 0 && sum(p.payload) < N && num < 50
			SIS.update_graph_threads_test(p,im,new_types)
			# println(sum(p.payload)/N)
			num +=1 
		end
		print("$(i/num_trials)\r")
		flush(STDOUT)
	end
	elapsed = toc()
	println()
	flush(STDOUT)

	println("elapsed = $elapsed, N = $N")
	return elapsed
end
	# println(p)
# println(new_types)


saving = true
N_range = [100,200,300]
elapsed_range = []
num_threads = nthreads()

for N in N_range
	elapsed = get_timing_threads(N)
	push!(elapsed_range,elapsed)
end

save("thread_data/$(num_threads)_threads.jld","N_range",N_range,"elapsed_range",elapsed_range,"num_threads",num_threads)



println("$N_range\n$elapsed_range\n$num_threads")


