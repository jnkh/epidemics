using SlurmNodes,JLD

N_range = [100,200,400,800,1200]
nprocs_range = [2,4,8,16,32,64,128,256]

elapsed_serial = zeros(length(N_range))
elapsed_parallel = zeros(length(N_range),length(nprocs_range))

for (i,N) in enumerate(N_range)
	myfun(N,M) = sum(randn(N,M)^2)
	Nlist = repmat([N],maximum(nprocs_range))
	map(N -> myfun(N,N),[10])#for compilation
	tic()
	map(N -> myfun(N,N),Nlist)
	elapsed_serial[i] = toc();
	for (j,nprocesses) in enumerate(nprocs_range)
		rmprocs(procs()[2:end])
		nl = get_partial_list_of_nodes(nprocesses)
		addprocs(nl)

		@everywhere myfun(N,M) = sum(randn(N,M)^2)



		pmap(N -> myfun(N,N),Nlist)#for compilation
		tic()
		pmap(N -> myfun(N,N),Nlist)
		elapsed_parallel[i,j] = toc();


		println("N: $N, nprocs: $(nprocs() == 1 ? nprocs() : nprocs()-1)")
		println("serial: $(elapsed_serial[i]), parallel: $(elapsed_parallel[i,j]), speedup: $(elapsed_serial[i]/elapsed_parallel[i,j])")
	end
end


save("multiprocessing_data/timing.jld","N_range",N_range,"nprocs_range",nprocs_range,"elapsed_serial",elapsed_serial,"elapsed_parallel",elapsed_parallel)




############minimal bug example#############
# addprocs(2)

# Nlist = repmat([1000],10)

# #define function to execute
# @everywhere myfun(N,M) = sum(randn(N,M)^2)

# #define some local variable
# @everywhere M = 1000 #will not work without @everywhere!

# #map over curried function: make sure all captured variables are defined @everywhere!
# @time pmap(N -> myfun(N,M),Nlist)
