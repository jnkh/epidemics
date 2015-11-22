using SlurmNodes

N_range = [100,200,400]
nprocs_range = [8,64,200]

for N in N_range
	myfun(N,M) = sum(randn(N,M)^2)
	Nlist = repmat([N],maximum(nprocs_range))
	map(N -> myfun(N,N),[10])#for compilation
	tic()
	map(N -> myfun(N,N),Nlist)
	elapsed_serial = toc()
	for nprocesses in nprocs_range 
		rmprocs(procs()[2:end])
		nl = get_partial_list_of_nodes(nprocesses)
		addprocs(nl)

		@everywhere myfun(N,M) = sum(randn(N,M)^2)



		pmap(N -> myfun(N,N),Nlist)#for compilation
		tic()
		pmap(N -> myfun(N,N),Nlist)
		elapsed_parallel = toc()


		println("N: $N, nprocs: $(nprocs())")
		println("serial: $elapsed_serial, parallel: $elapsed_parallel, speedup: $(elapsed_serial/elapsed_parallel)")
	end
end




############minimal bug example#############
# addprocs(2)

# Nlist = repmat([1000],10)

# #define function to execute
# @everywhere myfun(N,M) = sum(randn(N,M)^2)

# #define some local variable
# @everywhere M = 1000 #will not work without @everywhere!

# #map over curried function: make sure all captured variables are defined @everywhere!
# @time pmap(N -> myfun(N,M),Nlist)
