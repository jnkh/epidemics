using SlurmNodes

nl = get_partial_list_of_nodes(200)
addprocs(nl)

@everywhere N = 400
Nlist = repmat([N],200)
@everywhere myfun(N,M) = sum(randn(N,M)^2)
@everywhere M = N #doens't work without @everywhere!


tic()
map(N -> myfun(N,M),Nlist)
elapsed_serial = toc()

pmap(N -> myfun(N,M),Nlist)
tic()
pmap(N -> myfun(N,M),Nlist)
elapsed_parallel = toc()


println("serial: $elapsed_serial, parallel: $elapsed_parallel, speedup: $(elapsed_serial/elapsed_parallel)")




############minimal bug example#############
# addprocs(2)

# Nlist = repmat([1000],10)

# #define function to execute
# @everywhere myfun(N,M) = sum(randn(N,M)^2)

# #define some local variable
# @everywhere M = 1000 #will not work without @everywhere!

# #map over curried function: make sure all captured variables are defined @everywhere!
# @time pmap(N -> myfun(N,M),Nlist)