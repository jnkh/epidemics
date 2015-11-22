using SlurmNodes

N = 400
nl = get_partial_list_of_nodes(200)
addprocs(nl)

Nlist = repmat([N],200)
@everywhere myfun(N,M) = sum(randn(N,M)^2)
@everywhere M = N #doens't work without @everywhere!
@time pmap(N -> myfun(N,M),Nlist)
@time map(N -> myfun(N,M),Nlist)




############minimal bug example#############
# addprocs(2)

# Nlist = repmat([1000],10)

# #define function to execute
# @everywhere myfun(N,M) = sum(randn(N,M)^2)

# #define some local variable
# @everywhere M = 1000 #will not work without @everywhere!

# #map over curried function: make sure all captured variables are defined @everywhere!
# @time pmap(N -> myfun(N,M),Nlist)