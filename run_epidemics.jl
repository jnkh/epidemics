#Add the number of processors
using SIS,IM,PayloadGraph, Epidemics
using JLD

addprocs(CPU_CORES)

@everywhere begin

using SIS,IM,PayloadGraph, Epidemics

verbose = false

########## Set up model ###############

k = 4
#y_n = 0.1
c_r = 0.18
N = 5000 #1000
n_n = 2000#400#y_n*N
beta = 4.0/(c_r*n_n)
alpha = (N*beta)/n_n
if verbose println(N, ' ' ,alpha, ' ',beta) end

im = InfectionModel(x -> 1 + alpha*x , x -> 1 + beta);
imk = InfectionModel(x -> 1 + beta + get_s_eff(x,alpha,beta,k) , x -> 1 + beta);

y_n, y_minus, y_plus, y_p,critical_determinant = get_parameters(N,alpha,beta,verbose)

########## Set simulation params ###############

num_trials = 100
num_trials_mixed = 10000
fixation_threshold = 2*y_n
regular=true

graph_model = true


end

println("running in parallel on $(numprocs()-1) nodes...")
if graph_model
@time sizes,num_fixed,_,runs = 
run_epidemics_parallel(N,num_trials,im, (N,im)
    -> run_epidemic_graph(N,k,im,regular,fixation_threshold),true);
else
@time sizes,num_fixed,_,runs = 
run_epidemics_parallel(N,num_trials_mixed,im,(N,im)
    -> run_epidemic_well_mixed(N,imk,fixation_threshold),true);
end
println("done")



#=
srand(1)
@time sizes,num_fixed,_,runs = 
run_epidemics_parallel(N,num_trials,im, (N,im)
    -> run_epidemic_graph(N,k,im,regular,fixation_threshold),false);

println( mean(sizes) )

srand(1)


@time sizes,num_fixed,_,runs = 
run_epidemics(N,num_trials,im, (N,im)
    -> run_epidemic_graph(N,k,im,regular,fixation_threshold));

println( mean(sizes) )
=#

params = Dict{AbstractString,Any}("N" => N, "alpha" => alpha, "beta" => beta, "k" => k, "fixation_threshold" => fixation_threshold,"graph_model" => graph_model)

data_dir_path = "/mnt/D/windows/MIT/classes/6/338/project/data/"
filename = "epidemics_$(now()).jld"


save(data_dir_path + filename,"params",params,"sizes",sizes,"runs",runs,"num_fixed",num_fixed)
