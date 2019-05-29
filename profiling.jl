using Distributed
using Profile
# addprocs()
using Revise
push!(LOAD_PATH, pwd()*"/..")
push!(LOAD_PATH, pwd()*"/.")
import_path = "/Users/julian/Harvard/research/nowak/indirect_rec/src"
push!(LOAD_PATH, import_path)
using Epidemics,IM
using LightGraphs,GraphCreation,GraphGeneration

using Profile
Profile.clear()
plot_beta = false
beta = 0.01
alpha = 0.1
N = 2000#40000
k = 5 #[5,10,12,1000]#[1000,12,10,5]
num_trials = 400
num_trials_th = 1000
num_trials_sim = 1000
in_parallel = false
g = random_regular_graph(N,k) 
graph_type = custom_rg#regular_rg
gi = get_graph_information(graph_type,N=N,k=k,custom_g=g)
@profile sr = get_simulation_result(N,alpha,beta,gi,1,1,in_parallel=in_parallel)
Profile.clear_malloc_data()
@profile sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel)
