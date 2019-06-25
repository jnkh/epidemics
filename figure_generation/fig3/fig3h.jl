###Preamble###
using Distributed
import_path_desai = "/n/home07/juliankh/physics/research/desai/epidemics/src"
push!(LOAD_PATH, import_path_desai)
in_parallel = false
if in_parallel
    using SlurmNodes
	addprocs(get_list_of_nodes())
	#addprocs()
	println("Running on $(nprocs()) processes.")
else
	println("Running in serial.")
end
@everywhere using JLD2,FileIO,Random, Munkres, Distributions,StatsBase,NLsolve,LightGraphs,SimpleGraphs,Dierckx,PyPlot,Plots,PyCall,Dates
@everywhere import_path_desai = "/n/home07/juliankh/physics/research/desai/epidemics/src"
# @everywhere import_path_desai = "../.././"
@everywhere push!(LOAD_PATH, import_path_desai)
@everywhere import_path_nowak = "/n/home07/juliankh/physics/research/nowak/indirect_rec/src"
# @everywhere import_path_nowak = "/Users/julian/Harvard/research/nowak/indirect_rec/src/"
@everywhere push!(LOAD_PATH, import_path_nowak)
@everywhere using Epidemics,IM, GraphGeneration,DegreeDistribution,GraphGeneration,GraphCreation,GraphClustering,TwoLevelGraphs,DataAnalysis,GraphPlotting,Plotting

use_helvetica=true
helvetica_preamble = [
raw"\renewcommand{\familydefault}{\sfdefault}",raw"\usepackage{helvet}",raw"\everymath={\sf}",
raw"\usepackage{sansmath}",   # math-font matching  helvetica
raw"\sansmath"                # actually tell tex to use it!
# r'\usepackage[helvet]{sfmath}',
#     r'\usepackage{tgheros}',    # helvetica font
#     r'\usepackage{siunitx}',    # micro symbols
#     r'\sisetup{detect-all}',    # force siunitx to use the fonts
]
rcParams = PyDict(matplotlib["rcParams"])
if use_helvetica
    rcParams["text.latex.preamble"] = helvetica_preamble
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica"]
    # from matplotlib import rcParams
else
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Times New Roman"]
end
plt.rc("text",usetex=true)
rcParams["axes.linewidth"] = 2.0
rcParams["axes.spines.right"] = false
rcParams["axes.spines.top"] = false
rcParams["lines.linewidth"] = 1.5
rcParams["font.size"] = 20

###Functions###
save_path = "../data/fig3b.jld2"
generate_data = true
plot_beta = false

if generate_data
	sigma_k = 30
    beta = 0.1
    alpha = 1.0
    k = 10
    N = 10000
    yn_range = [-0.06,0.0,0.04]
    alpha_range = beta./((yn_range.+1.0/k))
    l1 = length(alpha_range)
    # tr_arr = Array{Any}(undef,l1)
    sr_arr = Array{Any}(undef,l1)
    tr_reg_arr = Array{Any}(undef, l1)
    num_trials = 100
    num_trials_th = 100
    num_trials_sim = 100
    num_trials_range = [10000,100_000,1000_000]
    in_parallel = true
    pregenerate_graph = true
    for (i,alpha) in enumerate(alpha_range)
        num_trials_sim = num_trials_range[i]
        num_trials_th = num_trials_range[i]
        graph_type = sigma_k == 0 ? regular_rg : gamma_rg
        gi = get_graph_information(gamma_rg,N=N,k=k,sigma_k=sigma_k,pregenerate_graph=pregenerate_graph)
    #     @time tr = get_theory_result(N,alpha,beta,gi,num_trials_th)
        @time    sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel)
        clean_result(sr)
    #     tr_arr[i] = tr
        sr_arr[i] = sr
        println(sparsity_cascade_condition(N,alpha,beta,k))
        gi = get_graph_information(regular_rg,N=N,k=k,sigma_k=sigma_k)
        tr = get_theory_result(N,alpha,beta,gi,num_trials_th)
        clean_result(tr)
        tr_reg_arr[i] = tr
    end
    JLD2.@save save_path sr_arr tr_reg_arr num_trials_range alpha_range yn_range beta N
else
    JLD2.@load save_path sr_arr tr_reg_arr num_trials_range alpha_range yn_range beta N
end

rmprocs(workers())


###Draw Figure###
arrow_fac = 3
labelsize=20
cm_min = 0.0
cm_max = 0.9
colors = color_range(length(alpha_range),"plasma",cm_min,cm_max)
# color_range = 0:length(k_range)
# N_min = minimum(N_range);N_max = maximum(N_range) 
for (i,alpha) in enumerate(alpha_range)
#     tr = tr_arr[i]
    sr = sr_arr[i]
#     intp = interpolate((tr.pr.yy,),tr.pr.pp,Gridded(Linear()))
    yn = yn_range[i]# sparsity_get_yn(alpha,beta,k)#_pfix_prediction_large_N
    println(yn)
#     plot_theory_result(tr,color=colors[i],label=latexstring("\$y_n = $yn\$"),linestyle="-",num_points = 10) 
    plot_simulation_result(sr,color=colors[i],label=latexstring("\$y_n = $yn\$"),
    linestyle="-",num_points = 10,error_region=true,error_num_points=40,alpha=0.3) 
    plot_theory_result(tr_reg_arr[i],color=colors[i],linestyle="--",num_points = 10,linewidth=1.0) 

end
# plot([],[],"--k",label=latexstring("\\sigma_k = 0"))
# plot([],[],"-k",label=latexstring("\\sigma_k = $(sigma_k)"))
legend(frameon=false,fontsize=labelsize-3)
gca().tick_params(labelsize=labelsize)
# xlim([N_min,N_max])
xlabel(L"Overall frequency, $y$",size=labelsize)
ylabel(L"P_{reach}(y)",size=labelsize)
# savefig("fig_3_degree_dist_p_reach.pdf",bbox_inches="tight",transparent=true)
ylim([1e-6,1e-0])
PyPlot.savefig("fig3h.pdf",bbox_inches="tight",transparent=true)
