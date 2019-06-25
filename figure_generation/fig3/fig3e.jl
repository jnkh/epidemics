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
# @everywhere import_path_desai = "/n/home07/juliankh/physics/research/desai/epidemics/src"
@everywhere import_path_desai = "../.././"
@everywhere push!(LOAD_PATH, import_path_desai)
# @everywhere import_path_nowak = "/n/home07/juliankh/physics/research/nowak/indirect_rec/src"
@everywhere import_path_nowak = "/Users/julian/Harvard/research/nowak/indirect_rec/src/"
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
	plot_beta = false
    sigma_k = 0
    k = 20
    m = 20
    beta = 0.1
    alpha = 0.88
    N = 10000
    l_range = [19,15,13,1]

    l1 = length(l_range)
    yy_arr = Array{Any}(undef,l1)
    tr_arr = Array{Any}(undef,l1)
    sr_arr = Array{Any}(undef,l1)
    pp_arr = Array{Any}(undef,l1)
    num_trials = 10
    num_trials_th = 100
    trials_sim_range = [10_000,100_000,1000_000,1000_000]
    pregenerate_graph = true
    for (i,l) in enumerate(l_range)
        num_trials_sim = trials_sim_range[i]
        r = k -l
        t = get_t(N,k,m,l)
    #     g_ = TwoLevelGraphs.generate_regular_two_level_graph(t)
    #     gi = get_graph_information(custom_rg,custom_g=g_)#,N=N,k=k,m=m,l=l,r=r,carry_temporal_info=false,pregenerate_graph=pregenerate_graph)
        gi = get_graph_information(two_level_rg,N=N,k=k,m=m,l=l,r=r,carry_temporal_info=false,pregenerate_graph=pregenerate_graph)
    #     tr = get_theory_result(N,alpha,beta,gi,num_trials_th)
        @time sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel)
        clean_result(sr)

        sr_arr[i] = sr
        
        println(phase_transition_condition(t,alpha,beta))
    end
    gi = get_graph_information(regular_rg,N=N,k=k)
    tr_reg = get_theory_result(N,alpha,beta,gi,num_trials_th)
    clean_result(tr_reg)
    JLD2.@save save_path sr_arr tr_reg N k m l_range alpha beta trials_sim_range
else
    JLD2.@load save_path sr_arr tr_reg N k m l_range alpha beta trials_sim_range
end

rmprocs(workers())


###Draw Figure###
JLD2.@load save_path sr_arr tr_reg N k m l_range alpha beta trials_sim_range 
arrow_fac = 3
labelsize=20
cm_min = 0.0
cm_max = 0.9
colors = color_range(length(l_range),"plasma",cm_min,cm_max)
for (i,l) in enumerate(l_range)
#     color_loc = cm(cm_max*(i-1)/length(l_range))
#     tr = tr_arr[i]
    sr = sr_arr[i]
#     intp = interpolate((tr.pr.yy,),tr.pr.pp,Gridded(Linear()))
    yn = sparsity_get_yn(alpha,beta,k)#_pfix_prediction_large_N
#     plot_theory_result(tr,color=color_loc,label=latexstring("\$k_i = $l\$"),linestyle="-",num_points = 10) 
    plot_simulation_result(sr,color=colors[i],label=latexstring("\$k_i = $l\$"),linestyle="-",num_points = 10) 
end
plot_theory_result(tr_reg,color="k",label="Regular",linestyle="--",num_points = 10,linewidth=0.8) 
legend(frameon=false,fontsize=labelsize-3)
gca().tick_params(labelsize=labelsize)
# xlim([N_min,N_max])
xlabel(L"Overall frequency, $y$",size=labelsize)
ylabel(L"P_{reach}(y)",size=labelsize)
ylim([1e-6,1e-0])
PyPlot.savefig("fig3e.pdf",bbox_inches="tight",transparent=true)
