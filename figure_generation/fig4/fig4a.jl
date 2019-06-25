###Preamble###
using Distributed
import_path_desai = "/n/home07/juliankh/physics/research/desai/epidemics/src"
push!(LOAD_PATH, import_path_desai)
in_parallel = true
if in_parallel
    using SlurmNodes
	addprocs(get_list_of_nodes())
	#addprocs()
	println("Running on $(nprocs()) processes.")
else
	println("Running in serial.")
end
@everywhere using JLD2,FileIO,Random, Munkres, Distributions,StatsBase,NLsolve,LightGraphs,SimpleGraphs,Dierckx,PyPlot,Plots,PyCall,Dates,Interpolations
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
save_path = "../data/fig4a.jld2"
generate_data = true
plot_beta = false

if generate_data
	plot_beta = false
    N = 50000;k=20;sigma_k = 0
    N_small = 2000
    # N_range = [120,200,320,440]#,800,1600,3200]
    # alpha_range = linspace(0.0,2.0,20)#8*[0.005,0.01,0.02,0.04,0.06,0.1,0.15,0.25]#,0.3]#collect(0.01:0.1:0.5)
    alpha_range = 10.0.^range(log10(0.01),stop=log10(1.0),length=30)#8*[0.005,0.01,0.02,0.04,0.06,0.1,0.15,0.25]#,0.3]#collect(0.01:0.1:0.5)
    k_range = [4,6,8,10,15,20,30]
    # alpha_range = logspace(log10(0.01),log10(2.0),10)#8*[0.005,0.01,0.02,0.04,0.06,0.1,0.15,0.25]#,0.3]#collect(0.01:0.1:0.5)
    if plot_beta
    #     sigma_k = 30
        iter_arr = collect(-0.2:0.01:0.2)#beta
    else
        beta = 0.025
        iter_arr = [4,6,8,10,14,18,22,26,30,34,38,40]#k
    end
    l2 = length(iter_arr)
    l1 = length(alpha_range)
    # l2 = length(N_range)
    phase_trans_arr = zeros(Float64,l1,l2)
    phase_trans_simple_arr = zeros(Float64,l1,l2)
    pfix_arr = zeros(Float64,l1,l2)
    pfix_arr_small = zeros(Float64,l1,l2)
    comm_ratio_arr = zeros(Float64,l1,l2)
    # beta = 0.05
    num_trials_sim = 4000
    num_trials_th = 100
    pregenerate_graph = true
    in_parallel = false

    for (i,alpha) in enumerate(alpha_range)
        for (j,iter_el) in enumerate(iter_arr)
            graph_type = sigma_k == 0 ? regular_rg : gamma_rg
            if plot_beta
                beta = iter_el
            else
                k = iter_el
            end
            gi = get_graph_information(graph_type,N=N,k=k,sigma_k=sigma_k,pregenerate_graph=pregenerate_graph)
            @time sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel) 
            pfix_arr[i,j] = get_pfix_sim(sr)[1]
    #         _,_,pfix,_ = Epidemics.get_simulation_yy_pp(gi,N,alpha,beta,num_trials_th,num_trials_sim=num_trials,use_theory=true)
    #          = pfix
            phase_trans_simple_arr[i,j] = sparsity_cascade_condition_simple(alpha,beta,k)
            phase_trans_arr[i,j] = sparsity_cascade_condition(N,alpha,beta,k)
    #         comm_ratio_arr[i,j] = get_community_graph_fixation_ratio(t,alpha,beta)
    #         N_small = Int(round(N/100))
            gi = get_graph_information(graph_type,N=N_small,k=k,sigma_k=sigma_k,pregenerate_graph=pregenerate_graph)
            @time sr = get_simulation_result(N_small,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel) 
    #         _,_,pfix,_ = Epidemics.get_simulation_yy_pp(gi,N_small,alpha,beta,num_trials_th,num_trials_sim=num_trials,use_theory=true)
            pfix_arr_small[i,j] = get_pfix_sim(sr)[1]
    #         println(comm_ratio_arr[i,j])
    #         sim_ratio, succ,pfix,pfixth = get_theory_sim_fixation_ratio(t,alpha,beta,num_trials)
    #         sim_ratio_arr[i,j] = sim_ratio
    #         succ_arr[i,j] = succ
    #         pfixth_arr[i,j] = pfixth
        end
    end
    # tr_complete.graph_information.graph=nothing
    # tr_complete.graph_information.graph_fn=nothing
	@save save_path pfix_arr pfix_arr_small phase_trans_arr phase_trans_simple_arr num_trials_sim num_trials_th alpha beta N N_small iter_arr
else
    @load save_path pfix_arr pfix_arr_small phase_trans_arr phase_trans_simple_arr num_trials_sim num_trials_th alpha beta N N_small iter_arr
end

rmprocs(workers())


###Draw Figure###
figure(dpi=200)
# axhline(xz,color="k",linestyle ="-")
# contourf(beta_range,alpha_range,pfix_arr,200,vmin=eps,vmax = 1.,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
vmin = 0.0
vmax = 1.01
tmp = clamp.(pfix_arr ./ (pfix_arr_small .+ 1e-10),vmin,vmax)

# contourf(l_range,N_range,pfix_arr,200,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
alpha_range_loc = range(minimum(alpha_range),stop=maximum(alpha_range),length=100)
if plot_beta
    xlabel(L"\beta",size=20)
#     yn_transition = beta_range'./(alpha_range_loc.*ones(x_arr)') - 1/k
else
    xlabel(L"k",size=20)
#     yn_transition = beta./(alpha_range_loc.*ones(x_arr)') - 1/k
end
contourf(iter_arr,alpha_range,tmp,500,vmin=vmin,vmax=vmax,cmap="gnuplot")#vmin=vmin,vmax = vmax,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
cb = colorbar(ticks=0:0.1:1)
contour(iter_arr,alpha_range,phase_trans_arr,[1.0],linestyles="--",colors="w") #negative selection regime starts
contour(iter_arr,alpha_range,phase_trans_simple_arr,[0.0],linestyles="-",colors="w") #above this we have at worst 1/sqrt(N) scaling
# contour(x_arr,alpha_range_loc,yn_transition,[-0.02,0.0,0.02],linestyles="--",colors="w")
contour(iter_arr,alpha_range,tmp,[sqrt(N_small/N),1.0],linestyles=":",linewidths=5,colors=["w","k"])#vmin=vmin,vmax = vmax,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
gca()[:tick_params](labelsize=15)
# gca()[:set_xticks]([1,5,10,15,19])
cb[:ax][:tick_params](labelsize=15)
# gca()[:set_yscale]("log")
ylabel(L"\alpha",size=20)
PyPlot.savefig("fig_4_sparsity_phase_transition.pdf",bbox_inches="tight",transparent=true)
