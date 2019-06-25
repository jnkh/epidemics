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
@everywhere using JLD2,FileIO,Random, Munkres, Distributions,StatsBase,NLsolve,LightGraphs,SimpleGraphs,Dierckx,PyPlot,Plots,PyCall,Dates,Interpolations
@everywhere import_path_desai = "/n/home07/juliankh/physics/research/desai/epidemics/src"
#@everywhere import_path_desai = "../.././"
@everywhere push!(LOAD_PATH, import_path_desai)
@everywhere import_path_nowak = "/n/home07/juliankh/physics/research/nowak/indirect_rec/src"
#@everywhere import_path_nowak = "/Users/julian/Harvard/research/nowak/indirect_rec/src/"
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
save_path = "../data/fig4b.jld2"
generate_data = true
plot_beta = false

if generate_data
N = 50000;k=20;m=20;l=19;r = k-l #N = 100000
N_small = 2000 #4000
alpha_range = 10.0.^range(log10(0.01),stop=log10(1.5),length=30)#8*[0.005,0.01,0.02,0.04,0.06,0.1,0.15,0.25]#,0.3]#collect(0.01:0.1:0.5)

beta = 0.025
#     l_range = [1,2,4,6,8,10,12,14,16,17,18,19] 
l_range = [1,4,8,11,13,14,15,16,17,18,19] 
l2 = length(l_range)
l1 = length(alpha_range)
# l2 = length(N_range)
phase_trans_arr = zeros(Float64,l1,l2)
pfix_arr = zeros(Float64,l1,l2)
pfix_arr_small = zeros(Float64,l1,l2)
comm_ratio_arr = zeros(Float64,l1,l2)
# beta = 0.05
num_trials = 400
num_trials_sim = 400_000
num_trials_th = 100
graph_type = two_level_rg
pregenerate_graph = true
for (i,alpha) in enumerate(alpha_range)
    for (j,l) in enumerate(l_range)
#     for (j,l) in enumerate(l_range)
#     for (j,beta) in enumerate(beta_range)
        r = k-l
#     for (j,N) in enumerate(N_range)
        t = get_t(N,k,m,l)
        gi = get_graph_information(graph_type,N=N,k=k,m=m,l=l,r=r,pregenerate_graph=pregenerate_graph)
        @time sr = get_simulation_result(N,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel) 
        pfix_arr[i,j] = get_pfix_sim(sr)[1]
#         _,_,pfix,_ = Epidemics.get_simulation_yy_pp(gi,N,alpha,beta,num_trials_th,num_trials_sim=num_trials,use_theory=true)
#         pfix_arr[i,j] = pfix
        phase_trans_arr[i,j] = phase_transition_condition(t,alpha,beta)
        comm_ratio_arr[i,j] = get_community_graph_fixation_ratio(t,alpha,beta)
#         N_small = Int(round(N/100))
        t = get_t(N_small,k,m,l)
        gi = get_graph_information(graph_type,N=N_small,k=k,m=m,l=l,r=r,pregenerate_graph=pregenerate_graph)
        @time sr = get_simulation_result(N_small,alpha,beta,gi,num_trials_th,num_trials_sim,in_parallel=in_parallel) 
        pfix_arr_small[i,j] = get_pfix_sim(sr)[1]
    print(".")
    end
    println()
end
	@save save_path pfix_arr pfix_arr_small phase_trans_arr comm_ratio_arr num_trials_sim num_trials_th alpha beta N N_small k_range 
else
    @load save_path pfix_arr pfix_arr_small phase_trans_arr comm_ratio_arr num_trials_sim num_trials_th alpha beta N N_small k_range 
end

rmprocs(workers())


using Interpolations
alpha_range_loc = 10.0.^range(log10(minimum(alpha_range)),stop=log10(maximum(alpha_range)),length=100)
t = get_t(N,k,m,l)
comm_ratio_loc = [get_community_graph_fixation_ratio(t,alpha,beta) for alpha in alpha_range_loc]
# get_community_graph_fixation_ratio(t,alpha,beta)
itp = interpolate((comm_ratio_loc,),alpha_range_loc,Gridded(Linear()))
alpha_crit_comm_ratio = itp[1.0]
alpha_crit_k = k*beta


figure(dpi=200)
# axhline(xz,color="k",linestyle ="-")
# contourf(beta_range,alpha_range,pfix_arr,200,vmin=eps,vmax = 1.,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
vmin = 0.0
vmax = 1.01
tmp = clamp.(pfix_arr ./ (pfix_arr_small .+ 1e-10),vmin,vmax)

# contourf(l_range,N_range,pfix_arr,200,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
alpha_range_loc = range(minimum(alpha_range),stop=maximum(alpha_range),length=100)
if plot_beta
    x_arr = beta_range
    xlabel(L"\beta",size=20)
    yn_transition = beta_range'./(alpha_range_loc.*ones(x_arr)') - 1/k
else
    xlabel(L"k_i",size=20)
    x_arr = l_range
    yn_transition = beta./(alpha_range_loc.*ones(length(x_arr))') .- 1/k
end
contourf(x_arr,alpha_range,tmp,500,vmin=vmin,vmax=vmax,cmap="gnuplot")#vmin=vmin,vmax = vmax,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
cb = colorbar(ticks=0:0.1:1)
contour(x_arr,alpha_range,phase_trans_arr,[1.0],colors="w") #mild to no scaling transition
contour(x_arr,alpha_range,comm_ratio_arr,[1.0],linestyles="-",colors="k") #ratio of exactly 1 transition
# contour(x_arr,alpha_range_loc,yn_transition,[-0.02,0.0,0.02],linestyles="--",colors="w")
contour(x_arr,alpha_range,tmp,[sqrt(N_small/N),1.0],linestyles=":",linewidths=5,colors=["w","k"])#vmin=vmin,vmax = vmax,cmap="inferno")#,norm=matplotlib[:colors][:LogNorm](vmin=0.001,vmax=0.1,clip=true))#logspace(0,1,20)[1:end-1]))#) #cmap="inferno"
gca()[:tick_params](labelsize=15)
gca()[:set_xticks]([1,5,10,15,19])
cb[:ax][:tick_params](labelsize=15)
# gca()[:set_yscale]("log")
ylabel(L"\alpha",size=20)
annotate("", xy=(1, alpha_crit_k), xytext=(-0.5, alpha_crit_k),arrowprops=Dict("facecolor"=>"w", "width"=>2.0,"headwidth"=>7.0, "shrink"=>0.04))
annotate("", xy=(19, alpha_crit_comm_ratio), xytext=(17.5, alpha_crit_comm_ratio),arrowprops=Dict("facecolor"=>"k", "width"=>2.0,"headwidth"=>7.0, "shrink"=>0.04))
# annotate("", ,arrowprops=Dict("facecolor"=>"k", "shrink"=>0.05))
PyPlot.savefig("fig_4_community_phase_transition.pdf",bbox_inches="tight",transparent=true)
