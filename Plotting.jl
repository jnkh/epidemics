module Plotting

using SIS,IM,PayloadGraph,PyPlot, Epidemics,JLD, TwoLevelGraphs,Dierckx
import LightGraphs

export plot_schematic,plot_schematics,

plot_w,plot_two_level_schematic


function plot_schematics(N,n_n,c_r,alpha,beta,im,imk,k,exact=false)
    plot_reach = true
    #pygui(true)
#     close("all")
    dx = 1/(2*N)
    x = collect(1/N:dx:1)
    if plot_reach
        y = P_reach_fast(im,N,1.0/N,x)
        yk = P_reach_fast(imk,N,1.0/N,x)
        plotfn = loglog
        plotstr = "reach"
    else
        y = IM.P_fix(im,N,x)
        yk = IM.P_fix(imk,N,x)
        plotfn = plot
        plotstr = "fix"
    end
    figure(2)#,figsize=(8,5))
    plotfn(x,y,"-r",label=latexstring("P_{$(plotstr)}(y)"))
    plotfn(x,yk,"-b",label=latexstring("P_{$(plotstr)}(y_{eff})"))
    plotfn(x,1/N./x,"--k",label=latexstring("P_{$(plotstr),neutral}"))
    xlim([1/N,1.0])
    y_n, y_minus, y_plus, y_p,critical_determinant = get_parameters(N,alpha,beta,exact=exact)
    axvline(y_n,linestyle="--",color="b",label=L"y_n")
    axvline(y_minus,linestyle="-.",color="r",label=L"y_1")
    axvline(y_plus,linestyle="-.",color="r",label=L"y_2")
    axvline(y_p,linestyle="-",color="b",label=L"y_p")
    xlabel(L"y")
    ylabel(latexstring("P_{$(plotstr)}(y)"))
    legend(loc="best")
    title(latexstring("\$y_n = $(n_n/N), c_r = $c_r, N = $N\$"))
    #savefig("p_fix_y_n = $(n_n/N), c_r = $c_r, N = $N.png")

    figure(1)#,figsize=(8,5))
    plot_schematic(n_n,c_r,N,k,true,exact)
    title(latexstring("\$y_n = $(n_n/N), c_r = $c_r, N = $N\$"))
    
    x = collect(1/N:0.01:1)
    figure(3)
    plot(x,get_s_eff_exact(x,alpha,beta,k,N),"-b",label=L"$s_{eff}(y)$")
    plot(x,get_s_eff_exact(x,alpha,beta,N-1,N),"-r",label=L"$s(y)$")
    grid(1)

end

function plot_schematic(n_n,c_r,N,k=N-1,plot_k=false,exact=false)
    beta = 4.0/(c_r*n_n)
    alpha = (N*beta)/n_n
#     println(N,alpha,beta)

    y_n, y_minus,y_plus,y_p,critical_determinant = get_parameters(N,alpha,beta,exact=exact)
    f(y) = alpha.*y.^2
    s(y) = f(y)./y - beta
    get_y_eff(y,k) = y.*(1 + (1-y)./(y.*k))
    get_s_eff(y,alpha,beta,k) = alpha*get_y_eff(y,k) - beta


    y_range = collect(0:1/(2*N):1)
    plot(y_range,1.0./abs(N*s(y_range)),"-r",label=L"$\frac{1}{N|s(y)|}$")
    if plot_k
        plot(y_range,1.0./abs(N*get_s_eff(y_range,alpha,beta,k)),"-b",label=L"$\frac{1}{N|s(y_{eff})|}$")
    end
    plot(y_range,y_range,"-k",label=L"$y$")
    axvline(y_n,linestyle="--",label=L"$y_n$")
    axvline(y_p,linestyle="-",label=L"$y_p$")
    if y_minus > 0
        axvline(y_minus,linestyle="-.",label=L"$y_1$")
        axvline(y_plus,linestyle="-.",label=L"$y_2$")
    end
    ylim([0,1])
    legend(prop=Dict{Any,Any}("size"=>15),loc="upper right")
    xlabel(L"$y$",size=20)
    if plot_k
        #figure(1)
        #plot(y_range,get_s_eff(y_range,alpha,beta,k),"-b",label=L"$s_{eff}(y)$")
        #plot(y_range,s(y_range),"-r",label=L"$s(y)$")
        legend(prop=Dict{Any,Any}("size"=>20),loc="upper right")
        xlabel(L"$y$",size=20)
    end
end

function plot_w(sizes,N,alpha,beta,k::Int,word = "two level")

    figure(4)
    sizes = (1 + beta).*sizes
    bins = logspace(log10(minimum(sizes)),log10(maximum(sizes)),150)
#     word = graph_model ? "graph" : "well-mixed"
    label = latexstring("$word, \$k = $k\$")
    PyPlot.plt[:hist](sizes,log=true,bins=bins,alpha=0.2,normed=true,label=label)

    gca()[:set_xscale]("log")

    w_range = bins[1:end]#logspace(log10(4*minimum(sizes)),log10(maximum(sizes)),30)

    P_w_th_range = normed_distribution(w_range,P_w_th(w_range,s(sqrt(w_range)./N,alpha,beta)))
    P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,get_s_eff(sqrt(w_range)./N,alpha,beta,k)))
    #P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,
    #get_s_effective_two_level_interp(sqrt(w_range)./N,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)))
    
    correction_fac = 5
    #plot(w_range,correction_fac*P_w_th_range,"-r",label=L"theory $k \to N-1$")#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    if true#graph_model
        plot(w_range,correction_fac*P_w_th_range_eff,"-b",label=latexstring("effective theory \$k = $k\$"))#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    end
        
    xlabel(L"$w$",size=20)
    ylabel(L"$P(w)$",size=20)

    legend(loc="lower left")
    ylim([1e-6,1e3])
    grid()

end
function plot_w(sizes,N,alpha,beta,s_eff_fn::Function, word = "two level")

    figure(4)
    sizes = (1 + beta).*sizes
    bins = logspace(log10(minimum(sizes)),log10(maximum(sizes)),150)
#     word = graph_model ? "graph" : "well-mixed"
    label = latexstring("$word, \$k = $k\$")
    PyPlot.plt[:hist](sizes,log=true,bins=bins,alpha=0.2,normed=true,label=label)

    gca()[:set_xscale]("log")

    w_range = bins[1:end]#logspace(log10(4*minimum(sizes)),log10(maximum(sizes)),30)

    P_w_th_range = normed_distribution(w_range,P_w_th(w_range,s(sqrt(w_range)./N,alpha,beta)))
    #P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,get_s_eff(sqrt(w_range)./N,alpha,beta,k)))
    P_w_th_range_eff = normed_distribution(w_range,P_w_th(w_range,
    s_eff_fn(sqrt(w_range))./N))
    
    correction_fac = 5
    #plot(w_range,correction_fac*P_w_th_range,"-r",label=L"theory $k \to N-1$")#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    if true#graph_model
        plot(w_range,correction_fac*P_w_th_range_eff,"-g",label="two-level eff. theory")#$P(w) \sim e^{- s(\sqrt{w})^2 w/4} w^{-3/2}/(1 + s(\sqrt{w}))$ (theory)')
    end
        
    xlabel(L"$w$",size=20)
    ylabel(L"$P(w)$",size=20)

    legend(loc="lower left")
    ylim([1e-6,1e3])
    grid()

end

function plot_two_level_schematic(t,alpha,beta,N,apply_finite_size=true)
    y_inf_fn,y_sq_inf_fn,y_susc_fn,y_sq_susc_fn = get_interpolations(t,alpha,beta,apply_finite_size)
    println("computed interpolations")
    # dt = get_dt_two_level(alpha,beta)
    # runs_well_mixed_tl = run_epidemics(100000, () -> run_epidemic_well_mixed_two_level(dt,N,y_susc_fn,y_sq_susc_fn,y_inf_fn,y_sq_inf_fn,alpha,beta,1.0));
	# yvals_well_mixed_tl,pvals_well_mixed_tl = get_p_reach(runs_well_mixed_tl,N)

    s(x) = get_s_effective_two_level_interp(x,alpha,beta,y_inf_fn,y_sq_inf_fn,y_susc_fn,y_sq_susc_fn)
    splus(x) = get_splus_effective_two_level_interp(x,alpha,beta,y_inf_fn,y_sq_inf_fn,y_susc_fn,y_sq_susc_fn)

    xx = logspace(log10(1/N),0,100) 
    pp = P_reach_fast(s,splus,N,1/N,xx)
#     loglog(xx,pp)
#     xlim([1/N,1])
    return xx,pp,s,splus
end

end