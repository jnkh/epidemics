module Plotting

using SIS,IM,PayloadGraph,PyPlot, Epidemics,JLD, TwoLevelGraphs,Dierckx
import LightGraphs

export plot_schematic,plot_schematics,plot_w,plot_two_level_schematic

function get_c_r(N,alpha,beta)
    return 4*alpha/(beta^2*N)
end

function get_n_n(N,alpha,beta)
    return beta/alpha*N
end

function get_alpha_beta(N,c_r,n_n)
    beta = 4.0/(c_r*n_n)
    alpha = (N*beta)/n_n
    return alpha,beta
end


f(y,alpha) = alpha.*y.^2
s(y,alpha,beta) = f(y,alpha)./y - beta
#get_y_eff(y,k) = y.*(1 + (1-y)./(y.*k))
#get_s_eff(y::Array,alpha,beta,k) = alpha*get_y_eff(y,k) - beta


function plot_schematics(N,n_n,alpha,beta,im,imk,k)
    plot_reach = true
    #pygui(true)
#     close("all")
    dx = 2*n_n/N/200
    x = collect(1/N:dx:2*n_n/N)
    if plot_reach
        y = IM.P_reach(im,N,1.0/N,x)
        yk = IM.P_reach(imk,N,1.0/N,x)
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
    xlim([1/N,2*n_n/N])
    y_n, y_minus, y_plus, y_p,critical_determinant = get_parameters(N,alpha,beta)
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
    plot_schematic(n_n,c_r,N,k,true)
    title(latexstring("\$y_n = $(n_n/N), c_r = $c_r, N = $N\$"))
    
    x = collect(1/N:0.01:1)
    figure(3)
    plot(x,get_s_eff(x,alpha,beta,k),"-b",label=L"$s_{eff}(y)$")
    plot(x,get_s_eff(x,alpha,beta,N-1),"-r",label=L"$s(y)$")
    grid(1)

end


function plot_schematic(n_n,c_r,N,k=N-1,plot_k=false)
    beta = 4.0/(c_r*n_n)
    alpha = (N*beta)/n_n
#     println(N,alpha,beta)

    y_n, y_minus,y_plus,y_p,critical_determinant = get_parameters(N,alpha,beta)
    f(y) = alpha.*y.^2
    s(y) = f(y)./y - beta
    get_y_eff(y,k) = y.*(1 + (1-y)./(y.*k))
    get_s_eff(y,alpha,beta,k) = alpha*get_y_eff(y,k) - beta


    y_range = collect(0:y_p/1000:1.9*y_p)
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
    ylim([0,1.9*y_p])
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

function plot_two_level_schematic(t,alpha,beta,N)
    y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp = get_interpolations(t,alpha,beta)
    println("computed interpolations")

    s(x) = get_s_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)
    splus(x) = get_splus_effective_two_level_interp(x,alpha,beta,y_inf_interp,y_sq_inf_interp,y_susc_interp,y_sq_susc_interp)

    xx = logspace(log10(1/N),0,100) 
    pp = P_reach_fast(s,splus,N,1/N,xx)
#     loglog(xx,pp)
#     xlim([1/N,1])
    return xx,pp,s
end

end