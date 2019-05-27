##################################################
######## A simple model of infection with variable timestep
##################################################


module IM

using Dierckx
#import Cubature
import QuadGK,PyPlot,Interpolations
export p_birth,p_death,InfectionModel, plot_schematic, get_parameters,
P_reach,P_fix,P_reach_fast,P_reach_raw_fast,

get_s_interp, get_s_integral_interp


struct InfectionModel
    birth_rate::Function
    death_rate::Function
    dt::Float64
end

eps = 1e-10
maxevals = 200

function InfectionModel(birth_rate::Function,death_rate::Function)
    desired_rate = 0.01
    max_rate = maximum([birth_rate(1.0),death_rate(1.0),birth_rate(0.0),death_rate(0.0)])
    dt = desired_rate/max_rate
    InfectionModel(birth_rate,death_rate,dt)
end



function p_birth(im::InfectionModel,x)
    return im.birth_rate(x)*im.dt
end

function p_death(im::InfectionModel,x)
    return im.death_rate(x)*im.dt
end

function s(im::InfectionModel,x)
    return (1- x)*(p_birth(im,x) - p_death(im,x))
end
    
function P_fix(im::InfectionModel,N::Int,x0::Real)
    s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
    a(x) = x*s(x)
    b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
    psi(x,a,b) = exp( -2* QuadGK.quadgk(y -> a(y)/b(y),0,x)[1])
    
    return QuadGK.quadgk(y -> psi(y,a,b),0,x0)[1]/QuadGK.quadgk(y -> psi(y,a,b),0,1)[1]
end

function P_fix(im::InfectionModel,N::Int,x0::Array)
    return [P_fix(im,N,xind) for xind in x0]
end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
#     s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
#     a(x) = x*s(x)
#     b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
#     psi(x,a,b) = exp( -2* QuadGK.quadgk(y -> a(y)/b(y),0,x)[1])
    
#     return QuadGK.quadgk(y -> psi(y,a,b),0,x0)[1]/QuadGK.quadgk(y -> psi(y,a,b),0,x1)[1]
# end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
#     return [P_reach(im,N,x0,xind) for xind in x1]
# end


function P_fix(s::Function,splus::Function,N::Int,x0::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi(x,a,b) = exp( -2* QuadGK.quadgk(y -> a(y)/b(y),eps,x)[1])
    
    return QuadGK.quadgk(y -> psi(y,a,b),eps,x0)[1]/QuadGK.quadgk(y -> psi(y,a,b),eps,1)[1]
end


function P_fix(s::Function,splus::Function,N::Int,x0::Array)
    return [P_fix(s,splus,N,xind) for xind in x0]
end


function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi(x,a,b) = exp( -2* QuadGK.quadgk(y -> a(y)/b(y),eps,x)[1])
    
    return QuadGK.quadgk(y -> psi(y,a,b),eps,x0)[1]/QuadGK.quadgk(y -> psi(y,a,b),eps,x1)[1]
end

function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
    return [P_reach(s,splus,N,x0,xind) for xind in x1]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
    s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
    a(x) = x*s(x)
    b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
    psi(x,a,b) = exp( -2* QuadGK.quadgk(y -> a(y)/b(y),0,x)[1])
    
    return QuadGK.quadgk(y -> psi(y,a,b),eps,x0)[1]/QuadGK.quadgk(y -> psi(y,a,b),eps,x1)[1]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
    return [P_reach(im,N,x0,xind) for xind in x1]
end




function P_reach_fast(im::InfectionModel,N::Int,x0::Real,x1::Array,slow_im=false;num_points=100)
    a(x) = (p_birth(im,x) - p_death(im,x)) #*x(1-x)
    b(x) = 1/N*(p_birth(im,x) + p_death(im,x)) #*x(1-x) 
    if slow_im
        a = get_interp_function(a,eps,1)
        b = get_interp_function(b,eps,1)
    end

    return P_reach_raw_fast(a,b,N,x0,x1,num_points)
end

function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi_interp = get_psi_interp(a,b,eps,N)
    
    return QuadGK.quadgk(y -> psi_interp(y),eps,x0,maxevals=maxevals)[1]/QuadGK.quadgk(y -> psi_interp(y),eps,x1,maxevals=maxevals)[1]
end

function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    return P_reach_raw_fast(a,b,N,x0,x1)
end

function P_reach_raw_fast(a::Function,b::Function,N::Int,x0::Real,x1::Array,num_points=100)
    a_over_b(y) = a(y)./b(y)
    return P_reach_raw_fast(a_over_b,N,x0,x1,num_points)
end

function P_reach_raw_fast(a_over_b::Function,N::Int,x0::Real,x1::Array,num_points=100)
    psi_interp = get_psi_interp(a_over_b,eps,N,num_points)

    try
        integration_fn(x_0,x) = QuadGK.quadgk(y -> psi_interp(y),x_0,x,maxevals=maxevals)[1] 
        denominator = 0*similar(psi_interp(x1)) #ensure the same type as psi_interp, which could be BigFloat
        denominator[1] = integration_fn(eps,x1[1])
        for i in 2:length(x1)
            denominator[i] = denominator[i-1] + integration_fn(x1[i-1],x1[i])
        end
        
        numerator = integration_fn(eps,x0) 
        ret = Float64.(numerator ./ denominator) #ensure return is float
        return ret
    catch DomainError
        # PyPlot.figure()
        # PyPlot.plot(xx,a_over_b(xx),"--k")
        # PyPlot.show()
        # PyPlot.figure()
        # PyPlot.plot(xx,psi_interp(xx),"--b")
        # PyPlot.show()
        xx = 0:0.01:1.0
        println(psi_interp(xx))
        throw(DomainError)
    end
end
    
# function get_psi_interp(a::Function,b::Function,eps::Real,N::Int)
#     psi(x,a,b) = exp( -2* QuadGK.quadgk(y -> a(y)/b(y),eps,x)[1])
#     xx = logspace(log10(eps),0,100)
#     yy = zeros(xx)
#     for i in 1:length(xx)
#         yy[i] = psi(xx[i],a,b)
#     end
#     psi_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
#     psi_interp(x) = evaluate(psi_spline,x)
#     return psi_interp
# end
function get_s_interp(im::InfectionModel,N::Int)
    # a(x) = x*(1-x)*(p_birth(im,x) - p_death(im,x))
    # b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))  
    # s(x) = 2*a(x)/(N*b(x))
    s(x) = 2*(p_birth(im,x) - p_death(im,x))/(p_birth(im,x) + p_death(im,x))
    return get_interp_function(s,eps,1,1000)
end

function get_s_integral_interp(im::InfectionModel,N::Int)
    s = get_s_interp(im,N)
    get_s_integral_interp(s)
end


function get_s_integral_interp(s::Function)
    S(x0,x1) = QuadGK.quadgk(s,x0,x1,maxevals=1000)[1]
    xx = linspace(eps,1,1000)
    yy = 0*similar(xx)
    yy[1] = S(eps,xx[1])
    for i in 2:length(xx)
        yy[i] = yy[i-1] + S(xx[i-1],xx[i])
    end
    S_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
    S_interp(x) = evaluate(S_spline,x)
    return S_interp 
end



function is_out_of_reasonable_range(x)
    return isnan(x) || abs(x) == Inf || abs(x) > 1e300
end

function get_psi_interp(a_over_b::Function,eps::Real,N::Int,num_points=100)
    psi(x0,x1) = QuadGK.quadgk(a_over_b,x0,x1,maxevals=maxevals)[1]
    xx = 10.0.^range(log10(eps),stop=0,length=num_points)
    yy = 0*similar(xx)
    yy[1] = psi(eps,xx[1])
    for i in 2:length(xx)
        yy[i] = yy[i-1] + psi(xx[i-1],xx[i])
    end
    psi_vec = exp.( - 2.0* yy)
    psi_spline = Spline1D(xx,psi_vec,k=1,bc="extrapolate")
    psi_interp(x) = evaluate(psi_spline,x)

    if any(is_out_of_reasonable_range.(psi_interp(xx))) #if the fortran routine fails, use julia and bigfloat
        psi_vec = exp.( -2.0*BigFloat.(yy))
        intp = Interpolations.interpolate((xx,),psi_vec,Interpolations.Gridded(Interpolations.Linear()))
        psi_interp_new(x) = intp[x]
        return psi_interp_new
    else
        return psi_interp
    end
end

function get_psi_interp(a::Function,b::Function,eps::Real,N::Int,num_points=100)
    a_over_b(y) = a(y)/b(y)
    return get_psi_interp(a_over_b,eps,N,num_points)
end

function get_interp_function(f::Function,x0,x1,num_points=100)
    xx = 10.0.^range(log10(x0),stop=log10(x1),length=num_points)
    yy = 0*similar(xx)
    for i in 1:length(xx)
        yy[i] = f(xx[i]) 
    end
    f_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
    f_interp(x) = evaluate(f_spline,x)
    return f_interp
end




end
