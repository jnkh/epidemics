##################################################
######## A simple model of infection with variable timestep
##################################################


module IM

#import Cubature
export p_birth,p_death,InfectionModel, plot_schematic, get_parameters,
P_reach,P_fix,P_reach_fast


type InfectionModel
    birth_rate::Function
    death_rate::Function
    dt
end

eps = 1e-5
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
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),0,x)[1])
    
    return quadgk(y -> psi(y,a,b),0,x0)[1]/quadgk(y -> psi(y,a,b),0,1)[1]
end

function P_fix(im::InfectionModel,N::Int,x0::Array)
    return [P_fix(im,N,xind) for xind in x0]
end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
#     s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
#     a(x) = x*s(x)
#     b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
#     psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),0,x)[1])
    
#     return quadgk(y -> psi(y,a,b),0,x0)[1]/quadgk(y -> psi(y,a,b),0,x1)[1]
# end

# function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
#     return [P_reach(im,N,x0,xind) for xind in x1]
# end


function P_fix(s::Function,splus::Function,N::Int,x0::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x)[1])
    
    return quadgk(y -> psi(y,a,b),eps,x0)[1]/quadgk(y -> psi(y,a,b),eps,1)[1]
end


function P_fix(s::Function,splus::Function,N::Int,x0::Array)
    return [P_fix(s,splus,N,xind) for xind in x0]
end


function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x)[1])
    
    return quadgk(y -> psi(y,a,b),eps,x0)[1]/quadgk(y -> psi(y,a,b),eps,x1)[1]
end

function P_reach(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
    return [P_reach(s,splus,N,x0,xind) for xind in x1]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Real)
    s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
    a(x) = x*s(x)
    b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))
    
    psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),0,x)[1])
    
    return quadgk(y -> psi(y,a,b),eps,x0)[1]/quadgk(y -> psi(y,a,b),eps,x1)[1]
end

function P_reach(im::InfectionModel,N::Int,x0::Real,x1::Array)
    return [P_reach(im,N,x0,xind) for xind in x1]
end




function P_reach_fast(im::InfectionModel,N::Int,x0::Real,x1::Array,slow_im=false)
    s(x) = (1-x)*(p_birth(im,x) - p_death(im,x))
    a(x) = x*s(x)
    b(x) = 1/N*(1-x)*x*(p_birth(im,x) + p_death(im,x))  
    if slow_im
        a = get_interp_function(a,eps,1)
        b = get_interp_function(b,eps,1)
    end
    psi_interp = get_psi_interp(a,b,eps,N)
    
    numerator = quadgk(y -> psi_interp(y),eps,x0,maxevals=maxevals)[1]
    denominator = [quadgk(y -> psi_interp(y),eps,x1_el,maxevals=maxevals)[1] for x1_el in x1] 
    return numerator ./ denominator
end

function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Real)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    
    psi_interp = get_psi_interp(a,b,eps,N)
    
    return quadgk(y -> psi_interp(y),eps,x0,maxevals=maxevals)[1]/quadgk(y -> psi_interp(y),eps,x1,maxevals=maxevals)[1]
end

function P_reach_fast(s::Function,splus::Function,N::Int,x0::Real,x1::Array)
    a(x) = s(x)
    b(x) = 1/N*(splus(x))
    return P_reach_raw_fast(a,b,N,x0,x1)
end

function P_reach_raw_fast(a::Function,b::Function,N::Int,x0::Real,x1::Array)
    psi_interp = get_psi_interp(a,b,eps,N)
    
    numerator = quadgk(y -> psi_interp(y),eps,x0,maxevals=maxevals)[1]
    denominator = [quadgk(y -> psi_interp(y),eps,x1_el,maxevals=maxevals)[1] for x1_el in x1] 
    return numerator ./ denominator
end
    
using Dierckx
# function get_psi_interp(a::Function,b::Function,eps::Real,N::Int)
#     psi(x,a,b) = exp( -2* quadgk(y -> a(y)/b(y),eps,x)[1])
#     xx = logspace(log10(eps),0,100)
#     yy = zeros(xx)
#     for i in 1:length(xx)
#         yy[i] = psi(xx[i],a,b)
#     end
#     psi_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
#     psi_interp(x) = evaluate(psi_spline,x)
#     return psi_interp
# end

function get_psi_interp(a::Function,b::Function,eps::Real,N::Int)
    psi(x0,x1) = quadgk(y -> a(y)/b(y),x0,x1,maxevals=maxevals)[1]
    xx = logspace(log10(eps),0,100)
    yy = zeros(xx)
    yy[1] = psi(eps,xx[1])
    for i in 2:length(xx)
        yy[i] = yy[i-1] + psi(xx[i-1],xx[i])
    end
    yy = exp( - 2.* yy)
    psi_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
    psi_interp(x) = evaluate(psi_spline,x)
    return psi_interp
end

function get_interp_function(f::Function,x0,x1)
    xx = logspace(log10(x0),log10(x1),100)
    yy = zeros(xx)
    for i in 1:length(xx)
        yy[i] = f(xx[i]) 
    end
    f_spline = Spline1D(xx,yy,k=1,bc="extrapolate")
    f_interp(x) = evaluate(f_spline,x)
    return f_interp
end




end
