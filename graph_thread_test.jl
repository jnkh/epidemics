using SIS

alpha = 0.1
beta= 0.1
N = 20
im = IM.InfectionModel(x -> 1 + alpha*x, x -> 1 + beta)

p = PayloadGraph.Graph(LightGraphs.Graph(N,Int(round(N*(N-1)/2))),zeros(Int,N))


