using SIS

alpha = 0.1
beta= 0.1
N = 2000
im = IM.InfectionModel(x -> 1 + alpha*x, x -> 1 + beta)

p = PayloadGraph.Graph(LightGraphs.Graph(N,Int(round(N*(N-1)/2))),zeros(Int,N))

p.payload[1] = 1
# p.payload[2] = 1
# p.payload[5] = 1
# p.payload[15] = 1
# p.payload[10] = 1
new_types = zeros(Int,N)

num = 0
@time while sum(p.payload) > 0 && sum(p.payload) < N && num < 50
	SIS.update_graph_threads_test(p,im,new_types)
	println(sum(p.payload)/N)
	num +=1 
end

# println(p)
# println(new_types)


