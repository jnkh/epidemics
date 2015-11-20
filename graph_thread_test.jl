using SIS,Base.Threads

alpha = 0.1
beta= 0.1
N = 100

const f1 = x -> 1 + alpha::Float64*x
const f2 = x -> 1 + beta::Float64
im = IM.InfectionModel(f1,f2)

p = PayloadGraph.Graph(LightGraphs.Graph(N,Int(round(N*(N-1)/2))),zeros(Int,N))

p.payload[1] = 1
# p.payload[2] = 1
# p.payload[5] = 1
# p.payload[15] = 1
# p.payload[10] = 1
new_types = zeros(Int,N)
num_trials = 6000

println("running on $(nthreads()) threads.")
@time for i = 1:num_trials
	num = 0
	new_types = zeros(Int,N)
	p.payload *= 0
	p.payload[1] = 1
	while sum(p.payload) > 0 && sum(p.payload) < N && num < 50
		SIS.update_graph_threads_test(p,im,new_types)
		println(sum(p.payload)/N)
		num +=1 
	end
	println("$(i/num_trials)")

end
# println(p)
# println(new_types)


