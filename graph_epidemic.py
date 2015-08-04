import networkx as nx
from graph_reciprocity import *
susceptible = 0
infected = 1
recovered = 2

def draw_graph(G,pos=None):
    k = 2*len(G.edges())/len(G.nodes())
    
    node_color = [1.0*G.node[i]['strategy']/recovered for i in range(len(G.nodes()))]
    #node_label = {i:G.node[i]['strategy'] for i in range(len(G.nodes()))}
    node_label = {n:n for n in G.nodes()}
    node_size = [250.0*len(G.edges([n]))/k for n in range(len(G.nodes()))]
    nx.draw_networkx_nodes(G, pos, node_color=node_color,node_size=node_size,alpha=0.5, cmap=None, vmin=0.0, vmax=1.0,node_label='s',with_labels=True)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G,pos,node_label)
    #nx.draw(G,node_color=node_color,labels=node_label,cmap='spring',pos=pos)

def create_graph(N_nodes,p_edge,n_infected):
    G = nx.gnp_random_graph(N_nodes,p_edge,directed=False)
    return set_graph_strategies(G, n_infected)

def create_small_world_graph(N_nodes,p_edge,n_infected):
    n = N_nodes
    k = int(p_edge*N_nodes)
    p = 0.5
    #Random graph
    #p_coop is the fraction of cooperators
    G = nx.watts_strogatz_graph(n,k,p)
    return set_graph_strategies(G, n_infected)


def create_scale_free_graph(N_nodes,p_edge,n_infected):
    #scale free and small world
    #Growing Scale-Free Networks with Tunable Clustering
    n = N_nodes
    m = int(0.5*p_edge*N_nodes)
    p = 1.0
    #Random graph
    #p_coop is the fraction of cooperators
    G = nx.powerlaw_cluster_graph(n,m,p)
    return set_graph_strategies(G, n_infected)


def set_graph_strategies(G, n_infected,pos_infected=None):
    if pos_infected== None:
        nodes = G.nodes()
        np.random.shuffle(nodes)
        for i,n in enumerate(nodes):
            if i >= n_infected:
                G.node[n]['strategy'] = susceptible
            else:
                G.node[n]['strategy'] = infected
            G.node[n]['score'] = 0
        return G
    if len(pos_infected) == n_infected and len(set(pos_infected)) == n_infected:
        nodes = G.nodes()
        for i,n in enumerate(nodes):
            if i in pos_infected:
                G.node[n]['strategy'] = infected
            else:
                G.node[n]['strategy'] = susceptible
            G.node[n]['score'] = 0
        return G
    else:
        print 'invalid node positions'
        return None

def update_node(G,G_old,n,p_recover,p_infect):
    current_strategy = G_old.node[n]['strategy']
    if current_strategy == infected:
        #infect neighbors
        k = 2*len(G.edges())/len(G.nodes())
        edges = G_old.edges([n])
        if len(edges) != 0:
            neighbors = [e[1] for e in edges]
            for n2 in neighbors:
                if G_old.node[n2]['strategy'] == susceptible:
                    x = get_infected_neighbor_fraction(G_old,n2)
                    p = p_infect(x)/k
                    if np.random.binomial(1,p):
                        G.node[n2]['strategy'] = infected
        #recover
        x = get_infected_neighbor_fraction(G_old,n)
        p = p_recover(x)
        if np.random.binomial(1,p):
            G.node[n]['strategy'] = recovered
        

def update_graph(G,p_recover=lambda x: 1.0,p_infect= lambda x: x):
    G_old = G.copy()
    nodes= G_old.nodes()
    for n in nodes:
        update_node(G,G_old,n,p_recover,p_infect)
    return

def get_num_infected(G):
    return get_num_with_strategy(G,infected)
   
def get_num_with_strategy(G,strategy=infected):
    strategies = [G.node[n]['strategy'] for n in G.nodes()]
    return len([s for s in strategies if s == strategy])
        
def get_infected_neighbor_fraction(G,n):
    edges = G.edges([n])
    if len(edges) == 0:
        return 0.0
    strategies = [G.node[e[1]]['strategy'] for e in edges]
    x = 1.0*len([s for s in strategies if s == infected])/len(edges)
    return x

def p_recover(x):
    return 1.0

def p_infect(x):
    #beta = 10
    #return 1.0/(1 + exp(-beta*(x - 0.4)))
    return x

def simulate_graph_trajectory_adaptive(N,alpha,beta,plotting=False,k=None):
    if k is None:
        k = N-1
    p_desired = 0.1
    ns = [1]
    ts = [0]
    p_edge = 1.0*k/(N-1)
    print p_edge
    G = create_graph(N,p_edge,ns[0])
    
    def f(y):
        return alpha*y**2
    def birth_rate(y):
        return 1.0 + f(y)/y
    def death_rate(y):
        return 1.0 + beta
    
    
    max_rate = np.max([death_rate(1.0),birth_rate(1.0)])
    dt = p_desired/max_rate
    
    p_recover = lambda x: dt*death_rate(x)
    p_infect = lambda x: dt*birth_rate(x)
    
    if plotting:
        figure(1)
        pos = nx.spring_layout(G,iterations=100,k=2.0/sqrt(N_nodes))
    
    while ns[-1] > 0 and ns[-1] < N:
        update_graph(G,p_recover,p_infect)
        num_infected = get_num_infected(G)
        ns.append(num_infected)
        ts.append(ts[-1]+dt)
        if plotting:
            draw_graph(G,pos=pos)
            pause(0.05)
    return np.array(ns),np.array(ts)