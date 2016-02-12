from pylab import *
import numpy.random as random
import networkx as nx#python graph library
cooperate = 0
defect = 1
cooperate_and_know = 2
b = 2
c = 0.5
payoffs = np.array([[b - c, -c,b-c],[b,0,0],[b-c,0,b-c]])

def plot_histograms(all_thresholds,p,N_nodes):

    flattened_thresholds = [item for sublist in all_thresholds for item in sublist]
    min_thresh = min([item for item in flattened_thresholds if item >= 1.0])
    max_thresh = max(flattened_thresholds)

    for j in range(len(all_thresholds)):
        thresholds = all_thresholds[j]
        subplot(1,len(all_thresholds),j+1)
        weights = ones_like(thresholds)/len(thresholds)
        ax1 = hist(thresholds,alpha = 0.4,bins=20,weights=weights)
        xlim([min_thresh,max_thresh])
        xlabel(r'$\left(\frac{b}{c}\right)^*$',size=25)
        if j == 0:
            ylabel(r'probability',size=15)
        else:
            gca().set_yticklabels([])
        grid()
    suptitle(r'$p = ' + str(p) + '$, $N_{nodes} = ' + str(N_nodes) + '$',size=20)
    show()


def plot_colored_graphs(all_graphs,all_thresholds,all_degrees,graph_names,p):
    figure()
    flattened_colors = [item for sublist in all_thresholds for item in sublist]
    vmin = min(flattened_colors)
    vmax = max(flattened_colors)
    N_nodes = len(all_thresholds[0])

    for j in range(len(all_thresholds)):
        thresholds = all_thresholds[j]
        degrees = all_degrees[j]
        G = all_graphs[j]
        k = 2*len(G.edges())/len(G.nodes())
        node_color = thresholds
        node_size = [100.0*degrees[i]/k for i in range(len(G.nodes()))]
        node_labels = {i:thresholds[i] for i in range(len(thresholds))}

        if j == -1:
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G,k=1.4/sqrt(len(G.nodes())))
        print vmin,vmax
        subplot(1,len(all_thresholds),j+1)
        ax2 = nx.draw_networkx_nodes(G,pos,alpha=0.6,node_color=node_color,vmin=vmin,vmax=vmax,node_size=node_size,labels=node_labels,with_labels=False)
        nx.draw_networkx_edges(G,pos,alpha=0.2)
        gca().set_xticks([])
        gca().set_yticks([])
        xlabel(graph_names[j],size=20)

    cbar = colorbar(ax2)
    #labels = ['{:.2f}'.format(vmin + (vmax-vmin)*float(item.get_text())) for item in cbar.ax.get_yticklabels()]
    #cbar.ax.set_yticklabels(labels)
    cbar.ax.set_ylabel(r'$\left(\frac{b}{c}\right)^*$',size=25)

    #cbar.ax.set_xlabel(r'node size $\sim k_i$',size=15)
    suptitle(r'$p = ' + str(p) + '$, $N_{nodes} = ' + str(N_nodes) + '$, node size $\sim k$',size=20)

    #normalize the number of edges for these graphs
    #see whether springlayout does more iterations
    show()

def get_payoff(player_strategy,opponent_strategy,payoffs):
    return payoffs[player_strategy,opponent_strategy]

def draw_graph(G):
    node_color = [0 if G.node[i]['strategy'] == cooperate else 0.1 for i in range(len(G.nodes()))]
    node_label = {i:G.node[i]['score'] for i in range(len(G.nodes()))}
    nx.draw(G,node_color=node_color,labels=node_label,cmap='spring')

def zero_scores(G):
    for n in G.nodes():
        G.node[n]['score'] = 0.0

def update_score(G,payoffs,prob_of_known_fn = lambda x,y: 1.0):
    for e in G.edges():
        update_score_on_edge(e,G,prob_of_known_fn,payoffs)

def update_score_on_edge(e,G,prob_of_known_fn,payoffs = np.array([[b - c, -c,b-c],[b,0,0],[b-c,0,b-c]])):
    
    s0 = G.node[e[0]]['strategy']
    s1 = G.node[e[1]]['strategy']
    
    #if the identity of opponent is known, change to state 2
    p_known = prob_of_known_fn(e,G)
    known = np.random.binomial(1,p_known)
    if s0 == cooperate and known:
        s0 = cooperate_and_know

    p_known = prob_of_known_fn(e[::-1],G)
    known = np.random.binomial(1,p_known)
    if s1 == cooperate and known:
        s1 = cooperate_and_know

    G.node[e[0]]['score'] += payoffs[s0,s1]
    G.node[e[1]]['score'] += payoffs[s1,s0]
    



        
def get_average_scores(G):
    #get the average scores for cooperators and defectors
    strategy_names = ['cooperate','defect']
    scores = np.zeros(2)
    num_nodes = np.zeros(2)
    for i in G.nodes():
        scores[G.node[i]['strategy']] += G.node[i]['score']
        num_nodes[G.node[i]['strategy']] += 1
    return scores/num_nodes

def create_graph(N_nodes,p_edge,p_coop,random_number=True):
    #Random graph
    #p_coop is the fraction of cooperators
    G = nx.gnp_random_graph(N_nodes,p_edge,directed=False)
    return set_graph_strategies(G, N_nodes,p_coop,random_number)
    
def set_graph_strategies(G, N_nodes,p_coop,random_number=True):
    if random_number:
        for n in G.nodes():
            G.node[n]['strategy'] = np.random.binomial(1,1-p_coop)
            G.node[n]['score'] = 0
    else:
        nodes = G.nodes()
        random.shuffle(nodes)
        for i,n in enumerate(nodes):
            if i >= int(round(p_coop*N_nodes)):
                G.node[n]['strategy'] = 1
            else:
                G.node[n]['strategy'] = 0
            G.node[n]['score'] = 0
    return G


def expected_return_c(b,c,p_coop,p_edge,N_nodes,p_know):
    return (p_coop*(b-c) + (1 - p_coop)*(1-p_know)*(-c))*N_nodes*p_edge

def expected_return_d(b,c,p_coop,p_edge,N_nodes,p_know):
    return (p_coop*(1-p_know)*b)*N_nodes*p_edge

def get_p_known(e,G,p):
    num_mutual_friends =len(set(G.neighbors(e[0])) & set(G.neighbors(e[1])))
    return 1 - (1-p)*(1- p**2)**num_mutual_friends

def get_p_known_first_order(mutual_neighbors,p):
    return 1-(1-p)*(1-p**2)**mutual_neighbors

def get_p_known_direct(e,G,p):
    return p


def create_small_world_graph(N_nodes,p_edge,p_coop,random_number=True):
    n = N_nodes
    k = int(p_edge*N_nodes)
    p = 0.5
    #Random graph
    #p_coop is the fraction of cooperators
    G = nx.watts_strogatz_graph(n,k,p)
    return set_graph_strategies(G, N_nodes,p_coop,random_number)

def create_scale_free_graph(N_nodes,p_edge,p_coop,random_number=True):
    #scale free and small world
    #Growing Scale-Free Networks with Tunable Clustering
    n = N_nodes
    m = int(0.5*p_edge*N_nodes)
    p = 1.0
    #Random graph
    #p_coop is the fraction of cooperators
    G = nx.powerlaw_cluster_graph(n,m,p)
    return set_graph_strategies(G, N_nodes,p_coop,random_number)

def get_average_scores_for_graph_type(N_nodes,p_edge,p_coop,p,num_trials,graph_fn,random_number=True):
    coop_score = 0
    def_score = 0
    for j in range(num_trials):
        G = graph_fn(N_nodes,p_edge,p_coop,random_number)
        update_score(G,payoffs,lambda x,y: get_p_known(x,y,p))
        scores = get_average_scores(G)
        coop_score += scores[0]
        def_score += scores[1]
    coop_score /= num_trials
    def_score /= num_trials
    return coop_score,def_score

def get_average_nash_scores_for_graph_type(N_nodes,p_edge,p_coop,p,num_trials,graph_fn,random_number=True,strategy = defect):
    score = 0.0
    num_edges = 0.0
    for j in range(num_trials):
        G = graph_fn(N_nodes,p_edge,p_coop,random_number)
        score += get_average_nash_score(G,payoffs,lambda x,y: get_p_known(x,y,p),strategy)
        num_edges += len(G.edges())
    score /= num_trials
    num_edges /= num_trials
    return score,num_edges


#This is the expected score per edge (i.e per neighbor of a given node).
def get_average_nash_score(G,payoffs,prob_of_known_fn,strategy = defect):
    score = 0
    for n in G.nodes():
        if strategy == defect:
            score += get_score_for_single_defector_node(n,G,payoffs,prob_of_known_fn)
        else:
            score += get_score_for_single_cooperator_node(n,G,payoffs,prob_of_known_fn)
    if len(G.nodes()) == 0:
        return 0.0
    return 0.5*score/len(G.edges())

def get_score_for_single_defector_node(n,G,payoffs = np.array([[b - c, -c,b-c],[b,0,0],[b-c,0,b-c]]),prob_of_known_fn = lambda x,y: 1.0):
    score = 0
    edges = G.edges([n])
    for e in edges:
        p_known = prob_of_known_fn(e[::-1],G)
        score += payoffs[defect,cooperate_and_know]*p_known + payoffs[defect,cooperate]*(1 - p_known)
    return score

def get_score_for_single_cooperator_node(n,G,payoffs = np.array([[b - c, -c,b-c],[b,0,0],[b-c,0,b-c]]),prob_of_known_fn = lambda x,y: 1.0):    
    score = 0
    edges = G.edges([n])
    for e in edges:
        p_known = prob_of_known_fn(e,G)
        score += payoffs[cooperate_and_know,defect]*p_known + payoffs[cooperate,defect]*(1 - p_known)
    return score

def get_average_mutual_neighbors_for_graph_type(N_nodes,p_edge,p,num_trials,create_graph_fn,random_number=True):
    p_known, p_known_curr, mutual_neighbors, mutual_neighbors_curr,num_edges,p_coop = 0.0,0.0,0.0,0.0,0.0,1.0
    for i in range(num_trials):
        G = create_graph_fn(N_nodes,p_edge,p_coop,random_number)
        print len(G.nodes()), len(G.edges())
        p_known_curr,mutual_neighbors_curr = get_average_p_known(G,prob_of_known_fn = lambda x,y: get_p_known(x,y,p))
        p_known += p_known_curr
        mutual_neighbors += mutual_neighbors_curr
        num_edges += len(G.edges())
    num_edges /= num_trials
    p_known /= num_trials
    mutual_neighbors /= num_trials
    return p_known,mutual_neighbors,num_edges

#This is the expected score per neighbor of a given node.
def get_average_p_known(G,prob_of_known_fn= lambda x,y: 1.0):
    p_known, p_known_curr, mutual_neighbors, mutual_neighbors_curr = 0.0,0.0,0.0,0.0
    for n in G.nodes():
        p_known_curr,mutual_neighbors_curr = add_p_known_for_single_node(G,n,prob_of_known_fn)    
        p_known += p_known_curr
        mutual_neighbors += mutual_neighbors_curr
    p_known /= 2*len(G.edges())
    mutual_neighbors /= 2*len(G.edges())
    return p_known,mutual_neighbors
        
        
def add_p_known_for_single_node(G,n,prob_of_known_fn):
    p_known, p_known_curr, mutual_neighbors, mutual_neighbors_curr = 0.0,0.0,0.0,0.0
    edges = G.edges([n])
    if len(edges) == 0:
        print 'no neighbors'
        return 1.0,0.0
    for e in edges:
        p_known += prob_of_known_fn(e[::-1],G)
        mutual_neighbors += len(set(G.neighbors(e[0])) & set(G.neighbors(e[1])))
#     p_known /= len(edges)
#     mutual_neighbors /= len(edges)
    return p_known,mutual_neighbors

def update_threshold(G,prob_of_known_fn = lambda x,y: 1.0):
    for n in G.nodes():
        num_edges = len(G.edges([n]))
        if len(G.edges([n])) > 0:
            p_known = add_p_known_for_single_node(G,n,prob_of_known_fn)[0]/len(G.edges([n]))
        else:
            p_known = 0
        G.node[n]['threshold'] = 1.0/p_known if p_known > 0 else 1.0
