import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import networkx as nx
import scipy

#-----------------Task 12-----------------------------------

CUT_IN_THRESHOLD = 3
CUT_OUT_THRESHOLD = 25
DIRECTORY = "Wind datasets"

G = nx.read_gexf("graph.gexf")

gust_speeds = []

# Loop for counting total of times that max gust speed goes over threshold
# in any datasets and also counting the total of timesteps
for file in os.listdir(DIRECTORY):
    file_path = f"{DIRECTORY}/{file}"
    df = pd.read_csv(file_path, usecols=["Time [Local time]", "Maximum gust speed [m/s]"])
    for speed in df["Maximum gust speed [m/s]"]: 
        gust_speeds.append(speed)

# Probability for turbine failure based on too big of a gust
k, loc, scale = scipy.stats.weibull_min.fit(gust_speeds, floc=0)
overall_failure_probability = scipy.stats.weibull_min.cdf(CUT_IN_THRESHOLD, k, loc=0, scale=scale) + (1 - 
                            scipy.stats.weibull_min.cdf(CUT_OUT_THRESHOLD, k, loc=0, scale=scale))
print(overall_failure_probability)

risk_scores = pd.read_csv("combined.csv", usecols=["location", "risk_score_model"])
#Adding failure probability for each turbine
risk_scores.insert(2, "failure_probability", risk_scores["risk_score_model"] 
                   * overall_failure_probability)


edgelist = nx.to_pandas_edgelist(G)
#print(edgelist)
print(risk_scores)

#Creating propagation matrix where row is source of failure
#and column is the target. Values in cells are probabilities for failure propagation. 
propagation_matrix = np.zeros((12,12))

edge_lookup = {}
for _, row in edgelist.iterrows():
    edge_lookup[(row["source"], row["target"])] = row["weight"]
    edge_lookup[(row["target"], row["source"])] = row["weight"]

for i in range(12):
    for j in range(12):
        if j==i:
            continue
        
        source = risk_scores["location"][i]
        target = risk_scores["location"][j]
        weight = edge_lookup.get((source, target), 0)
        propagation_matrix[i,j] = weight * risk_scores["risk_score_model"][j]

# Display of the propagation matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(propagation_matrix, cmap="YlOrRd")

plt.colorbar(im, ax=ax)

ax.set_xticks(range(12))
ax.set_yticks(range(12))
ax.set_xticklabels(risk_scores["location"], rotation=45, ha="right")
ax.set_yticklabels(risk_scores["location"])

for i in range(12):
    for j in range(12):
        ax.text(j, i, f"{propagation_matrix[i, j]:.3f}",
                ha="center", va="center", fontsize=7)

ax.set_title("Propagation Matrix [i → j]")
plt.tight_layout()
plt.savefig("plots_and_fiqures/propagation_matrix.png")
#plt.show()

#-----------------Task 12-----------------------------------
adj_list = {}
node_id = {}

for i, key in enumerate(risk_scores["location"]): 
    adj_list[key] = []
    node_id[key] = i

for source, target in edge_lookup.keys():
    adj_list[source].append(target)



#-----------------Task 13-----------------------------------

def one_simulation(queue_propagation, risk_score, failed, adjacency_list, propagation_matrix, node_id, 
                   queue_independent=None, primary=True):
    if not primary: 
        test = queue_independent.pop(0)
        if test in failed:
            return
        
        fail = np.random.rand() < risk_score["failure_probability"][node_id[test]]
        if not fail:
            return
          
        failed.append(test)
        for node in adjacency_list[test]:
            if node not in failed:
                queue_propagation.append((test, node))
        return


    failure_source, test = queue_propagation.pop(0)
    fail = np.random.rand() < propagation_matrix[node_id[failure_source], node_id[test]]
    if not fail:
        return

    failed.append(test)
    for node in adjacency_list[test]:
        if node not in failed:
            queue_propagation.append((test, node))

    

def simulate_propagation(risk_score, propagation_matrix, adjacency_list, n_simulations=10000):
    sets = []
    for i in range(n_simulations):
        failed = []
        queue_propagation = []
        sorted_risks = risk_score.sort_values("risk_score_model", ascending=False)
        queue_independent = list(sorted_risks["location"])

        while len(queue_independent) > 0 or len(queue_propagation) > 0:
            if len(queue_propagation) > 0:
                one_simulation(queue_propagation, risk_score, failed, adjacency_list, propagation_matrix, node_id)
                continue

            one_simulation(queue_propagation, risk_score, failed, adjacency_list, propagation_matrix, node_id, queue_independent, primary=False)
        
        sets.append(failed)

    return sets

results = simulate_propagation(risk_scores, propagation_matrix, adj_list)

sizes = [len(s) for s in results]
print(f"Mean failures per simulation: {np.mean(sizes):.2f}")
print(f"P(zero failures): {np.mean([s == 0 for s in sizes]):.3f}")
print(f"P(propagation > 3): {np.mean([s > 3 for s in sizes]):.3f}")

# Per-turbine failure rate
all_locations = list(risk_scores["location"])
failure_rates = {}

for loc in all_locations:
    rate = np.mean([loc in scenario for scenario in results])
    failure_rates[loc] = rate
    #print(f"{loc}: {rate*100:.1f}% of simulations")

def plot_propagation(failed, risk_score, edgelist, title="Propagation Simulation"):
    graph = nx.from_pandas_edgelist(edgelist, "source", "target", edge_attr="weight")
    
    pos = nx.spring_layout(graph, seed=42)  # or nx.kamada_kawai_layout(G)
    
    node_colors = ["red" if node in failed else "steelblue" for node in graph.nodes()]
    node_sizes  = [risk_score["risk_score_model"][node_id[node]] * 1000 for node in graph.nodes()]
    edge_weights = [graph[u][v]["weight"] * 3 for u, v in G.edges()]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white")
    nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.5)
    
    plt.title(title)
    plt.axis("off")
    plt.show()

fig = plt.clf()
plot_propagation(results[0], risk_scores, edgelist)


#-----------------Task 14-----------------------------------