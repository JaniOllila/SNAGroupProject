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

sorted = risk_scores.sort_values("risk_score_model", ascending=False)
queue_secondary = list(sorted["location"])

print(queue_secondary)


#-----------------Task 13-----------------------------------

def one_simulation(queue, failed, threshold, adjacency_list): 
    test = queue.pop(0)
    fail = np.random.rand() < threshold
    if fail:     
        failed.append(test)
        for node in adjacency_list[test]:
            if node not in failed:
                queue.append(node)
                

def simulate_propagation(risk_score, propagation_matrix, adjacency_list, n_simulations=10000):
    sets = []
    for i in range(n_simulations):
        failed = []
        queue_primary = []
        sorted = risk_score.sort_values("risk_score_model", ascending=False)
        queue_secondary = list(sorted["location"])

        while len(queue) > 0:
            test = queue.pop(0)
            fail = np.random.rand() < risk_score["failure_probability"][node_id[test]]
            if not fail:
                continue
                
            failures.append(test)
            for node in adjacency_list[test]:
                if node not in failures:
                    queue.append(node)
        
        sets.append(failures)

    return None



# Results
print(f"Mean turbines failed per simulation: {cascade_sizes.mean():.2f}")
print(f"Max cascade size: {cascade_sizes.max()}")
print(f"P(at least 1 failure): {(cascade_sizes > 0).mean():.3f}")
print(f"P(more than 3 failures): {(cascade_sizes > 3).mean():.3f}")

# Most vulnerable turbines (how often each turbine ends up failed)
failure_rates = failed_sets.mean(axis=0)
for i, loc in enumerate(risk_scores["location"]):
    print(f"{loc}: failed in {failure_rates[i]*100:.1f}% of simulations")
