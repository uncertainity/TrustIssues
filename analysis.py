import torch, json
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import copy
from copy import deepcopy


PERSONA_COLORS = {
    'very_kind': '#00cc00',       # Dark Green
    'moderately_kind': '#66ff66', # Light Green
    'neutral': '#ffff66',         # Yellow
    'moderately_selfish': '#ffad33', # Orange
    'extremely_selfish': '#ff4d4d'   # Red
}



def plot_network_topography(G, agent_personas, PERSONA_COLORS = PERSONA_COLORS, title="Network Topography", seed=42):
    plt.figure(figsize=(10, 7))

    # Stable layout
    pos = nx.spring_layout(G, seed=seed)

    # Node colors from personas
    node_colors = [PERSONA_COLORS[agent_personas[i]] for i in G.nodes()]

    # Draw graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=800,
        font_weight='bold',
        edge_color='gray'
    )

    plt.title(title)

    # Legend
    markers = [
        plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='', markersize=10)
        for color in PERSONA_COLORS.values()
    ]

    plt.legend(
        markers,
        PERSONA_COLORS.keys(),
        title="Personas",
        loc='upper left',
        bbox_to_anchor=(1, 1)
    )

    plt.show()
    


def extract_round_metric(round_result,agent_personas,num_agents,sim_id = None):
    dict_keys = list(round_result.keys())
    rows = []
    round_num = round_result["round"]
    round_wealth = round_result["wealth"]
    round_kidness = round_result["kindness"]
    round_actions = round_result["actions"]

    total_given = {i:0 for i in range(num_agents)}
    total_received = {i:0 for i in range(num_agents)}

    for giver,action in round_actions.items():
        for receiver,amounts in action.get("to_neighbors",{}).items():
            total_given[giver] += int(amounts)
            total_received[receiver] += int(amounts)
            
    edges = round_result["graph_edges"]
    G = nx.Graph()
    G.add_nodes_from(range(num_agents))
    G.add_edges_from(edges)
    
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G)
    except:
        eigenvector = {i:0 for i in range(num_agents)}

    #print("Given:",total_given)
    #print("kindness:",wealth)
    
    
    for agent in range(num_agents):
        rows.append({"Sim_ID":sim_id,"Round":round_num,"Agent":agent,"persona":agent_personas[agent],"Given":total_given[agent],
                     "Received":total_received[agent],"wealth":round_wealth[agent],"kindness":round_kidness[agent],
                     "eigenvector":eigenvector[agent],"betweenness":betweenness[agent],"closeness":closeness[agent],
                     "degree":degree[agent]})
        
    return rows


def extract_simulation_metric(simulation_results,config,agent_personas,simulation_id):
    sim_metric = []
    num_agents = config.num_agents
    
    for sim in range(len(simulation_results)):
        round_result = simulation_results[0]
        round_metric = extract_round_metric(round_result,agent_personas,num_agents,simulation_id)
        sim_metric.extend(round_metric)
    return sim_metric
        