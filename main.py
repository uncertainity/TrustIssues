import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from analysis import *
from util import *
from prompt_code import *
from games import BaseGameStructure
from gameConfig import GameConfig_1
import random
import pandas as pd

num_sims = 30
seeds = [random.randint(0, 10**9) for _ in range(num_sims)]

game_1_results = []
gameconfig_1 = GameConfig_1()

for i in range(len(seeds)):

    initial_graph_seed = seeds[i]
    
    gameconfig_1.graph_seed = initial_graph_seed
    game_1 = BaseGameStructure(gameconfig_1)
    simulation_results = game_1.mainGame()
    agent_personas = game_1.agent_personas.copy()
    sim_metrics = extract_simulation_metric(simulation_results,gameconfig_1,agent_personas,i+1)
    game_1_results += sim_metrics
    
    print(f"{i}-th run ended.")
    
df = pd.DataFrame(game_1_results)
filename = "game_1_" + "rounds_" + str(gameconfig_1.num_rounds) + "_sims" + str(num_sims) + ".csv"
df.to_csv(filename)
