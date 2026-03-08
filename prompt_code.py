import torch, json
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import copy
from copy import deepcopy

def get_dynamic_prompt(agent_id,agent_personas,graph,wealth_dict,kindness_dict,BASE_DYNAMIC_PROMPT):
    persona = agent_personas[agent_id]
    neighbors_list = list(graph.neighbors(agent_id))
    current_wealth = wealth_dict[agent_id]
    ## Globally seen kindness summary
    kindness_summary = "\n".join([f"Agent {idx}: {score:+.2f}" for idx, score in kindness_dict.items()])
    prompt = f"""{BASE_DYNAMIC_PROMPT}
        ### Current State for Agent {agent_id}:
        - **Persona:** {persona}
        - **Current Wealth:** {current_wealth} coins
        - **Current Neighbors:** {neighbors_list}

        ### Global Reputation (Kindness Scores):
        {kindness_summary}

        Remember to output only a JSON object.
        
        \nIMPORTANT: You must make a decision. Even if you give 0, explain why in the social_chain_of_thought. If you are a 'Kind' persona, you should actively try to give or connect.
        """
    return prompt
    
    