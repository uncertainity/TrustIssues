import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import copy
from copy import deepcopy
from analysis import *
from util import *
from prompt_code import *


class BaseGameStructure:
        
    def __init__(self,config):
        self.model_id = config.model_id
        self.NUM_AGENTS = config.num_agents
        self.NUM_ROUNDS = config.num_rounds
        self.START_ENDOWMENT = config.start_endowment
        self.BASE_DYNAMIC_PROMPT = config.base_prompt
        self.PERSONAS = config.personas
        self.wealth_dict = {i: config.start_endowment for i in range(config.num_agents)}
        self.kindness_dict = {i: 0.0 for i in range(config.num_agents)}
        self.agent_personas = {i: random.choice(list(config.personas.keys())) for i in range(config.num_agents)}
        
        self.graph_k = config.graph_k
        self.graph_p = config.graph_p
        self.graph_seed = config.graph_seed
            
        
    def getModel(self):
        quantization_config = BitsAndBytesConfig(
                                            load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4", # or "fp4"
                                            bnb_4bit_compute_dtype=torch.float16,
                                            bnb_4bit_use_double_quant=False,)
        model = AutoModelForCausalLM.from_pretrained(self.model_id,device_map = "auto",quantization_config = quantization_config,torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return model,tokenizer
    
    def execute_round_updates(self,graph,actions,wealth_dict,kindness_dict,model,tokenizer,alpha = 0.2):
        for agent_id,action in actions.items():
            for target in action.get("disconnected_from",[]):
                if graph.has_edge(agent_idx,target):
                    graph.remove_edge(agent_id,target)
        
        proposals = {i : set(actions[i].get("propose_connection_to",[])) for i in actions}
        #print("Proposals:",proposals)
        for i in range(self.NUM_AGENTS):
            for j in proposals[i]:
                if i in proposals.get(j, set()) and i != j:
                    graph.add_edge(i,j)
                    
        new_wealth = {i: wealth_dict[i] for i in range(self.NUM_AGENTS)}
        transfers = {i: {j: 0 for j in graph.neighbors(i)} for i in range(self.NUM_AGENTS)}

        
        for i, action in actions.items():
            to_give = action.get('to_neighbors', {})
            ##print("to give:",to_give)
            total_proposed = sum(max(0, int(v)) for v in to_give.values())

            # Budget check
            budget = wealth_dict[i]
            scale = 1.0 if total_proposed <= budget else budget / total_proposed

            for target_id_str, amount in to_give.items():
                target_id = int(target_id_str)
                if graph.has_edge(i, target_id):
                    actual_amount = int(int(amount) * scale)
                    new_wealth[i] -= actual_amount
                    new_wealth[target_id] += actual_amount
                    transfers[i][target_id] = actual_amount

        # 4. Update Global Kindness (EMA)
        for i in range(self.NUM_AGENTS):
            # Calculate how 'kind' agent i was to their neighbors
            neighbors_i = list(graph.neighbors(i))
            if not neighbors_i:
                # If no neighbors, kindness score drifts toward 0
                kindness_dict[i] = (1 - alpha) * kindness_dict[i]
                continue

            # Baseline: Average share of wealth
            fair_baseline = wealth_dict[i] / (len(neighbors_i) + 1)
            total_given = sum(transfers[i].values())
            avg_given = total_given / len(neighbors_i)

            # Normalize delta (-1 to 1 range)
            delta = (avg_given - fair_baseline) / 100
            delta = max(-1.0, min(1.0, delta))

            kindness_dict[i] = (1 - alpha) * kindness_dict[i] + alpha * delta
            
        for i in range(self.NUM_AGENTS):
            wealth_dict[i] = new_wealth[i]
            
        return kindness_dict,wealth_dict
    
    def run_evolutionary_simulation(self,initial_graph,agent_personas,model,tokenizer,num_rounds = 10, start_endowment = 100):
        current_graph = initial_graph.copy()
        # wealth_dict = {i: start_endowment for i in range(NUM_AGENTS)}
        # kindness_dict = {i: 0.0 for i in range(NUM_AGENTS)}
        wealth_dict = self.wealth_dict
        kindness_dict = self.kindness_dict
        history = []
        
        for r in range(num_rounds):
            print(f"\n{'='*20} STARTING ROUND {r} {'='*20}")
            round_actions = {}
            alive_ids = []
            prompts_by_agents = []
            for agent_id in range(self.NUM_AGENTS):
                persona = agent_personas[agent_id]
                
                if current_graph.degree(agent_id) == 0 and r > 0:
                    print(f"Agent {agent_id} ({persona}): ELIMINATED.")
                    round_actions[agent_id] = {"to_neighbors": {}, "disconnect_from": [], "propose_connection_to": []}
                    continue
                
                prompt = get_dynamic_prompt(agent_id, agent_personas,current_graph, wealth_dict, kindness_dict,self.BASE_DYNAMIC_PROMPT)
                alive_ids.append(agent_id)
                prompts_by_agents.append(prompt)
                
            responses = batched_llm_calls(prompts_by_agents,alive_ids,model,tokenizer)
            actions = segParser(responses,prompts_by_agents,model,tokenizer,max_retries = 2)
            
            for i in range(len(alive_ids)):
                persona = agent_personas[alive_ids[i]]
                action = actions[i]
                round_actions[alive_ids[i]] = action
                cot = action.get("social_chain_of_thought", "N/A")
                
                print(f"\n--- Agent {alive_ids[i]} ({persona}) ---")
                print(f"CoT: {str(cot)[:150]}...")
                #print(f"CoT: {action.get('social_chain_of_thought', 'N/A')[:150]}...")
                print(f"Giving: {action.get('to_neighbors', {})}")
                print(f"New Ties: {action.get('propose_connection_to', [])}")
                
            kindness_dict,wealth_dict = self.execute_round_updates(current_graph, round_actions, wealth_dict, kindness_dict,model,tokenizer)
            
            round_snapshot = {
                "round": r,
                "graph_edges": list(current_graph.edges()),
                "wealth": wealth_dict.copy(),
                "kindness": kindness_dict.copy(),
                "actions": copy.deepcopy(round_actions),
                "graph":current_graph.copy(),
            }
            history.append(round_snapshot)
            
            if current_graph.number_of_edges() == 0:
                print("\nCooperation has collapsed.")
                break
            
        return history
    
    def mainGame(self):
        model,tokenizer = self.getModel()
        G = nx.watts_strogatz_graph(n=self.NUM_AGENTS, k=self.graph_k, p=self.graph_p, seed=self.graph_seed)
        simulation_results = self.run_evolutionary_simulation(G, self.agent_personas,model,tokenizer,num_rounds=self.NUM_ROUNDS, start_endowment=self.START_ENDOWMENT)
        return simulation_results
        


