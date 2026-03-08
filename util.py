import torch, json
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import copy
from copy import deepcopy
from json_repair import repair_json

def parse_simulation_response(response_text):
    """
    Uses regex to extract JSON from the model response, sanitizes control
    characters that break json.loads, and ensures mandatory keys are present.
    """
    default = {
        "social_chain_of_thought": "AGENT FAILED TO PROVIDE REASONING.",
        "to_neighbors": {},
        "disconnect_from": [],
        "propose_connection_to": [],
        "reason": "No summary provided."
    }

    try:
        # 1. Extract the JSON block using regex
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)

            # 2. Sanitize: Replace unescaped control characters within the string
            # specific to newlines, tabs, and carriage returns often found in LLM CoT
            # We escape them so json.loads treats them as part of the string value
            sanitized_str = json_str.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

            # Note: Simple replacement might double-escape if the LLM already escaped them.
            # A more robust regex approach for interior control characters:
            sanitized_str = re.sub(r'[\x00-\x1F]+', ' ', json_str)

            parsed = json.loads(sanitized_str)

            # 3. Enforce mandatory keys and defaults
            for key, val in default.items():
                if key not in parsed:
                    parsed[key] = val
            return parsed
    except Exception as e:
        print(f"Parsing error during sanitization: {e}")

    return default


# def llm_call(messages,model,tokenizer,max_new_tokens = 512,max_retries = 2):
#     enc = tokenizer.apply_chat_template(messages,add_generation_prompt = True,return_tensors = "pt",return_dict = True).to(model.device)
#     with torch.no_grad():
#         output = model.generate(input_ids = enc["input_ids"],attention_mask = enc["attention_mask"],max_new_tokens = max_new_tokens,
#                                 do_sample = False,pad_token_id = tokenizer.eos_token_id)
#         try:
#             parse_simulation_response(output)
        
        
# def llm_call(messages, model,tokenizer, max_new_tokens=512):
#     # Helper to interface with the model for the evolutionary game
#     enc = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         return_dict=True
#     ).to(model.device)

#     with torch.no_grad():
#         output = model.generate(
#             input_ids=enc["input_ids"],
#             attention_mask=enc["attention_mask"],
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     response_text = tokenizer.decode(output[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    
    
#     return response_text   


def llm_call(messages, model, tokenizer, max_new_tokens=512, max_retries=2):
    def _gen(msgs, max_new):
        enc = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.decode(
            out[0][enc["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

    response_text = _gen(messages, max_new_tokens)
    parsed = parse_simulation_response(response_text)

    # If parser didn't fall back to default, accept it
    if parsed.get("reason") != "No summary provided." or re.search(r'\{.*\}', response_text, re.DOTALL):
        # (the 'or' part is optional; see note below)
        return parsed

    # Repair loop
    candidate_text = response_text
    for _ in range(max_retries):
        repair_messages = [
            {"role": "system", "content": "Output VALID JSON ONLY. No markdown. No extra text."},
            {"role": "user", "content": candidate_text},
        ]
        repaired_text = _gen(repair_messages, 200)
        parsed2 = parse_simulation_response(repaired_text)

        if parsed2.get("reason") != "No summary provided.":
            return parsed2

        candidate_text = repaired_text

    print("Falling back to default action.")
    print("RAW OUTPUT:\n", response_text)
    return parsed


def preparse_llm_json(text):
    """
    Extracts and cleans LLM JSON before json.loads().
    Fixes common issues:
    - single quotes
    - unquoted keys
    - trailing commas
    - control characters
    """

    # 1. Extract first JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None

    s = match.group(0)

    # 2. Remove control characters
    s = re.sub(r'[\x00-\x1F]+', ' ', s)

    # 3. Replace single quotes with double quotes
    s = re.sub(r"'", '"', s)

    # 4. Quote unquoted keys
    # {key: value} -> {"key": value}
    s = re.sub(r'(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:', r'"\1":', s)

    # 5. Remove trailing commas
    s = re.sub(r',\s*([}\]])', r'\1', s)

    return s


def batched_llm_calls(prompts_by_agents,agent_ids, model, tokenizer, max_new_tokens=1024):
    texts = []
    tokenizer.pad_token = tokenizer.eos_token
    for aid in agent_ids:
        messages = [{"role": "user", "content": prompts_by_agents[aid]}]
        text = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize = False)
        texts.append(text)
    print("Inside the batched llm calls")
    enc = tokenizer(texts,return_tensors="pt",padding=True,truncation=True).to(model.device)

    with torch.inference_mode():
        out = model.generate(**enc,max_new_tokens = max_new_tokens,do_sample = True,temperature = 0.8,pad_token_id = tokenizer.eos_token_id)
    
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    responses = {}
    for i,aid in enumerate(agent_ids):
        gen_tokens = out[i][input_lens[i]:]
        responses[aid] = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    return responses
    
    

def single_llm_call(response,model,tokenizer,max_new = 200):
    enc = tokenizer.apply_chat_template(response,add_generation_prompt=True,return_tensors="pt",return_dict=True).to(model.device)

    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"],max_new_tokens=max_new,do_sample=True,
            temperature=0.8,pad_token_id=tokenizer.eos_token_id,)

    return tokenizer.decode(out[0][enc["input_ids"].shape[-1]:],skip_special_tokens=True).strip()


def parse_simulation_response_with_error(response_text):
    """
    Uses regex to extract JSON from the model response, sanitizes control
    characters that break json.loads, and ensures mandatory keys are present.
    """
    default = {
        "social_chain_of_thought": "AGENT FAILED TO PROVIDE REASONING.",
        "to_neighbors": {},
        "disconnect_from": [],
        "propose_connection_to": [],
        "reason": "No summary provided."
    }

    try:
        # 1. Extract the JSON block using regex
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            sanitized_str = repair_json(json_str)

            # 2. Sanitize: Replace unescaped control characters within the string
            # specific to newlines, tabs, and carriage returns often found in LLM CoT
            # We escape them so json.loads treats them as part of the string value
            ##sanitized_str = json_str.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

            # Note: Simple replacement might double-escape if the LLM already escaped them.
            # A more robust regex approach for interior control characters:
            ##sanitized_str = re.sub(r'[\x00-\x1F]+', ' ', json_str)

            parsed = json.loads(sanitized_str)

            # 3. Enforce mandatory keys and defaults
            for key, val in default.items():
                if key not in parsed:
                    parsed[key] = val
            return parsed,None
    except Exception as e:
        print(f"Parsing error during sanitization: {e}")
        return default,e
    
    return default,None


def segParser(responses,prompts_by_agents,model,tokenizer,max_retries = 2):
    actions = []
    for r in range(len(responses)):
        response_text = responses[r]
        parsed,e = parse_simulation_response_with_error(response_text)
        # if parsed.get("reason") != "No summary provided." or re.search(r'\{.*\}', response_text, re.DOTALL):
        #     # (the 'or' part is optional; see note below)
        #     actions.append(parsed)
        if e is None:
            actions.append(normalize_action(parsed))
        else:
            original_text = prompts_by_agents[r]
            #print(f"Agent id: {r}")
            #print("Original Text:",original_text)
            for a in range(max_retries):
                print(f"retry no {a}")
                repair_messages = [
                    {"role": "system", "content": "Output VALID JSON ONLY. No markdown. No Extra Text."},
                    {"role": "user", "content": f"Parser error: {e}\n\nText to fix:\n{original_text}"},
                ]
                repaired_text = single_llm_call(repair_messages,model,tokenizer,512)
                parsed2,e = parse_simulation_response_with_error(repaired_text)

                #if parsed2.get("reason") != "No summary provided.":
                if e is None:
                    actions.append(normalize_action(parsed2))
                original_text = repaired_text
                
            if e is not None:
                actions.append(normalize_action(parsed2))
            
    return actions    

def normalize_action(action):
    default = {
        "to_neighbors": {},
        "disconnect_from": [],
        "propose_connection_to": [],
        "reason": "No summary provided.",
        "social_chain_of_thought": "AGENT FAILED TO PROVIDE REASONING."
    }

    if not isinstance(action, dict):
        return default

    action = {**default, **action}

    # to_neighbors must be a dict of agent_id -> numeric amount
    tn = action.get("to_neighbors", {})
    clean_tn = {}
    if isinstance(tn, dict):
        for k, v in tn.items():
            try:
                ki = int(k)
                vi = int(v)
                clean_tn[ki] = vi
            except Exception:
                continue
    action["to_neighbors"] = clean_tn

    # disconnect_from must be a list[int]
    df = action.get("disconnect_from", [])
    if not isinstance(df, list):
        df = []
    action["disconnect_from"] = [int(x) for x in df if str(x).isdigit()]

    # propose_connection_to must be a list[int]
    pc = action.get("propose_connection_to", [])
    if not isinstance(pc, list):
        pc = []
    action["propose_connection_to"] = [int(x) for x in pc if str(x).isdigit()]

    # reason / CoT must be strings
    if not isinstance(action.get("reason"), str):
        action["reason"] = default["reason"]
    if not isinstance(action.get("social_chain_of_thought"), str):
        action["social_chain_of_thought"] = default["social_chain_of_thought"]

    return action    