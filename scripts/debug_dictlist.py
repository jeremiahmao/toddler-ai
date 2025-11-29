#!/usr/bin/env python3
"""Debug DictList behavior with raw_obs."""

from toddler_ai.utils.dictlist import DictList

# Test 1: Basic attribute setting
print("Test 1: Basic attribute setting")
d = DictList()
d.raw_obs = ["obs1", "obs2", "obs3", "obs4"]
d.memory = [1, 2, 3, 4]
print(f"d.raw_obs exists: {'raw_obs' in d}")
print(f"d.memory exists: {'memory' in d}")
print(f"Keys in dict: {list(d.keys())}")
print()

# Test 2: Try accessing via attribute
print("Test 2: Access via attribute")
try:
    obs = d.raw_obs
    print(f"d.raw_obs = {obs}")
except KeyError as e:
    print(f"KeyError accessing d.raw_obs: {e}")
print()

# Test 3: Try accessing via dict key
print("Test 3: Access via dict key")
try:
    obs = d['raw_obs']
    print(f"d['raw_obs'] = {obs}")
except KeyError as e:
    print(f"KeyError accessing d['raw_obs']: {e}")
print()

# Test 4: Try indexing with list
print("Test 4: Indexing DictList with list field")
try:
    # This is what happens in DictList.__getitem__
    indices = [0, 1]
    sub_dict = DictList({key: value[indices[0]:indices[0]+1] for key, value in d.items()})
    print(f"After indexing [0]: {list(sub_dict.keys())}")
    print(f"sub_dict.raw_obs = {sub_dict.raw_obs if 'raw_obs' in sub_dict else 'MISSING'}")
    print(f"sub_dict.memory = {sub_dict.memory if 'memory' in sub_dict else 'MISSING'}")
except Exception as e:
    print(f"Error during indexing: {e}")
print()

# Test 5: The actual PPO pattern
print("Test 5: PPO pattern")
exps = DictList()
# Simulate raw observations from PPO
raw_obss = [{"image": f"img{i}", "instr": f"instr{i}"} for i in range(4)]
exps.raw_obs = raw_obss
exps.memory = [10, 20, 30, 40]
print(f"exps keys: {list(exps.keys())}")
print(f"'raw_obs' in exps: {'raw_obs' in exps}")

# Try to access like PPO does
try:
    batch_indices = [0, 1]
    batch_raw_obs = [exps.raw_obs[idx] for idx in batch_indices]
    print(f"Successfully accessed raw_obs: {batch_raw_obs[:2]}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Other error: {e}")