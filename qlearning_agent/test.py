import pickle
import numpy as np
from collections import defaultdict
import time

MODELS_PATH = "models"
MODEL_FILE = f"{MODELS_PATH}/qtable_improved.pkl"

print("Loading model...")
start = time.time()
with open(MODEL_FILE, 'rb') as f:
    data = pickle.load(f)
print(f"Loaded in {time.time() - start:.2f}s")

q_table_data = data['q_table'] if isinstance(data, dict) and 'q_table' in data else data
q_table = defaultdict(lambda: np.zeros(25, dtype=np.float32), q_table_data)
print(f"Q-table states: {len(q_table)}")

def get_action(obs: np.ndarray, valid_actions: list) -> int:
    state_key = tuple(obs.astype(np.int16).tolist())
    q_values = q_table[state_key]
    masked_q = np.full(25, -np.inf)
    masked_q[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q))

obs = np.zeros(62, dtype=np.int16)
obs[0] = 2
obs[1] = 1
obs[10] = 1
obs[20] = 1
obs[24] = 1
obs[25] = 1
obs[27] = 1
obs[49] = 4
obs[50] = 1

valid_actions = [0, 1, 3]
action = get_action(obs, valid_actions)
print(f"Test observation -> action: {action} (from valid: {valid_actions})")
print("Inference OK")