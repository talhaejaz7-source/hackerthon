import numpy as np
import random
import pickle
import os

lanes = ["TOP", "BOTTOM", "LEFT", "RIGHT"]

Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

MODEL_FILE = "q_table.pkl"

# Load model if exists
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        Q = pickle.load(f)

def save_model():
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(Q, f)

def get_state(vehicles):
    return tuple([vehicles[l] // 5 for l in lanes])

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(lanes)

    if state not in Q:
        Q[state] = {lane: 0 for lane in lanes}

    return max(Q[state], key=Q[state].get)

def update_q(state, action, reward, next_state):
    if state not in Q:
        Q[state] = {lane: 0 for lane in lanes}

    if next_state not in Q:
        Q[next_state] = {lane: 0 for lane in lanes}

    old = Q[state][action]
    future = max(Q[next_state].values())

    Q[state][action] = old + alpha * (reward + gamma * future - old)