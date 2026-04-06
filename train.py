import random
from ai_model import get_state, choose_action, update_q, save_model

lanes = ["TOP", "BOTTOM", "LEFT", "RIGHT"]

def random_traffic():
    return {lane: random.randint(0, 20) for lane in lanes}

episodes = 5000

for episode in range(episodes):
    vehicles = random_traffic()

    state = get_state(vehicles)
    action = choose_action(state)

    # simulate next traffic
    next_vehicles = random_traffic()
    next_state = get_state(next_vehicles)

    # reward: less total vehicles is better
    reward = -sum(next_vehicles.values())

    update_q(state, action, reward, next_state)

    if episode % 500 == 0:
        print(f"Training episode: {episode}")

# Save trained model
save_model()

print("✅ Training Complete & Model Saved!")