from flask import Flask, request, jsonify, render_template
from ai_model import get_state, choose_action

app = Flask(__name__)

reward_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.json
    vehicles = data.get("vehicles", {})

    state = get_state(vehicles)
    action = choose_action(state)

    # reward = negative congestion
    reward = -sum(vehicles.values())
    reward_history.append(reward)

    return jsonify({
        "green_lane": action
    })

@app.route("/graph", methods=["GET"])
def graph():
    return jsonify(reward_history[-20:])  # last 20 values

if __name__ == "__main__":
    app.run(debug=True)