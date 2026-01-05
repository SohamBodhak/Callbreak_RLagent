# Callbreak AI Arena (Web Interface)

**File:** `app_v6_algo.py`

This application serves as the interactive frontend for the Callbreak Reinforcement Learning Project. It is a **Flask + SocketIO** web application that allows human players to compete directly against trained Reinforcement Learning agents in a real-time, animated card game environment.

---

## 🌟 Key Features

* **Dynamic Model Switching**: Choose your opponent's "brain" in real-time. The app supports:
    * **PPO** (Proximal Policy Optimization)
    * **A2C** (Advantage Actor Critic)
    * **A3C** (Asynchronous Advantage Actor Critic)
* **MCTS Integration**: AI moves are not just raw policy outputs; they are refined using **Monte Carlo Tree Search (MCTS)** for strategic decision-making.
* **Adjustable Difficulty**: Control the depth of the MCTS simulations to balance speed vs. intelligence:
    * *25 Simulations (Fast)*
    * *100 Simulations (Balanced)*
    * *400 Simulations (Strong)*
* **Smart Bidding**: Uses a dedicated Neural Network (`BiddingNet`) to predict optimal bids based on hand strength.
* **Real-Time Animations**: Smooth card animations and instantaneous state updates using WebSockets.

---

## 📋 Prerequisites

Before running the application, ensure you have the following dependencies and files in your directory.

### Python Libraries
Ensure you have the following Python libraries installed in your environment:
* `flask`
* `flask-socketio`
* `torch`
* `numpy`
* `stable-baselines3`
* `sb3-contrib`
'''bash
pip install -r requirements.txt


### Required Project Files
The `app_v6_algo.py` script relies on these local Python modules (found in your repository):

* `callbreak_env.py`: The game logic environment.
* `mcts_agent.py`: The MCTS implementation.

### Model Checkpoints
The application attempts to load pre-trained models. Ensure these files are present in the **root directory** (same folder as the app script):

| File Name | Description |
| :--- | :--- |
| `ppo_callbreak_zero_pro.zip` | The trained PPO model (Stable Baselines 3). |
| `a2c_callbreak_final.zip` | The trained A2C model (Stable Baselines 3). |
| `a3c_callbreak_final.pth` | The PyTorch state dictionary for the A3C network. |
| `bidder_network.pth` | The trained neural network for predicting bids. |

> **Note:** If a model file is missing, the `AIManager` will print a failure message to the console. The app will typically continue running, defaulting to available models or fallback logic.

---

## 🚀 How to Run

### Navigate to the project directory
'''bash
cd Callbreak/Call-Break-The-card-game-in-python/
'''
### Run the application
'''bash
python app_v6_algo.py
'''
### Open the interface
Open your web browser and navigate to:
http://127.0.0.1:5001

---

## 🎮 How to Play

### Game Setup
* **Bot Intelligence:** Select which RL architecture you want to play against via the dropdown menu.
* **Simulation Depth:** Select the difficulty. Higher simulations = Smarter bots, but slightly slower thinking time.
* Click **Start Game**.

### Bidding
At the start of the round, analyze your cards. A modal will appear asking for your bid (1-13).
* Use the slider to choose your bid.
* Click confirm.

### Playing Cards
* When it is your turn, click on a card in your hand (bottom of screen) to play it.
* The system highlights valid moves and prevents illegal plays (e.g., failure to follow suit).

### Scoring
* The scoreboard updates in real-time.
* At the end of 13 tricks, the winner is declared based on standard Callbreak scoring rules (meeting your bid awards points; missing it deducts points).

---

## 🔧 Architecture Overview

The app uses a client-server architecture:

* **Frontend:** HTML5/CSS3/JavaScript embedded in the Python file. Uses `socket.io.js` for event listening.
* **Backend:** Flask handles the HTTP requests, while SocketIO manages the bidirectional event loop (e.g., `human_move`, `game_update`).
* **AI Logic:**
    * The `AIManager` class wraps the PyTorch/SB3 models.
    * The `DynamicMCTS` class performs lookahead search, using the selected Neural Network to predict priors and leaf values.

---

## ⚠️ Troubleshooting

* **"Address already in use":**
    If you get this error, another process is using port 5001. Change the port in the last line of `app_v6_algo.py`:

    socketio.run(app, debug=True, port=5002)

* **Models not loading:**
    Check the console output. If you see `! PPO Load Failed`, verify the `.zip` or `.pth` file is in the **exact same folder** as the script.
