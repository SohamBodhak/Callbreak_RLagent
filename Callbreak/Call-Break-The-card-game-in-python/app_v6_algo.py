import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# AI Imports
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C
from callbreak_env import CallbreakEnv
from mcts_agent import ISMCTSAgent, Node

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 1. A3C ARCHITECTURE DEFINITION ---
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.actor(x), self.critic(x)

# --- 2. BIDDING NETWORK ---
class BiddingNet(nn.Module):
    def __init__(self):
        super(BiddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# --- 3. DYNAMIC AI MANAGER ---
class AIManager:
    def __init__(self):
        self.models = {}
        self.current_model_name = "PPO" # Default
        self.simulations = 100
        
        print("Loading AI Models...")
        
        # Load PPO
        try:
            self.models["PPO"] = MaskablePPO.load("ppo_callbreak_zero_pro.zip", device="cpu")
            print(" - PPO Loaded.")
        except: print(" ! PPO Load Failed")

        # Load A2C
        try:
            self.models["A2C"] = A2C.load("a2c_callbreak_final.zip", device="cpu")
            print(" - A2C Loaded.")
        except: print(" ! A2C Load Failed")

        # Load A3C
        try:
            a3c_net = ActorCritic(156, 52)
            a3c_net.load_state_dict(torch.load("a3c_callbreak_final.pth", map_location='cpu'))
            a3c_net.eval()
            self.models["A3C"] = a3c_net
            print(" - A3C Loaded.")
        except: print(" ! A3C Load Failed")

        # Bidding
        self.bidding_model = BiddingNet()
        try:
            self.bidding_model.load_state_dict(torch.load("bidder_network.pth", map_location='cpu'))
            self.bidding_model.eval()
        except: self.bidding_model = None

    def get_action_probs(self, obs, mask):
        """Unified interface to get probabilities from ANY model."""
        model = self.models.get(self.current_model_name)
        if not model: return None # Should trigger random fallback
        
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
        mask_tensor = torch.as_tensor(mask).float().unsqueeze(0)
        
        with torch.no_grad():
            if self.current_model_name == "PPO":
                # PPO Logic
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi)
            
            elif self.current_model_name == "A2C":
                # A2C Logic (SB3)
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi)
                
            elif self.current_model_name == "A3C":
                # A3C Logic (PyTorch)
                logits, _ = model(obs_tensor)
            
            # Common Masking
            HUGE_NEG = -1e8
            logits[mask_tensor == 0] = HUGE_NEG
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
        return probs

    def predict_bid(self, hand_cards):
        if self.bidding_model:
            vec = torch.zeros(52)
            for c in hand_cards: vec[(c.rank-2)*4 + c.suit] = 1.0
            with torch.no_grad():
                bid = int(round(self.bidding_model(vec.unsqueeze(0)).item()))
        else:
            bid = sum(1 for c in hand_cards if c.rank>=13 or c.suit==3)/2 
        return max(1, min(int(bid), 8))

# Initialize Global Managers
env = CallbreakEnv()
ai_manager = AIManager()
current_bids = [0, 0, 0, 0]

# --- 4. MODIFIED MCTS AGENT (Uses AI Manager) ---
class DynamicMCTS:
    def determinize(self, real_hand, history_set, leader_idx, current_trick, tricks_won, player_perspective):
        # ... (Same determinization logic as before) ...
        # (Copying simplified version for brevity, logic identical to previous file)
        import copy, random
        from callbreak_env import Card
        
        all_cards = set(range(52))
        known_cards = set()
        for c in real_hand: known_cards.add((c.rank-2)*4 + c.suit)
        for idx in history_set: known_cards.add(idx)
        for c in current_trick: known_cards.add((c.rank-2)*4 + c.suit)
        
        unknown = list(all_cards - known_cards)
        random.shuffle(unknown)
        
        sim_env = CallbreakEnv()
        sim_env.reset()
        sim_env.hands[0] = [c for c in real_hand]
        
        chunk = len(unknown)//3
        def idx_to_cards(idxs): return [Card((i//4)+2, i%4) for i in idxs]
        
        sim_env.hands[1] = idx_to_cards(unknown[0:chunk])
        sim_env.hands[2] = idx_to_cards(unknown[chunk:2*chunk])
        sim_env.hands[3] = idx_to_cards(unknown[2*chunk:])
        
        sim_env.cards_played_history = copy.deepcopy(history_set)
        sim_env.trick_cards = [c for c in current_trick]
        sim_env.leader_idx = (leader_idx - player_perspective) % 4
        sim_env.tricks_won = [0]*4
        for i in range(4): sim_env.tricks_won[(i-player_perspective)%4] = tricks_won[i]
        sim_env.current_player = 0
        return sim_env

    def select_move(self, real_env):
        root = Node()
        p_idx = real_env.current_player
        
        for _ in range(ai_manager.simulations):
            node = root
            sim_env = self.determinize(real_env.hands[p_idx], real_env.cards_played_history, 
                                     real_env.leader_idx, real_env.trick_cards, 
                                     real_env.tricks_won, p_idx)
            
            # Selection
            path = []
            terminated = False
            while node.children and not terminated:
                action, node = max(node.children.items(), key=lambda i: i[1].ucb_score(node.visit_count))
                _, _, terminated, _, _ = sim_env.step(action)
                path.append(node)
            
            # Expansion
            leaf_val = 0
            if not terminated:
                probs = ai_manager.get_action_probs(sim_env._get_obs(), sim_env._get_valid_mask())
                
                # Check value estimate logic (Using PPO's value net for all for simplicity, or heuristic)
                # Since A2C/A3C value nets are different, we use a simple ROLLOUT or Heuristic here
                # to avoid complexity. Using simple Win/Loss heuristic at leaf.
                leaf_val = 0.5 # Neutral prior for non-terminal
                
                mask = sim_env._get_valid_mask()
                valid = [i for i,m in enumerate(mask) if m==1]
                for idx in valid:
                    prior = probs[idx] if probs is not None else 1.0/len(valid)
                    node.children[idx] = Node(parent=node, prior=prior)
            else:
                leaf_val = 1.0 if sim_env.tricks_won[0] > sim_env.tricks_won[1] else 0.0
            
            # Backprop
            node.visit_count += 1
            node.value_sum += leaf_val
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += leaf_val
                
        if not root.children: return 0 # Fallback
        return max(root.children.items(), key=lambda i: i[1].visit_count)[0]

mcts_bot = DynamicMCTS()

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Callbreak AI Arena</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { background-color: #222; color: white; font-family: 'Segoe UI', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; overflow: hidden; }
        #game-table { width: 900px; height: 600px; background-color: #2e8b57; border: 15px solid #5c3a21; border-radius: 150px; position: relative; box-shadow: inset 0 0 50px rgba(0,0,0,0.5); }
        .player-zone { position: absolute; display: flex; justify-content: center; align-items: center; }
        #top-bot { top: 20px; left: 50%; transform: translateX(-50%); }
        #left-bot { left: -40px; top: 50%; transform: translateY(-50%) rotate(90deg); }
        #right-bot { right: -40px; top: 50%; transform: translateY(-50%) rotate(-90deg); }
        #human-zone { bottom: 30px; left: 50%; transform: translateX(-50%); }
        .card { width: 70px; height: 100px; background: white; border-radius: 6px; margin-left: -35px; box-shadow: 2px 2px 5px rgba(0,0,0,0.4); display: flex; flex-direction: column; justify-content: space-between; padding: 5px; box-sizing: border-box; color: black; font-weight: bold; font-size: 20px; cursor: pointer; transition: transform 0.2s; user-select: none; }
        .card:hover { transform: translateY(-20px); z-index: 10; }
        .card.red { color: #d00000; }
        .card-back { background: #b22222; border: 2px solid white; }
        #trick-area { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
        .trick-card { position: absolute; left: 50%; top: 50%; transition: all 0.5s ease-out; transform: translate(-50%, -50%); margin-left: 0; pointer-events: auto; }
        .fly-anim { opacity: 0; transform: translate(-50%, -50%) scale(0.2) !important; }
        .fly-to-0 { top: 100% !important; left: 50% !important; }
        .fly-to-1 { top: 50% !important; left: 100% !important; }
        .fly-to-2 { top: 0% !important; left: 50% !important; }
        .fly-to-3 { top: 50% !important; left: 0% !important; }
        .p0 { transform: translate(-50%, 10%); } .p1 { transform: translate(60%, -50%) rotate(-15deg); } .p2 { transform: translate(-50%, -110%); } .p3 { transform: translate(-160%, -50%) rotate(15deg); }
        .overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); border-radius: 135px; display: none; flex-direction: column; justify-content: center; align-items: center; z-index: 100; }
        #controls-area { position: absolute; top: 60%; left: 50%; transform: translate(-50%, -50%); display: flex; flex-direction: column; align-items: center; gap: 10px; background: rgba(0,0,0,0.5); padding: 20px; border-radius: 15px; }
        select { padding: 10px; border-radius: 5px; background: #333; color: white; font-size: 16px; border: 1px solid #555; width: 200px; }
        label { font-weight: bold; margin-bottom: 5px; display: block; text-align: left; width: 100%; }
        .btn { padding: 15px 40px; font-size: 20px; border: none; border-radius: 50px; cursor: pointer; font-weight: bold; margin-top: 10px; }
        .btn-green { background: #2e8b57; color: white; }
        .btn-red { background: #d00000; color: white; }
        #mini-scoreboard { position:absolute; top:10px; left:20px; background:rgba(0,0,0,0.6); padding:15px; border-radius:8px; }
        .bid-info { color: gold; font-size: 14px; margin-left: 5px; }
        #status-msg { position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%); font-size: 28px; font-weight: bold; text-shadow: 2px 2px 4px black; pointer-events: none;}
        table { border-collapse: collapse; width: 60%; font-size: 24px; color: white; margin-bottom: 30px; }
        th, td { border-bottom: 1px solid #555; padding: 10px; text-align: center; }
        .winner-row { color: #2e8b57; font-weight: bold; }
    </style>
</head>
<body>
    <div id="game-table">
        <div id="status-msg">Welcome to Callbreak AI Arena</div>
        
        <div id="controls-area">
            <div>
                <label>Bot Intelligence (Model):</label>
                <select id="model-select">
                    <option value="PPO" selected>PPO (Proximal Policy Opt)</option>
                    <option value="A2C">A2C (Advantage Actor Critic)</option>
                    <option value="A3C">A3C (Asynchronous)</option>
                </select>
            </div>
            
            <div style="margin-top: 10px;">
                <label>Simulation Depth (MCTS):</label>
                <select id="difficulty">
                    <option value="25">Novice (25 Sims)</option>
                    <option value="100" selected>Intermediate (100 Sims)</option>
                    <option value="400">Expert (400 Sims)</option>
                </select>
            </div>
            
            <button id="start-btn" class="btn btn-green" onclick="startGame()">Start Game</button>
        </div>
        
        <div id="mini-scoreboard">
            <div>You: <span id="s0">0</span> <span id="b0" class="bid-info"></span></div>
            <div>Right: <span id="s1">0</span> <span id="b1" class="bid-info"></span></div>
            <div>Top: <span id="s2">0</span> <span id="b2" class="bid-info"></span></div>
            <div>Left: <span id="s3">0</span> <span id="b3" class="bid-info"></span></div>
        </div>
        
        <div id="top-bot" class="player-zone"></div>
        <div id="left-bot" class="player-zone"></div>
        <div id="right-bot" class="player-zone"></div>
        <div id="human-zone" class="player-zone"></div>
        <div id="trick-area"></div>

        <div id="bidding-modal" class="overlay">
            <h2 style="color:gold">Place Your Bid</h2>
            <div style="font-size: 60px; font-weight: bold; color: gold; margin: 20px 0;" id="bid-val-display">3</div>
            <input type="range" min="1" max="13" value="3" style="width: 300px;" id="bid-slider" oninput="document.getElementById('bid-val-display').innerText = this.value">
            <br><br>
            <button class="btn btn-green" onclick="submitBid()">Confirm Bid</button>
        </div>

        <div id="game-over-screen" class="overlay">
            <h1 id="winner-title" style="color:gold; font-size:48px; margin-bottom:20px;">YOU WON!</h1>
            <table id="result-table">
                <thead><tr><th>Player</th><th>Bid</th><th>Won</th><th>Score</th></tr></thead>
                <tbody id="result-body"></tbody>
            </table>
            <button class="btn btn-red" onclick="location.reload()">New Game</button>
        </div>
    </div>

    <script>
        const socket = io();
        const suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'};
        let isAnimating = false;

        function startGame() {
            const diff = document.getElementById('difficulty').value;
            const model = document.getElementById('model-select').value;
            
            document.getElementById('controls-area').style.display = 'none';
            socket.emit('start_game', { difficulty: parseInt(diff), model: model });
        }

        function submitBid() {
            const val = document.getElementById('bid-slider').value;
            document.getElementById('bidding-modal').style.display = 'none';
            socket.emit('human_bid', {bid: parseInt(val)});
        }

        socket.on('status_update', data => {
            document.getElementById('status-msg').innerText = data.msg;
        });

        socket.on('request_bid', () => {
            document.getElementById('status-msg').innerText = "";
            document.getElementById('bidding-modal').style.display = 'flex';
        });

        socket.on('trick_won', data => {
            isAnimating = true;
            const winnerIdx = data.winner;
            const cards = document.querySelectorAll('.trick-card');
            setTimeout(() => {
                cards.forEach(card => {
                    card.classList.add('fly-to-' + winnerIdx);
                    card.classList.add('fly-anim');
                });
            }, 50);
            setTimeout(() => {
                isAnimating = false;
                socket.emit('animation_done'); 
            }, 800);
        });

        socket.on('game_update', state => {
            state.scores.forEach((s, i) => document.getElementById('s'+i).innerText = s);
            state.bids.forEach((b, i) => document.getElementById('b'+i).innerText = b > 0 ? `/ ${b}` : '');
            if (!state.game_over) renderGame(state);
            else showGameOver(state);
        });

        function renderGame(state) {
            document.getElementById('human-zone').innerHTML = '';
            const hDiv = document.getElementById('human-zone');
            state.human_hand.forEach(c => {
                const el = document.createElement('div');
                el.className = `card ${c.color}`;
                el.innerHTML = `<div>${c.rank}</div><div>${suits[c.suit]}</div>`;
                el.onclick = () => socket.emit('human_move', {card_idx: c.raw_idx});
                hDiv.appendChild(el);
            });
            drawBot('top-bot', state.bot_counts[1]);
            drawBot('right-bot', state.bot_counts[0]);
            drawBot('left-bot', state.bot_counts[2]);

            if (isAnimating) return;
            const tDiv = document.getElementById('trick-area');
            tDiv.innerHTML = '';
            state.trick.forEach((c, idx) => {
                if(c) {
                    const el = document.createElement('div');
                    el.className = `card trick-card p${idx} ${c.color}`;
                    el.innerHTML = `<div>${c.rank}</div><div>${suits[c.suit]}</div>`;
                    tDiv.appendChild(el);
                }
            });
        }

        function showGameOver(state) {
            document.getElementById('bidding-modal').style.display = 'none';
            const names = ["You", "Right Bot", "Top Bot", "Left Bot"];
            const tbody = document.getElementById('result-body');
            tbody.innerHTML = '';
            let finalResults = [];
            for(let i=0; i<4; i++) {
                let bid = state.bids[i];
                let won = state.scores[i];
                let points = (won >= bid) ? bid + (won - bid) * 0.1 : -bid;
                finalResults.push({name: names[i], bid: bid, won: won, points: points});
            }
            finalResults.sort((a, b) => b.points - a.points);
            finalResults.forEach((r, i) => {
                const tr = document.createElement('tr');
                if (i===0) tr.classList.add('winner-row');
                tr.innerHTML = `<td>${r.name}</td><td>${r.bid}</td><td>${r.won}</td><td>${r.points.toFixed(1)}</td>`;
                tbody.appendChild(tr);
            });
            const isWin = finalResults[0].name === "You";
            const title = document.getElementById('winner-title');
            title.innerText = isWin ? "VICTORY!" : "DEFEAT";
            title.style.color = isWin ? "gold" : "red";
            document.getElementById('game-over-screen').style.display = 'flex';
        }

        function drawBot(id, count) {
            const div = document.getElementById(id);
            div.innerHTML = '';
            for(let i=0; i<count; i++) {
                const el = document.createElement('div'); el.className = 'card card-back'; div.appendChild(el);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def get_state_json():
    if not env: return {}
    def s_card(c):
        if not c: return None
        r_map = {11:'J', 12:'Q', 13:'K', 14:'A'}
        return { 'rank': r_map.get(c.rank, str(c.rank)), 'suit': ['C','D','H','S'][c.suit], 'color': 'red' if c.suit in [1,2] else 'black', 'raw_idx': (c.rank-2)*4 + c.suit }
    trick_vis = [None]*4
    if env.trick_cards:
        start = env.leader_idx
        for i, c in enumerate(env.trick_cards):
            trick_vis[(start+i)%4] = s_card(c)
    winner = -1
    if sum(env.tricks_won) == 13: winner = int(np.argmax(env.tricks_won))
    return {
        'human_hand': [s_card(c) for c in sorted(env.hands[0], key=lambda x:(x.suit, x.rank))],
        'bot_counts': [len(env.hands[1]), len(env.hands[2]), len(env.hands[3])],
        'trick': trick_vis,
        'scores': [int(x) for x in env.tricks_won],
        'bids': current_bids,
        'game_over': sum(env.tricks_won) == 13,
        'winner': winner
    }

def process_trick_result():
    w_rel = env._get_winner_idx(env.trick_cards)
    w_abs = (env.leader_idx + w_rel) % 4
    socketio.emit('trick_won', {'winner': w_abs})
    socketio.sleep(1.0)
    env.tricks_won[w_abs] += 1
    env.leader_idx = w_abs
    env.current_player = w_abs
    for c in env.trick_cards: env.cards_played_history.add((c.rank-2)*4 + c.suit)
    env.trick_cards = []
    socketio.emit('game_update', get_state_json())
    if env.current_player != 0 and sum(env.tricks_won) < 13: bot_play_loop()
    else: socketio.emit('status_update', {'msg': "Your Turn!"})

def bot_play_loop():
    global env
    while env and sum(env.tricks_won) < 13:
        if env.current_player == 0:
            socketio.emit('status_update', {'msg': "Your Turn!"})
            break
        socketio.sleep(0.5)
        socketio.emit('status_update', {'msg': f"Bot {env.current_player} ({ai_manager.current_model_name}) Thinking..."})
        
        action = mcts_bot.select_move(env)
        
        r, s = (action//4)+2, action%4
        c_obj = None
        for c in env.hands[env.current_player]:
            if c.rank==r and c.suit==s: c_obj=c; break
        if not c_obj and env.hands[env.current_player]: c_obj = env.hands[env.current_player][0]

        if c_obj:
            env.hands[env.current_player].remove(c_obj)
            env.trick_cards.append(c_obj)
            socketio.emit('game_update', get_state_json())
            if len(env.trick_cards) == 4:
                socketio.start_background_task(process_trick_result)
                break
            else:
                env.current_player = (env.current_player + 1) % 4

@socketio.on('start_game')
def handle_start(data):
    global current_bids
    if data:
        if 'difficulty' in data: ai_manager.simulations = data['difficulty']
        if 'model' in data: ai_manager.current_model_name = data['model']
        
    print(f"Game Started | Model: {ai_manager.current_model_name} | Sims: {ai_manager.simulations}")
    env.reset()
    current_bids = [0, 0, 0, 0]
    emit('game_update', get_state_json())
    emit('request_bid')

@socketio.on('human_bid')
def handle_bid(data):
    global current_bids
    human_bid = data['bid']
    current_bids[0] = human_bid
    for i in range(1, 4): current_bids[i] = ai_manager.predict_bid(env.hands[i])
    emit('status_update', {'msg': "Bidding Complete!"})
    socketio.sleep(1)
    emit('game_update', get_state_json())
    if env.current_player != 0: socketio.start_background_task(bot_play_loop)
    else: emit('status_update', {'msg': "Your Turn!"})

@socketio.on('human_move')
def handle_move(data):
    if env.current_player != 0: return
    idx = data['card_idx']
    mask = env._get_valid_mask()
    if mask[idx] == 0:
        emit('error_msg', {'msg': "Invalid Move!"})
        return
    r, s = (idx//4)+2, idx%4
    c_obj = None
    for c in env.hands[0]:
        if c.rank==r and c.suit==s: c_obj=c; break
    if c_obj:
        env.hands[0].remove(c_obj)
        env.trick_cards.append(c_obj)
        emit('game_update', get_state_json())
        if len(env.trick_cards) == 4: socketio.start_background_task(process_trick_result)
        else:
            env.current_player = (env.current_player + 1) % 4
            if env.current_player != 0: socketio.start_background_task(bot_play_loop)
            else: emit('status_update', {'msg': "Your Turn!"})

if __name__ == '__main__':
    print("Running on http://127.0.0.1:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)