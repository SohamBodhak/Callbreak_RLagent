import time
import threading
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# --- IMPORT AI ---
from callbreak_env import CallbreakEnv
from mcts_agent import ISMCTSAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 1. DEFINE BIDDING NETWORK ARCHITECTURE ---
class BiddingNet(nn.Module):
    def __init__(self):
        super(BiddingNet, self).__init__()
        # Based on standard architecture for this type of problem
        # 52 Inputs -> Hidden -> Output (Expected Tricks)
        self.net = nn.Sequential(
            nn.Linear(52, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- GLOBAL STATE ---
print("Initializing Game...")
env = CallbreakEnv()
env.reset()

print("Loading Playing Brain (MCTS)...")
agent = ISMCTSAgent()

print("Loading Bidding Brain...")
bidding_model = BiddingNet()
try:
    # Load the weights
    bidding_model.load_state_dict(torch.load("bidder_network.pth", map_location=torch.device('cpu')))
    bidding_model.eval() # Set to evaluation mode
    print("Bidding Network Loaded Successfully!")
except Exception as e:
    print(f"Warning: Could not load bidder_network.pth ({e}). using Heuristic fallback.")
    bidding_model = None

# Store current bids [Human, Right, Top, Left]
current_bids = [0, 0, 0, 0]

# --- HELPER: BOT BIDDING LOGIC ---
def predict_bot_bid(hand_cards):
    """Predicts a bid for a bot based on their hand."""
    if bidding_model:
        # 1. Convert Hand to One-Hot Tensor (Size 52)
        input_vec = torch.zeros(52)
        for card in hand_cards:
            idx = (card.rank - 2) * 4 + card.suit
            input_vec[idx] = 1.0
        
        # 2. Run Inference
        with torch.no_grad():
            prediction = bidding_model(input_vec.unsqueeze(0))
            bid = int(round(prediction.item()))
    else:
        # Fallback Heuristic (Count High Cards)
        bid = 0
        for c in hand_cards:
            if c.rank >= 13: bid += 1 # Kings and Aces
            if c.suit == 3: bid += 0.5 # Spades are valuable
        bid = int(round(bid))

    # Clamp bid between 1 and 8 (Safety)
    return max(1, min(bid, 8))

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Callbreak Pro</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            background-color: #222; color: white; font-family: 'Segoe UI', sans-serif;
            display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;
        }
        #game-table {
            width: 900px; height: 600px; background-color: #2e8b57;
            border: 15px solid #5c3a21; border-radius: 150px;
            position: relative; box-shadow: inset 0 0 50px rgba(0,0,0,0.5);
        }
        
        /* PLAYER ZONES */
        .player-zone { position: absolute; display: flex; justify-content: center; align-items: center; }
        #top-bot { top: 20px; left: 50%; transform: translateX(-50%); }
        #left-bot { left: -40px; top: 50%; transform: translateY(-50%) rotate(90deg); }
        #right-bot { right: -40px; top: 50%; transform: translateY(-50%) rotate(-90deg); }
        #human-zone { bottom: 30px; left: 50%; transform: translateX(-50%); }

        /* CARDS */
        .card {
            width: 70px; height: 100px; background: white; border-radius: 6px;
            margin-left: -35px; box-shadow: 2px 2px 5px rgba(0,0,0,0.4);
            display: flex; flex-direction: column; justify-content: space-between;
            padding: 5px; box-sizing: border-box; color: black; font-weight: bold; font-size: 20px;
            cursor: pointer; transition: transform 0.2s; user-select: none;
        }
        .card:hover { transform: translateY(-20px); z-index: 10; }
        .card.red { color: #d00000; }
        .card-back { background: #b22222; border: 2px solid white; }

        /* TRICK AREA */
        #trick-area { position: absolute; top: 50%; left: 50%; width: 10px; height: 10px; }
        .trick-card {
            position: absolute; left: 50%; top: 50%; transition: all 0.4s ease;
            transform: translate(-50%, -50%); margin-left: 0;
        }
        .p0 { transform: translate(-50%, 10%); }
        .p1 { transform: translate(60%, -50%) rotate(-15deg); }
        .p2 { transform: translate(-50%, -110%); }
        .p3 { transform: translate(-160%, -50%) rotate(15deg); }

        /* OVERLAYS (Game Over & Bidding) */
        .overlay {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85); border-radius: 135px;
            display: none; flex-direction: column; justify-content: center; align-items: center;
            z-index: 100;
        }
        
        /* BIDDING UI SPECIFICS */
        #bidding-modal h2 { font-size: 36px; color: gold; margin-bottom: 10px; }
        #bid-val-display { font-size: 60px; font-weight: bold; color: white; margin: 20px 0; }
        input[type=range] {
            width: 300px; -webkit-appearance: none; background: transparent;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; height: 30px; width: 30px; border-radius: 50%;
            background: gold; cursor: pointer; margin-top: -12px;
        }
        input[type=range]::-webkit-slider-runnable-track {
            width: 100%; height: 8px; cursor: pointer; background: #555; border-radius: 5px;
        }

        /* SCOREBOARD */
        #mini-scoreboard {
            position:absolute; top:10px; left:20px; 
            background:rgba(0,0,0,0.6); padding:15px; border-radius:8px; font-size: 16px;
        }
        .bid-info { color: gold; font-size: 14px; margin-left: 5px; }

        .btn { padding: 15px 40px; font-size: 20px; border: none; border-radius: 50px; cursor: pointer; font-weight: bold; }
        .btn-green { background: #2e8b57; color: white; }
        .btn-red { background: #d00000; color: white; }
        #status-msg { position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%); font-size: 28px; font-weight: bold; text-shadow: 2px 2px 4px black; pointer-events: none; }
    
        /* Result Table */
        table { border-collapse: collapse; width: 60%; font-size: 24px; color: white; margin-bottom: 30px; }
        th, td { border-bottom: 1px solid #555; padding: 10px; text-align: center; }
        .winner-row { color: #2e8b57; font-weight: bold; }
    </style>
</head>
<body>
    <div id="game-table">
        <div id="status-msg">Connecting...</div>
        <button id="start-btn" class="btn btn-green" onclick="startGame()" style="position:absolute; top:60%; left:50%; transform:translate(-50%, -50%);">Deal Cards</button>
        
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
            <h2>Place Your Bid</h2>
            <div id="bid-val-display">3</div>
            <input type="range" min="1" max="13" value="3" id="bid-slider" oninput="document.getElementById('bid-val-display').innerText = this.value">
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

        function startGame() {
            document.getElementById('start-btn').style.display = 'none';
            socket.emit('start_game');
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

        socket.on('game_update', state => {
            // Update Scores & Bids
            state.scores.forEach((s, i) => document.getElementById('s'+i).innerText = s);
            state.bids.forEach((b, i) => document.getElementById('b'+i).innerText = b > 0 ? `/ ${b}` : '');

            if (!state.game_over) {
                renderGame(state);
            } else {
                showGameOver(state);
            }
        });

        function renderGame(state) {
            // Clear
            ['human-zone', 'trick-area', 'top-bot', 'left-bot', 'right-bot'].forEach(id => document.getElementById(id).innerHTML = '');

            // Human Hand
            const hDiv = document.getElementById('human-zone');
            state.human_hand.forEach(c => {
                const el = document.createElement('div');
                el.className = `card ${c.color}`;
                el.innerHTML = `<div>${c.rank}</div><div>${suits[c.suit]}</div>`;
                el.onclick = () => socket.emit('human_move', {card_idx: c.raw_idx});
                hDiv.appendChild(el);
            });

            // Bots
            drawBot('top-bot', state.bot_counts[1]);
            drawBot('right-bot', state.bot_counts[0]);
            drawBot('left-bot', state.bot_counts[2]);

            // Trick
            const tDiv = document.getElementById('trick-area');
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
            document.getElementById('bidding-modal').style.display = 'none'; // Safety
            
            const names = ["You", "Right Bot", "Top Bot", "Left Bot"];
            const tbody = document.getElementById('result-body');
            tbody.innerHTML = '';
            
            // Calculate Callbreak Scores (Logic: Won >= Bid ? Bid + 0.1*(Won-Bid) : -Bid)
            let finalResults = [];
            for(let i=0; i<4; i++) {
                let bid = state.bids[i];
                let won = state.scores[i];
                let points = 0;
                if (won >= bid) points = bid + (won - bid) * 0.1;
                else points = -bid;
                finalResults.push({name: names[i], bid: bid, won: won, points: points});
            }
            
            // Sort by Points
            finalResults.sort((a, b) => b.points - a.points);

            finalResults.forEach((r, i) => {
                const tr = document.createElement('tr');
                if (i===0) tr.classList.add('winner-row');
                tr.innerHTML = `<td>${r.name}</td><td>${r.bid}</td><td>${r.won}</td><td>${r.points.toFixed(1)}</td>`;
                tbody.appendChild(tr);
            });
            
            document.getElementById('winner-title').innerText = finalResults[0].name === "You" ? "VICTORY!" : "DEFEAT";
            document.getElementById('winner-title').style.color = finalResults[0].name === "You" ? "gold" : "red";
            document.getElementById('game-over-screen').style.display = 'flex';
        }

        function drawBot(id, count) {
            const div = document.getElementById(id);
            for(let i=0; i<count; i++) {
                const el = document.createElement('div');
                el.className = 'card card-back';
                div.appendChild(el);
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
        return {
            'rank': r_map.get(c.rank, str(c.rank)),
            'suit': ['C','D','H','S'][c.suit],
            'color': 'red' if c.suit in [1,2] else 'black',
            'raw_idx': (c.rank-2)*4 + c.suit
        }

    trick_vis = [None]*4
    if env.trick_cards:
        start = env.leader_idx
        for i, c in enumerate(env.trick_cards):
            trick_vis[(start+i)%4] = s_card(c)

    winner = -1
    if sum(env.tricks_won) == 13:
        winner = int(np.argmax(env.tricks_won))

    return {
        'human_hand': [s_card(c) for c in sorted(env.hands[0], key=lambda x:(x.suit, x.rank))],
        'bot_counts': [len(env.hands[1]), len(env.hands[2]), len(env.hands[3])],
        'trick': trick_vis,
        'scores': [int(x) for x in env.tricks_won],
        'bids': current_bids, # Include bids in update
        'game_over': sum(env.tricks_won) == 13,
        'winner': winner
    }

def bot_play_loop():
    global env
    while env and sum(env.tricks_won) < 13:
        if env.current_player == 0:
            socketio.emit('status_update', {'msg': "Your Turn!"})
            break
        
        socketio.sleep(0.7)
        socketio.emit('status_update', {'msg': f"Bot {env.current_player} Thinking..."})
        
        action = agent.select_move(env)
        
        r, s = (action//4)+2, action%4
        c_obj = None
        for c in env.hands[env.current_player]:
            if c.rank==r and c.suit==s: c_obj=c; break
        if not c_obj and env.hands[env.current_player]: c_obj = env.hands[env.current_player][0]

        if c_obj:
            env.hands[env.current_player].remove(c_obj)
            env.trick_cards.append(c_obj)
            
            if len(env.trick_cards) == 4:
                socketio.emit('game_update', get_state_json())
                socketio.sleep(1.5)
                w_rel = env._get_winner_idx(env.trick_cards)
                w_abs = (env.leader_idx + w_rel) % 4
                env.tricks_won[w_abs] += 1
                env.leader_idx = w_abs
                env.current_player = w_abs
                for c in env.trick_cards:
                    env.cards_played_history.add((c.rank-2)*4 + c.suit)
                env.trick_cards = []
            else:
                env.current_player = (env.current_player + 1) % 4

            socketio.emit('game_update', get_state_json())

@socketio.on('start_game')
def handle_start():
    global current_bids
    env.reset()
    current_bids = [0, 0, 0, 0] # Reset bids
    
    # Show Deal state
    emit('game_update', get_state_json())
    
    # Request Bid from Human
    emit('request_bid')

@socketio.on('human_bid')
def handle_bid(data):
    global current_bids
    human_bid = data['bid']
    current_bids[0] = human_bid
    
    # Generate Bot Bids
    for i in range(1, 4):
        bot_bid = predict_bot_bid(env.hands[i])
        current_bids[i] = bot_bid
        
    emit('status_update', {'msg': "Bidding Complete! Game Starting..."})
    socketio.sleep(1)
    
    emit('game_update', get_state_json())
    
    # Determine who starts (usually dealer+1 or winner of previous, here strictly random/set in env)
    # If it's a bot's turn, start the loop
    if env.current_player != 0:
        socketio.start_background_task(bot_play_loop)
    else:
        emit('status_update', {'msg': "Your Turn!"})

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
        
        if len(env.trick_cards) == 4:
            socketio.emit('game_update', get_state_json())
            socketio.sleep(1.5)
            w_rel = env._get_winner_idx(env.trick_cards)
            w_abs = (env.leader_idx + w_rel) % 4
            env.tricks_won[w_abs] += 1
            env.leader_idx = w_abs
            env.current_player = w_abs
            for c in env.trick_cards:
                env.cards_played_history.add((c.rank-2)*4 + c.suit)
            env.trick_cards = []
        else:
            env.current_player = (env.current_player + 1) % 4
        
        emit('game_update', get_state_json())
        
        if env.current_player != 0:
            socketio.start_background_task(bot_play_loop)
        else:
             emit('status_update', {'msg': "Your Turn!"})

if __name__ == '__main__':
    print("Running on http://127.0.0.1:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)