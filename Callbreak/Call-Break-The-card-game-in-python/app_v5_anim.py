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

# --- 1. DEFINE BIDDING NETWORK ---
class BiddingNet(nn.Module):
    def __init__(self):
        super(BiddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# --- GLOBAL STATE ---
print("Initializing Game...")
env = CallbreakEnv()
env.reset()

print("Loading AI Brain...")
agent = ISMCTSAgent()
bidding_model = BiddingNet()
try:
    bidding_model.load_state_dict(torch.load("bidder_network.pth", map_location=torch.device('cpu')))
    bidding_model.eval()
except:
    bidding_model = None

current_bids = [0, 0, 0, 0]
CURRENT_SIMULATIONS = 100 

def predict_bot_bid(hand_cards):
    if bidding_model:
        vec = torch.zeros(52)
        for c in hand_cards: vec[(c.rank-2)*4 + c.suit] = 1.0
        with torch.no_grad():
            bid = int(round(bidding_model(vec.unsqueeze(0)).item()))
    else:
        bid = sum(1 for c in hand_cards if c.rank>=13 or c.suit==3)/2 
    return max(1, min(int(bid), 8))

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
            display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; overflow: hidden;
        }
        #game-table {
            width: 900px; height: 600px; background-color: #2e8b57;
            border: 15px solid #5c3a21; border-radius: 150px;
            position: relative; box-shadow: inset 0 0 50px rgba(0,0,0,0.5);
        }
        
        /* ZONES */
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

        /* TRICK AREA - FULL TABLE OVERLAY */
        #trick-area { 
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
            pointer-events: none; /* Let clicks pass through to controls */
        }
        .trick-card {
            position: absolute; left: 50%; top: 50%; transition: all 0.5s ease-out;
            transform: translate(-50%, -50%); margin-left: 0;
            pointer-events: auto;
        }
        
        /* CARD POSITIONS ON TABLE */
        .p0 { transform: translate(-50%, 10%); }
        .p1 { transform: translate(60%, -50%) rotate(-15deg); }
        .p2 { transform: translate(-50%, -110%); }
        .p3 { transform: translate(-160%, -50%) rotate(15deg); }

        /* FLIGHT ANIMATIONS */
        .fly-anim { opacity: 0; transform: translate(-50%, -50%) scale(0.2) !important; }
        
        /* DESTINATIONS (Where cards fly to) */
        .fly-to-0 { top: 100% !important; left: 50% !important; } /* Human */
        .fly-to-1 { top: 50% !important; left: 100% !important; } /* Right */
        .fly-to-2 { top: 0% !important;   left: 50% !important; } /* Top */
        .fly-to-3 { top: 50% !important;  left: 0% !important; }   /* Left */

        /* OVERLAYS */
        .overlay {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85); border-radius: 135px;
            display: none; flex-direction: column; justify-content: center; align-items: center;
            z-index: 100;
        }
        
        #controls-area {
            position: absolute; top: 60%; left: 50%; transform: translate(-50%, -50%);
            display: flex; flex-direction: column; align-items: center; gap: 10px;
        }
        select { padding: 10px; border-radius: 5px; background: #333; color: white; font-size: 16px; border: 1px solid #555; }
        #bid-val-display { font-size: 60px; font-weight: bold; color: gold; margin: 20px 0; }
        input[type=range] { width: 300px; }
        
        .btn { padding: 15px 40px; font-size: 20px; border: none; border-radius: 50px; cursor: pointer; font-weight: bold; }
        .btn-green { background: #2e8b57; color: white; }
        .btn-red { background: #d00000; color: white; }
        
        #mini-scoreboard { position:absolute; top:10px; left:20px; background:rgba(0,0,0,0.6); padding:15px; border-radius:8px; }
        .bid-info { color: gold; font-size: 14px; margin-left: 5px; }
        #status-msg { position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%); font-size: 28px; font-weight: bold; text-shadow: 2px 2px 4px black; }

        table { border-collapse: collapse; width: 60%; font-size: 24px; color: white; margin-bottom: 30px; }
        th, td { border-bottom: 1px solid #555; padding: 10px; text-align: center; }
        .winner-row { color: #2e8b57; font-weight: bold; }
    </style>
</head>
<body>
    <div id="game-table">
        <div id="status-msg">Welcome to Callbreak</div>
        
        <div id="controls-area">
            <button id="start-btn" class="btn btn-green" onclick="startGame()">Deal Cards</button>
            <div id="diff-selector">
                <label>Difficulty:</label>
                <select id="difficulty">
                    <option value="25">Novice</option>
                    <option value="100" selected>Intermediate</option>
                    <option value="400">Expert</option>
                </select>
            </div>
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
        let isAnimating = false;

        function startGame() {
            const diff = document.getElementById('difficulty').value;
            document.getElementById('controls-area').style.display = 'none';
            socket.emit('start_game', { difficulty: parseInt(diff) });
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

        // --- ANIMATION LOGIC ---
        socket.on('trick_won', data => {
            isAnimating = true;
            const winnerIdx = data.winner;
            const cards = document.querySelectorAll('.trick-card');
            
            // 1. Force a small delay to ensure cards are rendered before flying
            setTimeout(() => {
                cards.forEach(card => {
                    // Add flight class
                    card.classList.add('fly-to-' + winnerIdx);
                    card.classList.add('fly-anim'); // Adds fade + shrink
                });
            }, 50);

            // 2. Unlock UI after animation finishes
            setTimeout(() => {
                isAnimating = false;
                // Ask server for the clean table state if not already received
                socket.emit('animation_done'); 
            }, 800);
        });

        socket.on('game_update', state => {
            state.scores.forEach((s, i) => document.getElementById('s'+i).innerText = s);
            state.bids.forEach((b, i) => document.getElementById('b'+i).innerText = b > 0 ? `/ ${b}` : '');

            if (!state.game_over) {
                renderGame(state);
            } else {
                showGameOver(state);
            }
        });

        function renderGame(state) {
            // Draw Hand (Always safe)
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

            // Trick Area Logic:
            // If we are currently animating the flight, DO NOT redraw the trick area (it would reset card positions)
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
    """Background task to handle animation and state update"""
    # 1. Notify Frontend to animate (Frontend has cards, it just flies them)
    w_rel = env._get_winner_idx(env.trick_cards)
    w_abs = (env.leader_idx + w_rel) % 4
    
    # Send the signal to start animation
    socketio.emit('trick_won', {'winner': w_abs})
    
    # 2. Server waits for animation duration (Non-blocking sleep in bg thread)
    socketio.sleep(1.0)
    
    # 3. Update Logical State
    env.tricks_won[w_abs] += 1
    env.leader_idx = w_abs
    env.current_player = w_abs
    for c in env.trick_cards:
        env.cards_played_history.add((c.rank-2)*4 + c.suit)
    env.trick_cards = []
    
    # 4. Send Clean State
    socketio.emit('game_update', get_state_json())

    # 5. Continue Game Loop if needed
    if env.current_player != 0 and sum(env.tricks_won) < 13:
        bot_play_loop()
    else:
        socketio.emit('status_update', {'msg': "Your Turn!"})


def bot_play_loop():
    global env
    while env and sum(env.tricks_won) < 13:
        if env.current_player == 0:
            socketio.emit('status_update', {'msg': "Your Turn!"})
            break
        
        socketio.sleep(0.5)
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
            
            socketio.emit('game_update', get_state_json())

            if len(env.trick_cards) == 4:
                # Trick Over! Break loop and handle animation in separate sequence
                process_trick_result()
                break # Exit this loop, process_trick_result will restart it if needed
            else:
                env.current_player = (env.current_player + 1) % 4

@socketio.on('start_game')
def handle_start(data):
    global current_bids, CURRENT_SIMULATIONS
    if data and 'difficulty' in data:
        CURRENT_SIMULATIONS = data['difficulty']
    env.reset()
    current_bids = [0, 0, 0, 0]
    emit('game_update', get_state_json())
    emit('request_bid')

@socketio.on('human_bid')
def handle_bid(data):
    global current_bids
    human_bid = data['bid']
    current_bids[0] = human_bid
    for i in range(1, 4):
        current_bids[i] = predict_bot_bid(env.hands[i])
    emit('status_update', {'msg': "Bidding Complete!"})
    socketio.sleep(1)
    emit('game_update', get_state_json())
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
        
        # Immediate update so user sees their card played
        emit('game_update', get_state_json())

        if len(env.trick_cards) == 4:
            # Start background task for animation to avoid blocking main thread
            socketio.start_background_task(process_trick_result)
        else:
            env.current_player = (env.current_player + 1) % 4
            if env.current_player != 0:
                socketio.start_background_task(bot_play_loop)
            else:
                emit('status_update', {'msg': "Your Turn!"})

if __name__ == '__main__':
    print("Running on http://127.0.0.1:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)