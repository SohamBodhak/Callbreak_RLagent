import time
import threading
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# --- IMPORT AI ---
from callbreak_env import CallbreakEnv
from mcts_agent_rotate import ISMCTSAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- GLOBAL STATE ---
print("Initializing Game...")
env = CallbreakEnv()
env.reset()

print("Loading AI Brain...")
agent = ISMCTSAgent()
print("System Ready.")

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
        
        .player-zone { position: absolute; display: flex; justify-content: center; align-items: center; }
        #top-bot { top: 20px; left: 50%; transform: translateX(-50%); }
        #left-bot { left: -40px; top: 50%; transform: translateY(-50%) rotate(90deg); }
        #right-bot { right: -40px; top: 50%; transform: translateY(-50%) rotate(-90deg); }
        #human-zone { bottom: 30px; left: 50%; transform: translateX(-50%); }

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

        #trick-area { position: absolute; top: 50%; left: 50%; width: 10px; height: 10px; }
        .trick-card {
            position: absolute; left: 50%; top: 50%; transition: all 0.4s ease;
            transform: translate(-50%, -50%); margin-left: 0;
        }
        .p0 { transform: translate(-50%, 10%); }
        .p1 { transform: translate(60%, -50%) rotate(-15deg); }
        .p2 { transform: translate(-50%, -110%); }
        .p3 { transform: translate(-160%, -50%) rotate(15deg); }

        /* GAME OVER SCREEN */
        #game-over-screen {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85); border-radius: 135px;
            display: none; flex-direction: column; justify-content: center; align-items: center;
            z-index: 100;
        }
        h1 { font-size: 48px; margin: 0 0 20px 0; color: gold; text-shadow: 2px 2px 5px black; }
        table { border-collapse: collapse; width: 60%; font-size: 24px; color: white; margin-bottom: 30px; }
        th, td { border-bottom: 1px solid #555; padding: 10px; text-align: center; }
        th { color: #aaa; }
        .winner-row { color: #2e8b57; font-weight: bold; }
        .human-row { color: gold; }
        .btn { padding: 15px 40px; font-size: 20px; border: none; border-radius: 50px; cursor: pointer; font-weight: bold; }
        .btn-green { background: #2e8b57; color: white; }
        .btn-red { background: #d00000; color: white; }

        #status-msg {
            position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%);
            font-size: 28px; font-weight: bold; text-shadow: 2px 2px 4px black; pointer-events: none;
        }
        #mini-scoreboard {
            position:absolute; top:10px; left:20px; 
            background:rgba(0,0,0,0.5); padding:10px; border-radius:8px;
        }
    </style>
</head>
<body>
    <div id="game-table">
        <div id="status-msg">Connecting...</div>
        <button id="start-btn" class="btn btn-green" onclick="startGame()" style="position:absolute; top:60%; left:50%; transform:translate(-50%, -50%);">Start Game</button>
        
        <div id="mini-scoreboard">
            <div>You: <span id="s0">0</span></div>
            <div>Right: <span id="s1">0</span></div>
            <div>Top: <span id="s2">0</span></div>
            <div>Left: <span id="s3">0</span></div>
        </div>
        
        <div id="top-bot" class="player-zone"></div>
        <div id="left-bot" class="player-zone"></div>
        <div id="right-bot" class="player-zone"></div>
        <div id="human-zone" class="player-zone"></div>
        <div id="trick-area"></div>

        <div id="game-over-screen">
            <h1 id="winner-title">YOU WON!</h1>
            <table id="result-table">
                <thead><tr><th>Rank</th><th>Player</th><th>Tricks</th></tr></thead>
                <tbody id="result-body"></tbody>
            </table>
            <button class="btn btn-red" onclick="location.reload()">End Game</button>
        </div>
    </div>

    <script>
        const socket = io();
        const suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'};

        function startGame() {
            document.getElementById('start-btn').style.display = 'none';
            socket.emit('start_game');
        }

        socket.on('status_update', data => {
            document.getElementById('status-msg').innerText = data.msg;
        });
        socket.on('error_msg', data => alert(data.msg));

        socket.on('game_update', state => {
            state.scores.forEach((s, i) => document.getElementById('s'+i).innerText = s);
            if (!state.game_over) {
                renderGame(state);
            } else {
                showGameOver(state);
            }
        });

        function renderGame(state) {
            document.getElementById('human-zone').innerHTML = '';
            document.getElementById('trick-area').innerHTML = '';
            document.getElementById('top-bot').innerHTML = '';
            document.getElementById('left-bot').innerHTML = '';
            document.getElementById('right-bot').innerHTML = '';

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
            document.getElementById('human-zone').innerHTML = '';
            document.getElementById('trick-area').innerHTML = '';
            document.getElementById('top-bot').innerHTML = '';
            document.getElementById('left-bot').innerHTML = '';
            document.getElementById('right-bot').innerHTML = '';
            document.getElementById('status-msg').innerText = '';

            const names = ["You", "Right Bot", "Top Bot", "Left Bot"];
            let results = state.scores.map((score, idx) => {
                return { name: names[idx], score: score, isHuman: idx===0 };
            });
            results.sort((a, b) => b.score - a.score);

            const tbody = document.getElementById('result-body');
            tbody.innerHTML = '';
            results.forEach((r, i) => {
                const tr = document.createElement('tr');
                if (i === 0) tr.classList.add('winner-row');
                if (r.isHuman) tr.classList.add('human-row');
                tr.innerHTML = `<td>#${i+1}</td><td>${r.name}</td><td>${r.score}</td>`;
                tbody.appendChild(tr);
            });

            const winnerIdx = state.winner;
            const title = document.getElementById('winner-title');
            if (winnerIdx === 0) {
                title.innerText = "VICTORY!";
                title.style.color = "gold";
            } else {
                title.innerText = "DEFEAT";
                title.style.color = "red";
            }
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

    # --- THE FIX IS HERE: Convert NumPy types to Python Ints ---
    winner = -1
    if sum(env.tricks_won) == 13:
        winner = int(np.argmax(env.tricks_won)) # Cast to int()

    return {
        'human_hand': [s_card(c) for c in sorted(env.hands[0], key=lambda x:(x.suit, x.rank))],
        'bot_counts': [len(env.hands[1]), len(env.hands[2]), len(env.hands[3])],
        'trick': trick_vis,
        'scores': [int(x) for x in env.tricks_won], # Cast list to ints
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
    env.reset()
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