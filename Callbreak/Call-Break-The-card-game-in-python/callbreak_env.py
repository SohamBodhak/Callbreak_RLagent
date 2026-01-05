import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import sys

try:
    from card import Deck, Card
except ImportError:
    class Card:
        def __init__(self, rank, suit): self.rank, self.suit = rank, suit
        def __repr__(self): return f"{self.rank}-{self.suit}"
    class Deck:
        def __init__(self): self.cards = [Card(r, s) for s in range(4) for r in range(2, 15)]
        def pop_card(self): return self.cards.pop()
        def shuffle(self): random.shuffle(self.cards)

class CallbreakEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CallbreakEnv, self).__init__()
        self.action_space = spaces.Discrete(52)
        # 156 floats (Hand + Board + History)
        self.observation_space = spaces.Box(low=0, high=1, shape=(156,), dtype=np.float32)
        
        self.deck = None
        self.hands = []
        self.trick_cards = []
        self.tricks_won = [0] * 4
        self.cards_played_history = set()
        self.current_player = 0
        self.leader_idx = 0

    # --- THE SMART STRATEGY (Baked In) ---
    def _heuristic_strategy(self, hand, trick_cards):
        """Standard logic: Win cheap or throw trash."""
        if not hand: return None

        # 1. Filter Valid Moves
        lead_suit = trick_cards[0].suit if trick_cards else None
        valid = hand
        if lead_suit:
            s = [c for c in hand if c.suit == lead_suit]
            if s: valid = s
            else:
                sp = [c for c in hand if c.suit == 3]
                if sp: valid = sp
        
        # 2. Analyze Board
        current_winner_is_spade = False
        current_best_rank = -1
        
        if trick_cards:
            best_card = trick_cards[0]
            for i, c in enumerate(trick_cards):
                if i==0: continue
                # Update best card logic (Standard Callbreak)
                if c.suit == 3 and best_card.suit != 3: best_card = c
                elif c.suit == 3 and best_card.suit == 3:
                    if c.rank > best_card.rank: best_card = c
                elif c.suit == trick_cards[0].suit and best_card.suit != 3:
                    if c.rank > best_card.rank: best_card = c
            
            if best_card.suit == 3: current_winner_is_spade = True
            current_best_rank = best_card.rank

        # 3. Decision
        winning_moves = []
        for c in valid:
            is_winner = False
            if not trick_cards: is_winner = True # Lead high?
            else:
                if c.suit == lead_suit and not current_winner_is_spade:
                    if c.rank > current_best_rank: is_winner = True
                elif c.suit == 3 and not current_winner_is_spade:
                    is_winner = True
                elif c.suit == 3 and current_winner_is_spade:
                    if c.rank > current_best_rank: is_winner = True
            
            if is_winner: winning_moves.append(c)

        if winning_moves:
            winning_moves.sort(key=lambda x: x.rank)
            return winning_moves[0] # Smallest winner
        else:
            valid.sort(key=lambda x: x.rank)
            return valid[0] # Smallest trash

    def _get_obs(self):
        hand_obs = np.zeros(52, dtype=np.float32)
        for c in self.hands[0]: 
            idx = (c.rank - 2) * 4 + c.suit
            if 0 <= idx < 52: hand_obs[idx] = 1.0
        
        board_obs = np.zeros(52, dtype=np.float32)
        for c in self.trick_cards:
            idx = (c.rank - 2) * 4 + c.suit
            if 0 <= idx < 52: board_obs[idx] = 1.0
            
        history_obs = np.zeros(52, dtype=np.float32)
        for idx in self.cards_played_history:
            if 0 <= idx < 52: history_obs[idx] = 1.0
            
        return np.concatenate([hand_obs, board_obs, history_obs])

    def _get_valid_mask(self):
        mask = np.zeros(52, dtype=np.float32)
        if not self.hands[0]: return mask
        
        lead_card = self.trick_cards[0] if self.trick_cards else None
        hand = self.hands[0]
        valid_cards = hand
        
        if lead_card:
            same_suit = [c for c in hand if c.suit == lead_card.suit]
            if same_suit: valid_cards = same_suit
            else:
                spades = [c for c in hand if c.suit == 3]
                if spades: valid_cards = spades
        
        for c in valid_cards:
            idx = (c.rank - 2) * 4 + c.suit
            if 0 <= idx < 52: mask[idx] = 1.0
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = Deck()
        if hasattr(self.deck, 'shuffle'): self.deck.shuffle()
        else: random.shuffle(self.deck.cards)
            
        self.hands = [[], [], [], []]
        for i in range(52): self.hands[i % 4].append(self.deck.pop_card())
            
        self.trick_cards = []
        self.tricks_won = [0] * 4
        self.cards_played_history = set()
        self.leader_idx = random.randint(0, 3)
        self.current_player = self.leader_idx
        
        # Fast-Forward
        while self.current_player != 0:
            self._bot_play(self.current_player)
            self.current_player = (self.current_player + 1) % 4
            
        return self._get_obs(), {"action_mask": self._get_valid_mask()}

    def step(self, action_idx):
        action_idx = int(action_idx)
        if not self.hands[0]: 
            return self._get_obs(), 0, True, False, {"action_mask": self._get_valid_mask()}

        mask = self._get_valid_mask()
        if mask[action_idx] == 0:
            valid_indices = [i for i, m in enumerate(mask) if m == 1]
            if not valid_indices: return self._get_obs(), 0, True, False, {"error": "No Moves"}
            action_idx = random.choice(valid_indices)

        card_obj = None
        for c in self.hands[0]:
            if ((c.rank - 2) * 4 + c.suit) == action_idx: card_obj = c; break
        
        if card_obj is None: return self._get_obs(), -10, True, False, {"error": "Sync"}

        self.hands[0].remove(card_obj)
        self.trick_cards.append(card_obj)
        self.cards_played_history.add(action_idx)
        
        self.current_player = 1
        while len(self.trick_cards) < 4:
            self._bot_play(self.current_player)
            self.current_player = (self.current_player + 1) % 4
            
        # Resolve Trick
        winner_rel = self._get_winner_idx(self.trick_cards)
        winner_abs = (self.leader_idx + winner_rel) % 4
        self.tricks_won[winner_abs] += 1
        self.leader_idx = winner_abs 
        
        reward = 1.0 if winner_abs == 0 else 0.0
        terminated = sum(self.tricks_won) == 13
            
        self.trick_cards = []
        if not terminated:
            self.current_player = self.leader_idx
            while self.current_player != 0:
                self._bot_play(self.current_player)
                self.current_player = (self.current_player + 1) % 4
                
        return self._get_obs(), reward, terminated, False, {"action_mask": self._get_valid_mask()}

    def _bot_play(self, player_idx):
        if not self.hands[player_idx]: return
        # USE HEURISTIC STRATEGY
        card = self._heuristic_strategy(self.hands[player_idx], self.trick_cards)
        self.hands[player_idx].remove(card)
        self.trick_cards.append(card)
        self.cards_played_history.add((card.rank-2)*4 + card.suit)

    def _get_winner_idx(self, trick):
        if not trick: return 0
        best_idx, best_card = 0, trick[0]
        lead_suit = trick[0].suit
        for i, c in enumerate(trick):
            if i==0: continue
            if c.suit == 3 and best_card.suit != 3: best_card=c; best_idx=i
            elif c.suit == 3 and best_card.suit == 3: 
                if c.rank > best_card.rank: best_card=c; best_idx=i
            elif c.suit == lead_suit and best_card.suit != 3:
                if c.rank > best_card.rank: best_card=c; best_idx=i
        return best_idx