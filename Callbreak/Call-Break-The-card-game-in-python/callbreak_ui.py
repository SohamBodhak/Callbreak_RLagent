import pygame
import sys
import time
import threading
import numpy as np
from callbreak_env import CallbreakEnv
from mcts_agent_rotate import ISMCTSAgent

# --- CONFIGURATION ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BG_COLOR = (34, 139, 34)  # Poker Table Green
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 20, 60)
GRAY = (100, 100, 100)
HIGHLIGHT = (255, 215, 0) # Gold

CARD_WIDTH = 80
CARD_HEIGHT = 120
FONT_SIZE = 24

# MCTS Settings
MCTS_SIMS = 25  # Lower than 50 to keep the UI responsive (approx 1-2 sec wait)

class CallbreakUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Callbreak AI: Human vs MCTS Super-Bots")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', FONT_SIZE, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 48, bold=True)

        # Initialize Game Engine
        self.env = CallbreakEnv()
        self.env.reset()
        
        # Initialize MCTS Bot
        print("Loading MCTS Brain...")
        self.bot_agent = ISMCTSAgent()
        # Override simulation count for UI responsiveness
        global SIMULATIONS
        SIMULATIONS = MCTS_SIMS 
        
        self.human_turn = False
        self.game_over = False
        self.winner_text = ""
        
        # Threading for Bot Thinking (prevents UI freeze)
        self.bot_thinking = False
        self.bot_thread = None
        self.selected_bot_action = None

    def draw_card(self, card, x, y, faded=False):
        """Draws a card sprite procedurally."""
        # Card Body
        rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        color = GRAY if faded else WHITE
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, BLACK, rect, 2, border_radius=8)
        
        # Text (Rank and Suit)
        # Suits: 0=C, 1=D, 2=H, 3=S
        suit_symbols = ['♣', '♦', '♥', '♠']
        suit_colors = [BLACK, RED, RED, BLACK]
        rank_map = {11:'J', 12:'Q', 13:'K', 14:'A'}
        
        rank_str = str(rank_map.get(card.rank, card.rank))
        suit_str = suit_symbols[card.suit]
        text_col = suit_colors[card.suit]
        
        if faded: text_col = (50, 50, 50)

        # Top Left
        r_surf = self.font.render(rank_str, True, text_col)
        s_surf = self.font.render(suit_str, True, text_col)
        self.screen.blit(r_surf, (x + 5, y + 5))
        self.screen.blit(s_surf, (x + 5, y + 25))
        
        # Center Big Suit
        big_s_surf = pygame.font.SysFont('Arial', 48).render(suit_str, True, text_col)
        self.screen.blit(big_s_surf, (x + 20, y + 35))

    def get_hand_positions(self, player_idx, num_cards):
        """Calculates coordinates to center the hand on the correct side."""
        spacing = 40
        total_width = (num_cards - 1) * spacing + CARD_WIDTH
        
        if player_idx == 0: # Human (Bottom)
            start_x = (SCREEN_WIDTH - total_width) // 2
            y = SCREEN_HEIGHT - CARD_HEIGHT - 20
            return [(start_x + i*spacing, y) for i in range(num_cards)]
            
        elif player_idx == 1: # Right
            start_y = (SCREEN_HEIGHT - total_width) // 2
            x = SCREEN_WIDTH - CARD_WIDTH - 20
            return [(x, start_y + i*spacing) for i in range(num_cards)]
            
        elif player_idx == 2: # Top
            start_x = (SCREEN_WIDTH - total_width) // 2
            y = 20
            return [(start_x + i*spacing, y) for i in range(num_cards)]
            
        elif player_idx == 3: # Left
            start_y = (SCREEN_HEIGHT - total_width) // 2
            x = 20
            return [(x, start_y + i*spacing) for i in range(num_cards)]

    def handle_click(self, pos):
        if not self.human_turn or self.game_over: return
        
        hand = self.env.hands[0]
        # Sort hand for display (Suit then Rank)
        # Note: We must map the visual click back to the actual card index in env
        # Visual hand is sorted, Env hand might not be.
        
        # 1. Get Visual Sorted Hand
        visual_hand = sorted(hand, key=lambda x: (x.suit, x.rank))
        positions = self.get_hand_positions(0, len(visual_hand))
        
        clicked_card = None
        
        # Check collision (iterate backwards to click top card first)
        for i in range(len(positions) - 1, -1, -1):
            x, y = positions[i]
            rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
            if rect.collidepoint(pos):
                clicked_card = visual_hand[i]
                break
        
        if clicked_card:
            # Validate Move
            mask = self.env._get_valid_mask()
            idx = (clicked_card.rank - 2) * 4 + clicked_card.suit
            
            if mask[idx] == 1.0:
                self.apply_move(0, clicked_card, idx)
            else:
                print("Invalid Move! Follow suit.")

    def bot_think_worker(self):
        """Runs MCTS in background thread."""
        action_idx = self.bot_agent.select_move(self.env)
        self.selected_bot_action = action_idx

    def update_game_state(self):
        # 1. Check for Game Over
        if sum(self.env.tricks_won) == 13:
            self.game_over = True
            # Determine winner
            w_idx = np.argmax(self.env.tricks_won)
            if w_idx == 0: self.winner_text = "YOU WIN!"
            else: self.winner_text = f"Bot {w_idx} Wins!"
            return

        # 2. Check Round End (Trick Full)
        if len(self.env.trick_cards) == 4:
            pygame.time.delay(1500) # Pause to see result
            
            # Resolve Trick Logic manually since we aren't using env.step loop
            winner_rel = self.env._get_winner_idx(self.env.trick_cards)
            winner_abs = (self.env.leader_idx + winner_rel) % 4
            self.env.tricks_won[winner_abs] += 1
            self.env.leader_idx = winner_abs
            self.env.current_player = winner_abs
            self.env.trick_cards = []
            self.env.cards_played_history.update(self.env.trick_cards_history_indices) # Custom tracking
            return

        # 3. Determine Turn
        cp = self.env.current_player
        
        if cp == 0:
            self.human_turn = True
        else:
            self.human_turn = False
            if not self.bot_thinking:
                self.bot_thinking = True
                self.selected_bot_action = None
                # Start Thread
                self.bot_thread = threading.Thread(target=self.bot_think_worker)
                self.bot_thread.start()
            
            # Check if thread finished
            if self.selected_bot_action is not None:
                # Apply Bot Move
                idx = self.selected_bot_action
                # Convert idx back to card obj
                r, s = (idx // 4) + 2, idx % 4
                card_obj = None
                for c in self.env.hands[cp]:
                    if c.rank == r and c.suit == s: card_obj = c; break
                
                self.apply_move(cp, card_obj, idx)
                self.bot_thinking = False
                self.selected_bot_action = None

    def apply_move(self, p_idx, card_obj, action_idx):
        self.env.hands[p_idx].remove(card_obj)
        self.env.trick_cards.append(card_obj)
        # We need to temporarily track what indices are on board for history update later
        if not hasattr(self.env, 'trick_cards_history_indices'):
            self.env.trick_cards_history_indices = []
        if len(self.env.trick_cards) == 1: self.env.trick_cards_history_indices = [] # Reset on lead
        
        self.env.trick_cards_history_indices.append(action_idx)
        self.env.current_player = (self.env.current_player + 1) % 4

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        # 1. Draw Hands
        mask = self.env._get_valid_mask() if self.human_turn else None
        
        for p in range(4):
            # Sort hands
            visual_hand = sorted(self.env.hands[p], key=lambda x: (x.suit, x.rank))
            positions = self.get_hand_positions(p, len(visual_hand))
            
            for i, (x, y) in enumerate(positions):
                card = visual_hand[i]
                
                if p == 0: # HUMAN - Face Up
                    idx = (card.rank - 2) * 4 + card.suit
                    is_valid = True
                    if self.human_turn and mask[idx] == 0: is_valid = False
                    
                    # Highlight valid cards on hover (optional) or just gray out invalid
                    self.draw_card(card, x, y, faded=(not is_valid))
                else:
                    # BOTS - Face Down
                    # FIX: Only draw cards if the bot actually HAS them in the environment
                    actual_hand_size = len(self.env.hands[p])
                    
                    # Recalculate positions based on ACTUAL hand size
                    positions = self.get_hand_positions(p, actual_hand_size)
                    
                    for i, (x, y) in enumerate(positions):
                        rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
                        pygame.draw.rect(self.screen, (139, 0, 0), rect, border_radius=8)
                        pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=8)

        # 2. Draw Table (Trick)
        center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        # Positions for cards played: Bottom, Right, Top, Left
        trick_offsets = [(0, 60), (100, 0), (0, -60), (-100, 0)]
        
        # We need to map trick cards to who played them.
        # This is tricky because self.env.trick_cards is just a list.
        # We know who led: self.env.leader_idx
        
        current_trick_size = len(self.env.trick_cards)
        if current_trick_size > 0:
            start_p = self.env.leader_idx
            for i, card in enumerate(self.env.trick_cards):
                p_idx = (start_p + i) % 4
                off_x, off_y = trick_offsets[p_idx]
                self.draw_card(card, center_x + off_x - CARD_WIDTH//2, center_y + off_y - CARD_HEIGHT//2)

        # 3. UI Info
        # Scoreboard
        score_text = f"YOU: {self.env.tricks_won[0]}  |  Bot R: {self.env.tricks_won[1]}  |  Bot T: {self.env.tricks_won[2]}  |  Bot L: {self.env.tricks_won[3]}"
        s_surf = self.font.render(score_text, True, WHITE)
        self.screen.blit(s_surf, (SCREEN_WIDTH//2 - s_surf.get_width()//2, 10))

        # Status Text
        if self.game_over:
            status = self.winner_text
            col = HIGHLIGHT
        elif self.human_turn:
            status = "Your Turn"
            col = WHITE
        else:
            status = f"Bot {self.env.current_player} Thinking..."
            col = (200, 200, 200)
            
        st_surf = self.big_font.render(status, True, col)
        self.screen.blit(st_surf, (SCREEN_WIDTH//2 - st_surf.get_width()//2, SCREEN_HEIGHT - 200))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left Click
                        self.handle_click(event.pos)
            
            if not self.game_over:
                self.update_game_state()
            
            self.draw()
            self.clock.tick(30)

if __name__ == "__main__":
    game = CallbreakUI()
    game.run()