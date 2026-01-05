import pygame
import sys
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
ERROR_COLOR = (255, 69, 0) # Orange-Red for errors

CARD_WIDTH = 80
CARD_HEIGHT = 120
FONT_SIZE = 24
MCTS_SIMS = 25 

class CallbreakUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Callbreak AI: Human vs MCTS Super-Bots")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', FONT_SIZE, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 48, bold=True)
        self.msg_font = pygame.font.SysFont('Arial', 36, bold=True)

        self.env = CallbreakEnv()
        self.env.reset()
        
        print("Loading MCTS Brain...")
        self.bot_agent = ISMCTSAgent()
        
        self.human_turn = False
        self.game_over = False
        self.winner_text = ""
        self.message_text = "" # To show "Invalid Move"
        self.message_timer = 0
        
        # KEYBOARD SELECTION STATE
        self.selected_index = 0 # Which card (0 to N-1) is highlighted
        
        self.bot_thinking = False
        self.bot_thread = None
        self.selected_bot_action = None

    def draw_card(self, card, x, y, faded=False, selected=False):
        """Draws a card sprite procedurally."""
        rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        
        # Highlight Logic (Gold Border + Pop Up)
        if selected:
            rect.y -= 20 # Pop up animation
            y -= 20
        
        color = GRAY if faded else WHITE
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        
        # Border
        border_col = HIGHLIGHT if selected else BLACK
        border_thick = 5 if selected else 2
        pygame.draw.rect(self.screen, border_col, rect, border_thick, border_radius=8)
        
        # Content
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
        spacing = 45 if player_idx == 0 else 30 # Wider spacing for human
        total_width = (num_cards - 1) * spacing + CARD_WIDTH
        
        if player_idx == 0: # Human (Bottom)
            start_x = (SCREEN_WIDTH - total_width) // 2
            y = SCREEN_HEIGHT - CARD_HEIGHT - 30
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

    def handle_input(self, key):
        if not self.human_turn or self.game_over: return
        
        # Get Visual Hand (Sorted)
        hand = self.env.hands[0]
        visual_hand = sorted(hand, key=lambda x: (x.suit, x.rank))
        hand_size = len(visual_hand)
        
        if hand_size == 0: return

        if key == pygame.K_LEFT:
            self.selected_index = (self.selected_index - 1) % hand_size
            
        elif key == pygame.K_RIGHT:
            self.selected_index = (self.selected_index + 1) % hand_size
            
        elif key == pygame.K_RETURN: # Enter Key
            # Ensure index is safe (in case hand changed weirdly)
            if self.selected_index >= hand_size: self.selected_index = hand_size - 1
            
            selected_card = visual_hand[self.selected_index]
            
            # Validation
            mask = self.env._get_valid_mask()
            idx = (selected_card.rank - 2) * 4 + selected_card.suit
            
            if mask[idx] == 1.0:
                self.apply_move(0, selected_card, idx)
                # Reset selection to 0 or keep nearby
                self.selected_index = max(0, self.selected_index - 1)
            else:
                self.show_message("Invalid Move! Follow Suit.")

    def show_message(self, text):
        self.message_text = text
        self.message_timer = 60 # Show for 2 seconds (30fps * 2)

    def bot_think_worker(self):
        action_idx = self.bot_agent.select_move(self.env)
        self.selected_bot_action = action_idx

    def update_game_state(self):
        # 1. Game Over Check
        if sum(self.env.tricks_won) == 13:
            self.game_over = True
            w_idx = np.argmax(self.env.tricks_won)
            self.winner_text = "YOU WIN!" if w_idx == 0 else f"Bot {w_idx} Wins!"
            return

        # 2. Trick End Logic
        if len(self.env.trick_cards) == 4:
            pygame.time.delay(1200) 
            winner_rel = self.env._get_winner_idx(self.env.trick_cards)
            winner_abs = (self.env.leader_idx + winner_rel) % 4
            self.env.tricks_won[winner_abs] += 1
            self.env.leader_idx = winner_abs
            self.env.current_player = winner_abs
            self.env.trick_cards = []
            self.env.cards_played_history.update(self.env.trick_cards_history_indices) 
            return

        # 3. Turn Logic
        cp = self.env.current_player
        if cp == 0:
            self.human_turn = True
        else:
            self.human_turn = False
            if not self.bot_thinking:
                self.bot_thinking = True
                self.selected_bot_action = None
                self.bot_thread = threading.Thread(target=self.bot_think_worker)
                self.bot_thread.start()
            
            if self.selected_bot_action is not None:
                idx = self.selected_bot_action
                r, s = (idx // 4) + 2, idx % 4
                card_obj = None
                for c in self.env.hands[cp]:
                    if c.rank == r and c.suit == s: card_obj = c; break
                
                # Bot Ghost Card Fix: Only play if they have it
                if card_obj:
                    self.apply_move(cp, card_obj, idx)
                else:
                    # Fallback (Safety valve for '0 of Spades' issues)
                    # Just play first card they actually have
                    if self.env.hands[cp]:
                        fallback = self.env.hands[cp][0]
                        f_idx = (fallback.rank - 2)*4 + fallback.suit
                        self.apply_move(cp, fallback, f_idx)

                self.bot_thinking = False
                self.selected_bot_action = None

    def apply_move(self, p_idx, card_obj, action_idx):
        if card_obj in self.env.hands[p_idx]:
            self.env.hands[p_idx].remove(card_obj)
            self.env.trick_cards.append(card_obj)
            
            if not hasattr(self.env, 'trick_cards_history_indices'):
                self.env.trick_cards_history_indices = []
            if len(self.env.trick_cards) == 1: 
                self.env.trick_cards_history_indices = []
            
            self.env.trick_cards_history_indices.append(action_idx)
            self.env.current_player = (self.env.current_player + 1) % 4

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        # 1. Draw Hands
        mask = self.env._get_valid_mask() if self.human_turn else None
        
        for p in range(4):
            # Strict Hand Size Check (Fixes Visual Sync bugs)
            actual_size = len(self.env.hands[p])
            if actual_size == 0: continue
            
            # Sort hands for visual clarity
            visual_hand = sorted(self.env.hands[p], key=lambda x: (x.suit, x.rank))
            positions = self.get_hand_positions(p, actual_size)
            
            for i, (x, y) in enumerate(positions):
                if p == 0: # HUMAN
                    card = visual_hand[i]
                    idx = (card.rank - 2) * 4 + card.suit
                    is_valid = True
                    if self.human_turn and mask is not None and idx < len(mask) and mask[idx] == 0: 
                        is_valid = False
                    
                    # Selection Logic
                    is_selected = (i == self.selected_index and self.human_turn)
                    
                    self.draw_card(card, x, y, faded=(not is_valid), selected=is_selected)
                else:
                    # BOTS - Face Down
                    rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
                    pygame.draw.rect(self.screen, (139, 0, 0), rect, border_radius=8)
                    pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=8)

        # 2. Table
        center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        trick_offsets = [(0, 60), (100, 0), (0, -60), (-100, 0)]
        
        if len(self.env.trick_cards) > 0:
            start_p = self.env.leader_idx
            for i, card in enumerate(self.env.trick_cards):
                p_idx = (start_p + i) % 4
                off_x, off_y = trick_offsets[p_idx]
                self.draw_card(card, center_x + off_x - CARD_WIDTH//2, center_y + off_y - CARD_HEIGHT//2)

        # 3. Text Info
        score_text = f"YOU: {self.env.tricks_won[0]}  |  Right: {self.env.tricks_won[1]}  |  Top: {self.env.tricks_won[2]}  |  Left: {self.env.tricks_won[3]}"
        s_surf = self.font.render(score_text, True, WHITE)
        self.screen.blit(s_surf, (SCREEN_WIDTH//2 - s_surf.get_width()//2, 10))

        # Status / Error Messages
        if self.message_timer > 0:
            self.message_timer -= 1
            msg_surf = self.msg_font.render(self.message_text, True, ERROR_COLOR)
            # Draw with shadow
            shadow = self.msg_font.render(self.message_text, True, BLACK)
            self.screen.blit(shadow, (SCREEN_WIDTH//2 - shadow.get_width()//2 + 2, SCREEN_HEIGHT - 250 + 2))
            self.screen.blit(msg_surf, (SCREEN_WIDTH//2 - msg_surf.get_width()//2, SCREEN_HEIGHT - 250))
        
        # Turn Indicator
        if self.game_over:
            status = self.winner_text
            col = HIGHLIGHT
        elif self.human_turn:
            status = "YOUR TURN (Select with Arrows, Enter to Play)"
            col = WHITE
        else:
            status = f"Bot {self.env.current_player} is Thinking..."
            col = (200, 200, 200)
            
        st_surf = self.big_font.render(status, True, col)
        self.screen.blit(st_surf, (SCREEN_WIDTH//2 - st_surf.get_width()//2, SCREEN_HEIGHT - 180))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    sys.exit()
                # KEYBOARD EVENTS
                if event.type == pygame.KEYDOWN:
                    self.handle_input(event.key)
            
            if not self.game_over:
                self.update_game_state()
            
            self.draw()
            self.clock.tick(30)

if __name__ == "__main__":
    game = CallbreakUI()
    game.run()