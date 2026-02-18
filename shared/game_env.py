import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from typing import List, Optional
from enum import IntEnum
import pickle


# ============================================================
# GAME RULES (Hazz2):
#   DOS   (2) -> next player must draw +2 cards (stackable)
#   SIETE (7) -> player who played it chooses the active suit
#               (playable only on matching rank or matching suit)
#   AS    (1) -> skips the next player's turn
#               (playable only on matching rank or matching suit)
#   All other cards: playable on matching rank or matching suit
# ============================================================


class Suit(IntEnum):
    OROS = 0     # Coins
    COPAS = 1    # Cups
    ESPADAS = 2  # Swords
    BASTOS = 3   # Clubs


class Rank(IntEnum):
    AS = 1
    DOS = 2
    TRES = 3
    CUATRO = 4
    CINCO = 5
    SEIS = 6
    SIETE = 7
    SOTA = 10
    CABALLO = 11
    REY = 12


SUIT_NAMES = {0: "Coins", 1: "Cups", 2: "Swords", 3: "Clubs"}
RANK_NAMES = {
    1: "Ace", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 10: "Jack", 11: "Knight", 12: "King"
}


class Card:
    __slots__ = ['suit', 'rank', '_hash']

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
        self._hash = hash((suit, rank))

    def __repr__(self):
        return f"{RANK_NAMES.get(int(self.rank), self.rank)}-{SUIT_NAMES.get(int(self.suit), self.suit)}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return self._hash

    def to_dict(self):
        return {
            "suit": int(self.suit),
            "rank": int(self.rank),
            "repr": repr(self),
            "suit_name": SUIT_NAMES.get(int(self.suit), ""),
            "rank_name": RANK_NAMES.get(int(self.rank), ""),
        }


class Hazz2Env(gym.Env):
    """
    Two-player Hazz2 environment for Q-Learning training.

    Playability rule (same for all cards including specials):
      - A card is playable if it matches the top card rank OR the current active suit.
      - Under a penalty stack, only DOS (2) can be played.

    Special card effects (applied AFTER the card is played):
      DOS   (2) -> penalty stack +2 on next player
      SIETE (7) -> player chooses the active suit
                   (in this 2-player env simulation, suit stays same after SIETE)
      AS    (1) -> skip opponent's next turn
    """

    def __init__(self):
        super().__init__()

        self.valid_ranks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        self.rank_to_idx = {r: i for i, r in enumerate(self.valid_ranks)}

        self.full_deck = [
            Card(Suit(s), Rank(r))
            for s in range(4)
            for r in self.valid_ranks
        ]

        self.observation_space = spaces.Box(low=0, high=25, shape=(62,), dtype=np.int16)
        self.action_space = spaces.Discrete(25)

        self.agent_hand: List[Card] = []
        self.opponent_hand: List[Card] = []
        self.discard_pile: List[Card] = []
        self.deck: List[Card] = []
        self.current_suit = None
        self.penalty_stack = 0
        self.skip_opponent = False
        self.game_over = False
        self.winner = None
        self.total_turns = 0
        self.consecutive_draws = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.deck = self.full_deck.copy()
        random.shuffle(self.deck)

        self.agent_hand = [self.deck.pop() for _ in range(4)]
        self.opponent_hand = [self.deck.pop() for _ in range(4)]

        # First card on discard pile must not be a special card
        first_card = self.deck.pop()
        while first_card.rank in [Rank.AS, Rank.DOS, Rank.SIETE]:
            self.deck.append(first_card)
            random.shuffle(self.deck)
            first_card = self.deck.pop()

        self.discard_pile = [first_card]
        self.current_suit = first_card.suit
        self.penalty_stack = 0
        self.skip_opponent = False
        self.game_over = False
        self.winner = None
        self.total_turns = 0
        self.consecutive_draws = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(62, dtype=np.int16)

        # [0-9]: agent hand card counts by rank index
        for card in self.agent_hand:
            obs[self.rank_to_idx[card.rank]] += 1

        # [10-19]: top card rank one-hot
        # [20-23]: top card suit one-hot
        top = self.discard_pile[-1]
        obs[10 + self.rank_to_idx[top.rank]] = 1
        obs[20 + int(top.suit)] = 1

        # [24-48]: valid action mask
        for action in self.get_valid_actions():
            if action < 25:
                obs[24 + action] = 1

        obs[49] = min(len(self.opponent_hand), 25)  # opponent hand size
        obs[50] = 1                                  # is my turn (always 1)
        obs[51] = min(self.penalty_stack, 10)        # penalty stack

        return obs

    def get_valid_actions(self) -> List[int]:
        valid = [i for i, c in enumerate(self.agent_hand) if self._is_playable(c)]
        valid.append(len(self.agent_hand))  # draw action index
        return valid

    def _is_playable(self, card: Card) -> bool:
        """
        A card is playable if:
          - Under penalty stack: only DOS (2) is playable.
          - Otherwise: card rank matches top card rank, OR card suit matches current active suit.
        Note: SIETE and AS follow the exact same playability rule as all other cards.
        """
        top = self.discard_pile[-1]
        if self.penalty_stack > 0:
            return card.rank == Rank.DOS
        return (card.rank == top.rank or card.suit == self.current_suit)

    def _is_playable_opponent(self, card: Card) -> bool:
        """Same playability rule for the opponent."""
        top = self.discard_pile[-1]
        if self.penalty_stack > 0:
            return card.rank == Rank.DOS
        return (card.rank == top.rank or card.suit == self.current_suit)

    def step(self, action: int):
        if self.game_over:
            return self._get_observation(), 0, True, False, {}

        draw_action = len(self.agent_hand)
        reward = 0
        info = {}

        if action == draw_action:
            # Draw action
            if self.penalty_stack > 0:
                for _ in range(self.penalty_stack):
                    if self.deck:
                        self.agent_hand.append(self.deck.pop())
                self.penalty_stack = 0
                reward = -2
            else:
                if self.deck:
                    self.agent_hand.append(self.deck.pop())
                reward = -0.5
            self.consecutive_draws += 1

        else:
            if action >= len(self.agent_hand):
                return self._get_observation(), -5, False, False, {"error": "invalid_action"}

            card = self.agent_hand[action]
            if not self._is_playable(card):
                return self._get_observation(), -5, False, False, {"error": "unplayable_card"}

            self.agent_hand.pop(action)
            self.discard_pile.append(card)
            self.current_suit = card.suit
            self.consecutive_draws = 0

            # Apply special effects
            if card.rank == Rank.DOS:
                # DOS: stack +2 penalty on next player
                self.penalty_stack += 2
                reward = 0.5
            elif card.rank == Rank.SIETE:
                # SIETE: player chooses suit — in simulation suit stays the same
                reward = 1.0
            elif card.rank == Rank.AS:
                # AS: skip opponent's next turn
                self.skip_opponent = True
                reward = 0.5
            else:
                reward = 0.2

        # Check agent win
        if not self.agent_hand:
            self.game_over = True
            self.winner = "agent"
            return self._get_observation(), 10, True, False, {"winner": "agent"}

        # Opponent turn — skipped if agent played AS
        if not self.skip_opponent:
            self._opponent_turn()
        else:
            self.skip_opponent = False

        # Check opponent win
        if not self.opponent_hand:
            self.game_over = True
            self.winner = "opponent"
            return self._get_observation(), -10, True, False, {"winner": "opponent"}

        self.total_turns += 1

        if self.total_turns > 200:
            self.game_over = True
            return self._get_observation(), 0, True, False, {"winner": "draw"}

        return self._get_observation(), reward, False, False, info

    def _opponent_turn(self):
        if not self.opponent_hand:
            return

        # Under penalty stack: play DOS or draw all penalty cards
        if self.penalty_stack > 0:
            for i, card in enumerate(self.opponent_hand):
                if card.rank == Rank.DOS:
                    self.opponent_hand.pop(i)
                    self.discard_pile.append(card)
                    self.current_suit = card.suit
                    self.penalty_stack += 2
                    return
            for _ in range(self.penalty_stack):
                if self.deck:
                    self.opponent_hand.append(self.deck.pop())
            self.penalty_stack = 0
            return

        # Normal turn: play a random valid card or draw
        playable = [c for c in self.opponent_hand if self._is_playable_opponent(c)]
        if playable:
            card = random.choice(playable)
            self.opponent_hand.remove(card)
            self.discard_pile.append(card)
            self.current_suit = card.suit
            if card.rank == Rank.DOS:
                self.penalty_stack += 2
            elif card.rank == Rank.AS:
                # AS: opponent skips agent's next turn
                self.skip_opponent = True
            # SIETE: opponent played Seven — in simulation suit stays the same
        else:
            if self.deck:
                self.opponent_hand.append(self.deck.pop())


class QLearningAgent:
    """Tabular Q-Learning agent — used for inference only during the multi-agent game."""

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(25, dtype=np.float32))
        self.total_updates = 0

    def _state_to_key(self, obs: np.ndarray) -> tuple:
        return tuple(obs.astype(np.int16).tolist())

    def get_action(self, obs: np.ndarray, valid_actions: List[int]) -> int:
        """Greedy action — no exploration during inference."""
        state_key = self._state_to_key(obs)
        q_values = self.q_table[state_key]
        masked_q = np.full(25, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def save(self, path: str):
        data = {
            'q_table': dict(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        q_table_data = data['q_table'] if isinstance(data, dict) and 'q_table' in data else data
        self.q_table = defaultdict(lambda: np.zeros(25, dtype=np.float32), q_table_data)
        if isinstance(data, dict):
            self.alpha = data.get('alpha', self.alpha)
            self.gamma = data.get('gamma', self.gamma)
            self.epsilon = data.get('epsilon', self.epsilon)
            self.total_updates = data.get('total_updates', self.total_updates)