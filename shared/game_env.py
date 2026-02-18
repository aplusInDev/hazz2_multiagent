import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from typing import List, Optional
from enum import IntEnum
import pickle


class Suit(IntEnum):
    OROS = 0
    COPAS = 1
    ESPADAS = 2
    BASTOS = 3


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


class Card:
    __slots__ = ['suit', 'rank', '_hash']

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
        self._hash = hash((suit, rank))

    def __repr__(self):
        return f"{self.rank.value}-{['O','C','E','B'][self.suit]}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return self._hash

    def to_dict(self):
        return {"suit": int(self.suit), "rank": int(self.rank), "repr": repr(self)}

    @classmethod
    def from_dict(cls, d):
        return cls(Suit(d["suit"]), Rank(d["rank"]))


class Hazz2Env(gym.Env):
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.deck = self.full_deck.copy()
        random.shuffle(self.deck)

        self.agent_hand = [self.deck.pop() for _ in range(4)]
        self.opponent_hand = [self.deck.pop() for _ in range(4)]

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

        for card in self.agent_hand:
            rank_idx = self.rank_to_idx[card.rank]
            obs[rank_idx] += 1

        top_card = self.discard_pile[-1]
        rank_idx = self.rank_to_idx[top_card.rank]
        obs[10 + rank_idx] = 1
        obs[20 + top_card.suit] = 1

        valid_actions = self.get_valid_actions()
        for action in valid_actions:
            if action < 25:
                obs[24 + action] = 1

        obs[49] = min(len(self.opponent_hand), 25)
        obs[50] = 1
        obs[51] = min(self.penalty_stack, 10)

        return obs

    def get_valid_actions(self) -> List[int]:
        valid = []
        for i, card in enumerate(self.agent_hand):
            if self._is_playable(card):
                valid.append(i)
        valid.append(len(self.agent_hand))
        return valid

    def _is_playable(self, card: Card) -> bool:
        top_card = self.discard_pile[-1]
        if self.penalty_stack > 0:
            return card.rank == Rank.DOS
        return (card.rank == top_card.rank or card.suit == self.current_suit or
                card.rank == Rank.AS)

    def step(self, action: int):
        if self.game_over:
            return self._get_observation(), 0, True, False, {}

        draw_action = len(self.agent_hand)
        reward = 0
        info = {}

        if action == draw_action:
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

            if card.rank == Rank.DOS:
                self.penalty_stack += 2
                reward = 0.5
            elif card.rank == Rank.AS:
                reward = 1.0
            elif card.rank == Rank.SIETE:
                self.skip_opponent = True
                reward = 0.5
            else:
                reward = 0.2

        if not self.agent_hand:
            self.game_over = True
            self.winner = "agent"
            reward = 10
            return self._get_observation(), reward, True, False, {"winner": "agent"}

        if not self.skip_opponent:
            self._opponent_turn()
        else:
            self.skip_opponent = False

        if not self.opponent_hand:
            self.game_over = True
            self.winner = "opponent"
            reward = -10
            return self._get_observation(), reward, True, False, {"winner": "opponent"}

        self.total_turns += 1

        if self.total_turns > 200:
            self.game_over = True
            reward = 0
            return self._get_observation(), reward, True, False, {"winner": "draw"}

        return self._get_observation(), reward, False, False, info

    def _opponent_turn(self):
        if not self.opponent_hand:
            return

        if self.penalty_stack > 0:
            has_dos = any(c.rank == Rank.DOS for c in self.opponent_hand)
            if has_dos:
                for i, card in enumerate(self.opponent_hand):
                    if card.rank == Rank.DOS:
                        self.opponent_hand.pop(i)
                        self.discard_pile.append(card)
                        self.current_suit = card.suit
                        self.penalty_stack += 2
                        return
            else:
                for _ in range(self.penalty_stack):
                    if self.deck:
                        self.opponent_hand.append(self.deck.pop())
                self.penalty_stack = 0
                return

        playable = [c for c in self.opponent_hand if self._is_playable_for_opponent(c)]
        if playable:
            card = random.choice(playable)
            self.opponent_hand.remove(card)
            self.discard_pile.append(card)
            self.current_suit = card.suit
            if card.rank == Rank.DOS:
                self.penalty_stack += 2
            elif card.rank == Rank.SIETE:
                self.skip_opponent = False
        else:
            if self.deck:
                self.opponent_hand.append(self.deck.pop())

    def _is_playable_for_opponent(self, card: Card) -> bool:
        top_card = self.discard_pile[-1]
        if self.penalty_stack > 0:
            return card.rank == Rank.DOS
        return (card.rank == top_card.rank or card.suit == self.current_suit or
                card.rank == Rank.AS)

    def get_state_for_player(self, player_hand: List[Card]) -> dict:
        top_card = self.discard_pile[-1] if self.discard_pile else None
        return {
            "hand": [c.to_dict() for c in player_hand],
            "top_card": top_card.to_dict() if top_card else None,
            "current_suit": int(self.current_suit),
            "penalty_stack": self.penalty_stack,
            "deck_size": len(self.deck),
            "game_over": self.game_over,
            "winner": self.winner,
        }


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(25))
        self.total_updates = 0

    def get_state_key(self, obs: np.ndarray) -> tuple:
        return tuple(obs[:52].tolist())

    def get_action(self, obs: np.ndarray, valid_actions: List[int]) -> int:
        state_key = self.get_state_key(obs)
        q_values = self.q_table[state_key]
        valid_q = [(a, q_values[a]) for a in valid_actions]
        return max(valid_q, key=lambda x: x[1])[0]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(25), data)
