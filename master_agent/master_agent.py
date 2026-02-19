import asyncio
import json
import logging
import random
import sys
import os
from collections import Counter
from typing import List

sys.path.insert(0, '/app/shared')

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from game_env import Hazz2Env, Card, Suit, Rank

logging.basicConfig(level=logging.INFO, format='%(asctime)s [MASTER] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# GAME RULES (Hazz2):
#   All cards (including specials) are playable only when:
#     - Card rank matches top card rank, OR
#     - Card suit matches current active suit.
#   Under a penalty stack, ONLY DOS (2) can be played.
#
#   Special effects after playing:
#     DOS   (2) -> next player draws +2 (stackable)
#     SIETE (7) -> player who played it chooses the active suit
#     AS    (1) -> next player's turn is skipped
# ============================================================

XMPP_SERVER = os.environ.get("XMPP_SERVER", "ejabberd")
MASTER_JID = os.environ.get("MASTER_JID", f"master@{XMPP_SERVER}")
MASTER_PASSWORD = os.environ.get("MASTER_PASSWORD", "master_pass")
QAGENT_JID = os.environ.get("QAGENT_JID", f"qagent@{XMPP_SERVER}")
RANDOM_JID = os.environ.get("RANDOM_JID", f"randomagent@{XMPP_SERVER}")
HUMAN_JID = os.environ.get("HUMAN_JID", f"human@{XMPP_SERVER}")
HEURISTIC_JID = os.environ.get("HEURISTIC_JID", f"heuristic@{XMPP_SERVER}")

ALL_PLAYERS = ["human", "qagent", "randomagent", "heuristic"]

JID_TO_PLAYER = {
    HUMAN_JID: "human",
    QAGENT_JID: "qagent",
    RANDOM_JID: "randomagent",
    HEURISTIC_JID: "heuristic",
}

PLAYER_TO_JID = {v: k for k, v in JID_TO_PLAYER.items()}


class GameState:
    def __init__(self):
        self.env = Hazz2Env()
        self.hands = {}
        self.active_players = []
        self.turn_order = []
        self.current_turn_idx = 0
        self.game_active = False
        self.penalty_stack = 0
        self.skip_next = False
        self.total_turns = 0
        self.deck = []
        self.discard_pile = []
        self.current_suit = None
        self.finish_order = []
        self.awaiting_suit_choice = False
        self.suit_chooser = None

    def reset(self, watch_mode: bool = False):
        deck = self.env.full_deck.copy()
        random.shuffle(deck)

        # In watch mode, human is a spectator — only agents play
        players = [p for p in ALL_PLAYERS if p != "human"] if watch_mode else ALL_PLAYERS.copy()
        self.active_players = players.copy()
        self.finish_order = []
        self.hands = {p: [] for p in players}
        for _ in range(4):
            for p in players:
                self.hands[p].append(deck.pop())

        # First card on table must not be a special card
        first_card = deck.pop()
        while first_card.rank in [Rank.AS, Rank.DOS, Rank.SIETE]:
            deck.append(first_card)
            random.shuffle(deck)
            first_card = deck.pop()

        self.deck = deck
        self.discard_pile = [first_card]
        self.current_suit = first_card.suit
        self.penalty_stack = 0
        self.skip_next = False
        self.game_active = True
        self.turn_order = players.copy()
        random.shuffle(self.turn_order)
        self.current_turn_idx = 0
        self.total_turns = 0
        self.awaiting_suit_choice = False
        self.suit_chooser = None

    @property
    def current_player(self):
        active = [p for p in self.turn_order if p in self.active_players]
        if not active:
            return None
        return active[self.current_turn_idx % len(active)]

    def next_turn(self):
        active = [p for p in self.turn_order if p in self.active_players]
        if not active:
            return
        self.current_turn_idx = (self.current_turn_idx + 1) % len(active)
        self.total_turns += 1

    def top_card(self):
        return self.discard_pile[-1] if self.discard_pile else None

    def is_playable(self, card: Card) -> bool:
        """
        A card is playable if:
          - Under penalty stack: ONLY DOS (2).
          - Otherwise: card rank == top card rank, OR card suit == current active suit.
        This rule applies equally to ALL cards including AS (1) and SIETE (7).
        """
        top = self.top_card()
        if top is None:
            return True
        if self.penalty_stack > 0:
            return card.rank == Rank.DOS
        return (card.rank == top.rank or card.suit == self.current_suit)

    def get_valid_card_indices(self, player: str) -> List[int]:
        return [i for i, c in enumerate(self.hands[player]) if self.is_playable(c)]

    def _reset_deck_if_needed(self):
        """Recycle discard pile into deck when deck is exhausted, keeping the top card."""
        if len(self.deck) == 0 and len(self.discard_pile) > 1:
            top_card = self.discard_pile.pop()
            self.deck = self.discard_pile.copy()
            random.shuffle(self.deck)
            self.discard_pile = [top_card]

    def apply_draw(self, player: str, count: int = 1) -> list:
        drawn = []
        for _ in range(count):
            self._reset_deck_if_needed()
            if self.deck:
                card = self.deck.pop()
                self.hands[player].append(card)
                drawn.append(card.to_dict())
        return drawn

    def apply_play(self, player: str, card_idx: int) -> dict:
        hand = self.hands[player]
        if card_idx >= len(hand):
            return {"valid": False, "error": "invalid_card_index"}
        card = hand[card_idx]
        if not self.is_playable(card):
            return {"valid": False, "error": "card_not_playable"}

        hand.pop(card_idx)
        self.discard_pile.append(card)
        self.current_suit = card.suit

        effect = {}
        if card.rank == Rank.DOS:
            # DOS: stack +2 on next player
            self.penalty_stack += 2
            effect["penalty"] = self.penalty_stack
        elif card.rank == Rank.SIETE:
            # SIETE: player who played it must choose the active suit
            self.awaiting_suit_choice = True
            self.suit_chooser = player
            effect["seven"] = True
        elif card.rank == Rank.AS:
            # AS: skip next player's turn
            self.skip_next = True
            effect["skip"] = True

        return {"valid": True, "card": card.to_dict(), "effect": effect}

    def eliminate_player(self, player: str):
        self.finish_order.append(player)
        self.active_players.remove(player)
        logger.info(f"{player} finished in position {len(self.finish_order)}")

    def player_state_view(self, player: str) -> dict:
        top = self.top_card()
        return {
            "hand": [c.to_dict() for c in self.hands[player]],
            "hand_size": len(self.hands[player]),
            "top_card": top.to_dict() if top else None,
            "current_suit": int(self.current_suit) if self.current_suit is not None else 0,
            "penalty_stack": self.penalty_stack,
            "deck_size": len(self.deck),
            "current_player": self.current_player,
            "active_players": self.active_players,
            "opponents": {p: len(self.hands[p]) for p in self.active_players if p != player},
            "valid_card_indices": self.get_valid_card_indices(player),
            "game_active": self.game_active,
            "total_turns": self.total_turns,
            "finish_order": self.finish_order,
        }

    def spectator_view(self) -> dict:
        """Full board view for spectators: shows all hand sizes, no private cards."""
        top = self.top_card()
        return {
            "spectator": True,
            "top_card": top.to_dict() if top else None,
            "current_suit": int(self.current_suit) if self.current_suit is not None else 0,
            "penalty_stack": self.penalty_stack,
            "deck_size": len(self.deck),
            "current_player": self.current_player,
            "active_players": self.active_players,
            "all_hand_sizes": {p: len(self.hands[p]) for p in self.active_players},
            "game_active": self.game_active,
            "total_turns": self.total_turns,
            "finish_order": self.finish_order,
        }

    def agent_observation(self, player: str) -> dict:
        """Build the observation vector sent to the Q-Learning agent."""
        import numpy as np
        hand = self.hands[player]
        valid_ranks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        rank_to_idx = {r: i for i, r in enumerate(valid_ranks)}

        obs = np.zeros(62, dtype=np.int16)
        for card in hand:
            obs[rank_to_idx.get(int(card.rank), 0)] += 1

        top = self.top_card()
        if top:
            obs[10 + rank_to_idx.get(int(top.rank), 0)] = 1
            obs[20 + int(top.suit)] = 1

        valid_idxs = self.get_valid_card_indices(player)
        for i in valid_idxs:
            if i < 25:
                obs[24 + i] = 1
        draw_action = len(hand)
        if draw_action < 25:
            obs[24 + draw_action] = 1

        other_players = [p for p in self.active_players if p != player]
        if other_players:
            obs[49] = min(len(self.hands[other_players[0]]), 25)
        obs[50] = 1
        obs[51] = min(self.penalty_stack, 10)

        return {
            "observation": obs.tolist(),
            "valid_actions": valid_idxs + [draw_action],
            "hand_size": len(hand),
        }


class MasterAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_state = GameState()
        self.connected_players = set()
        self.game_started = False
        self.round_number = 0
        self.round_results = []
        self.stop_requested = False
        self.watch_mode = False           # human is spectator, not a player
        self.watch_rounds_remaining = 0   # rounds left in watch session

    class RegistrationBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg is None:
                return

            sender = str(msg.sender).split("/")[0]
            performative = msg.get_metadata("performative")

            if performative == "subscribe":
                if sender not in self.agent.connected_players:
                    self.agent.connected_players.add(sender)
                    logger.info(f"Player registered: {sender} ({len(self.agent.connected_players)}/4)")

                reply = Message(to=sender)
                reply.set_metadata("performative", "confirm")
                reply.body = json.dumps({"status": "registered"})
                await self.send(reply)

            elif performative == "command":
                try:
                    data = json.loads(msg.body)
                except Exception:
                    return

                cmd = data.get("command")

                if cmd == "start":
                    if self.agent.game_started:
                        reply = Message(to=sender)
                        reply.set_metadata("performative", "inform")
                        reply.body = json.dumps({"info": "Game is already running."})
                        await self.send(reply)
                        return
                    expected = set(JID_TO_PLAYER.keys())
                    if not expected.issubset(self.agent.connected_players):
                        missing = [JID_TO_PLAYER[j] for j in expected - self.agent.connected_players]
                        reply = Message(to=sender)
                        reply.set_metadata("performative", "inform")
                        reply.body = json.dumps({"info": f"Not all players connected. Missing: {missing}"})
                        await self.send(reply)
                        return
                    self.agent.stop_requested = False
                    self.agent.game_started = True
                    self.agent.round_number += 1
                    logger.info(f"Round {self.agent.round_number} started by human.")
                    await self.agent.start_game(self)

                elif cmd == "stop":
                    self.agent.stop_requested = True
                    self.agent.game_started = False
                    self.agent.game_state.game_active = False
                    self.agent.watch_mode = False
                    self.agent.watch_rounds_remaining = 0
                    logger.info("Stop requested by human.")
                    await self.agent.broadcast_stop(self)

                elif cmd == "watch":
                    rounds = data.get("rounds", 1)
                    if self.agent.game_started:
                        reply = Message(to=sender)
                        reply.set_metadata("performative", "inform")
                        reply.body = json.dumps({"info": "Game is already running."})
                        await self.send(reply)
                        return
                    expected = set(JID_TO_PLAYER.keys())
                    if not expected.issubset(self.agent.connected_players):
                        missing = [JID_TO_PLAYER[j] for j in expected - self.agent.connected_players]
                        reply = Message(to=sender)
                        reply.set_metadata("performative", "inform")
                        reply.body = json.dumps({"info": f"Not all players connected. Missing: {missing}"})
                        await self.send(reply)
                        return
                    self.agent.watch_mode = True
                    self.agent.watch_rounds_remaining = rounds
                    self.agent.stop_requested = False
                    self.agent.game_started = True
                    self.agent.round_number += 1
                    logger.info(f"Watch mode: {rounds} round(s) starting.")
                    await self.agent.start_game(self)

    class ActionBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=2)
            if msg is None:
                return

            performative = msg.get_metadata("performative")
            sender = str(msg.sender).split("/")[0]

            try:
                data = json.loads(msg.body)
            except Exception:
                return

            gs = self.agent.game_state
            if not gs.game_active:
                return

            player = JID_TO_PLAYER.get(sender)
            if player is None:
                return

            # Handle suit choice (sent after playing SIETE)
            if performative == "suit_choice":
                if gs.awaiting_suit_choice and gs.suit_chooser == player:
                    suit = data.get("suit", 0)
                    gs.current_suit = suit
                    gs.awaiting_suit_choice = False
                    gs.suit_chooser = None
                    logger.info(f"{player} chose suit: {suit}")
                    result = {"action": "suit_chosen", "suit": suit, "player": player}
                    await self.agent.broadcast_state(result, self)
                    if gs.skip_next:
                        gs.skip_next = False
                        gs.next_turn()
                    gs.next_turn()
                    await self.agent.request_action(self)
                return

            if performative != "action":
                return

            if player != gs.current_player:
                reply = Message(to=sender)
                reply.set_metadata("performative", "reject")
                reply.body = json.dumps({"error": "not_your_turn", "current_player": gs.current_player})
                await self.send(reply)
                # Re-request from the correct player so the game doesn't freeze
                await self.agent.request_action(self)
                return

            action_type = data.get("action")
            result = {}

            if action_type == "draw":
                if gs.penalty_stack > 0:
                    count = gs.penalty_stack
                    drawn = gs.apply_draw(player, count)
                    gs.penalty_stack = 0
                    result = {"action": "draw", "count": count, "player": player, "drawn": drawn}
                else:
                    drawn = gs.apply_draw(player, 1)
                    result = {"action": "draw", "count": 1, "player": player, "drawn": drawn}

            elif action_type == "play":
                card_idx = data.get("card_index")
                if card_idx is None:
                    reply = Message(to=sender)
                    reply.set_metadata("performative", "reject")
                    reply.body = json.dumps({"error": "missing_card_index"})
                    await self.send(reply)
                    await self.agent.request_action(self)
                    return

                play_result = gs.apply_play(player, card_idx)
                if not play_result["valid"]:
                    reply = Message(to=sender)
                    reply.set_metadata("performative", "reject")
                    reply.body = json.dumps({"error": play_result["error"]})
                    await self.send(reply)
                    await self.agent.request_action(self)
                    return

                result = {
                    "action": "play",
                    "card": play_result["card"],
                    "effect": play_result["effect"],
                    "player": player,
                }

                # SIETE was played: wait for suit choice before advancing turn
                if gs.awaiting_suit_choice:
                    if player == "human":
                        await self.agent.broadcast_state(result, self)
                        await self.agent.request_suit_choice(player, self)
                        return
                    else:
                        # Non-human agents: auto-choose most frequent suit in hand
                        suit = self.agent.auto_choose_suit(gs, player)
                        gs.current_suit = suit
                        gs.awaiting_suit_choice = False
                        gs.suit_chooser = None
                        result["effect"]["chosen_suit"] = suit
                        logger.info(f"{player} auto-chose suit: {suit}")

            else:
                reply = Message(to=sender)
                reply.set_metadata("performative", "reject")
                reply.body = json.dumps({"error": "unknown_action"})
                await self.send(reply)
                await self.agent.request_action(self)
                return

            logger.info(f"Action applied: {player} -> {action_type}")

            # Check if this player finished (empty hand)
            if not gs.hands[player]:
                pos = len(gs.finish_order) + 1
                gs.eliminate_player(player)
                result["finished"] = True
                result["position"] = pos

                await self.agent.broadcast_state(result, self)

                # One player left = loser, round ends
                if len(gs.active_players) == 1:
                    loser = gs.active_players[0]
                    gs.eliminate_player(loser)
                    gs.game_active = False
                    self.agent.game_started = False
                    self.agent.round_results.append({
                        "round": self.agent.round_number,
                        "finish_order": gs.finish_order.copy(),
                        "turns": gs.total_turns,
                    })
                    await self.agent.broadcast_round_over(gs.finish_order.copy(), self)
                    if not self.agent.stop_requested:
                        await asyncio.sleep(3)
                        # Watch mode: count down remaining rounds
                        if self.agent.watch_mode:
                            self.agent.watch_rounds_remaining -= 1
                            if self.agent.watch_rounds_remaining <= 0:
                                self.agent.watch_mode = False
                                await self.agent.broadcast_stop(self)
                                return
                        self.agent.round_number += 1
                        self.agent.game_started = True
                        logger.info(f"Starting round {self.agent.round_number} automatically.")
                        await self.agent.start_game(self)
                    return

                # Advance turn (apply skip if AS was played)
                if gs.skip_next:
                    gs.skip_next = False
                    gs.next_turn()
                gs.next_turn()
                await self.agent.request_action(self)
                return

            # Advance turn
            if gs.skip_next:
                gs.skip_next = False
                gs.next_turn()
            gs.next_turn()

            # Turn limit safety check
            if gs.total_turns > 500:
                gs.game_active = False
                self.agent.game_started = False
                order = gs.finish_order.copy() + gs.active_players.copy()
                self.agent.round_results.append({
                    "round": self.agent.round_number,
                    "finish_order": order,
                    "turns": gs.total_turns,
                })
                await self.agent.broadcast_round_over(order, self)
                if not self.agent.stop_requested:
                    await asyncio.sleep(3)
                    self.agent.round_number += 1
                    self.agent.game_started = True
                    await self.agent.start_game(self)
                return

            await self.agent.broadcast_state(result, self)
            await self.agent.request_action(self)

    def auto_choose_suit(self, gs, player: str) -> int:
        """Non-human agents auto-choose the suit they hold most of."""
        hand = gs.hands[player]
        if not hand:
            return 0
        suit_counts = Counter(int(c.suit) for c in hand)
        return suit_counts.most_common(1)[0][0]

    async def start_game(self, behaviour):
        self.game_state.reset(watch_mode=self.watch_mode)
        logger.info(f"Round {self.round_number} — turn order: {self.game_state.turn_order}")
        await self.broadcast_state({
            "action": "game_start",
            "turn_order": self.game_state.turn_order,
            "round": self.round_number,
            "watch_mode": self.watch_mode,
            "watch_rounds_remaining": self.watch_rounds_remaining,
        }, behaviour)
        await self.request_action(behaviour)

    async def broadcast_state(self, last_action: dict, behaviour):
        for player, jid in PLAYER_TO_JID.items():
            msg = Message(to=jid)
            msg.set_metadata("performative", "inform")
            if self.watch_mode and player == "human":
                # Human is spectator: send full board view without a personal hand
                state = self.game_state.spectator_view()
            else:
                state = self.game_state.player_state_view(player)
            state["last_action"] = last_action
            state["round"] = self.round_number
            state["watch_mode"] = self.watch_mode
            state["watch_rounds_remaining"] = self.watch_rounds_remaining
            msg.body = json.dumps(state)
            await behaviour.send(msg)

    async def request_action(self, behaviour):
        gs = self.game_state
        if not gs.game_active:
            return

        # Iterative loop — skip human in watch mode without recursion.
        # Recursion here would stack-overflow if all active players were
        # somehow skipped (e.g. watch mode with only human left by accident).
        max_skips = len(gs.active_players) + 1
        skips = 0
        while True:
            current = gs.current_player
            if current is None:
                return
            if self.watch_mode and current == "human":
                gs.next_turn()
                skips += 1
                if skips > max_skips:
                    logger.error("request_action: could not find a non-human player to act")
                    return
                continue
            break

        jid = PLAYER_TO_JID[current]
        msg = Message(to=jid)
        msg.set_metadata("performative", "request")

        # Q-Learning agent gets observation vector; all others get full state with hand
        if current == "qagent":
            obs_data = gs.agent_observation(current)
            msg.body = json.dumps({"request": "action", "player": current, **obs_data})
        else:
            state = gs.player_state_view(current)
            state["request"] = "action"
            msg.body = json.dumps(state)

        await behaviour.send(msg)
        logger.info(f"Requested action from: {current}")

    async def request_suit_choice(self, player: str, behaviour):
        jid = PLAYER_TO_JID[player]
        msg = Message(to=jid)
        msg.set_metadata("performative", "request")
        msg.body = json.dumps({"request": "suit_choice"})
        await behaviour.send(msg)
        logger.info(f"Requested suit choice from: {player}")

    async def broadcast_round_over(self, finish_order: list, behaviour):
        for jid in PLAYER_TO_JID.values():
            msg = Message(to=jid)
            msg.set_metadata("performative", "inform")
            msg.body = json.dumps({
                "round_over": True,
                "round": self.round_number,
                "finish_order": finish_order,
                "loser": finish_order[-1] if finish_order else None,
                "all_rounds": self.round_results,
                "stop_requested": self.stop_requested,
                "watch_mode": self.watch_mode,
                "watch_rounds_remaining": self.watch_rounds_remaining,
            })
            await behaviour.send(msg)
        logger.info(f"Round {self.round_number} over. Finish order: {finish_order}")

    async def broadcast_stop(self, behaviour):
        for jid in PLAYER_TO_JID.values():
            msg = Message(to=jid)
            msg.set_metadata("performative", "inform")
            msg.body = json.dumps({
                "game_stopped": True,
                "all_rounds": self.round_results,
                "total_rounds": self.round_number,
            })
            await behaviour.send(msg)
        logger.info("Game stopped. Session report sent.")

    async def setup(self):
        logger.info(f"Master Agent starting: {self.jid}")

        subscribe_template = Template()
        subscribe_template.set_metadata("performative", "subscribe")
        command_template = Template()
        command_template.set_metadata("performative", "command")
        reg_template = subscribe_template | command_template

        action_template = Template()
        action_template.set_metadata("performative", "action")
        suit_template = Template()
        suit_template.set_metadata("performative", "suit_choice")
        action_reg_template = action_template | suit_template

        self.add_behaviour(self.RegistrationBehaviour(), reg_template)
        self.add_behaviour(self.ActionBehaviour(), action_reg_template)


async def main():
    agent = MasterAgent(MASTER_JID, MASTER_PASSWORD)
    await agent.start(auto_register=True)
    logger.info("Master Agent running. Waiting for players.")

    try:
        while True:
            await asyncio.sleep(1)
            if not agent.is_alive():
                break
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())