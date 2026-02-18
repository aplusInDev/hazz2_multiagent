import asyncio
import json
import logging
import random
import sys
import os

sys.path.insert(0, '/app/shared')

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from game_env import Hazz2Env, Card, Suit, Rank

logging.basicConfig(level=logging.INFO, format='%(asctime)s [MASTER] %(message)s')
logger = logging.getLogger(__name__)

XMPP_SERVER = os.environ.get("XMPP_SERVER", "ejabberd")
MASTER_JID = os.environ.get("MASTER_JID", f"master@{XMPP_SERVER}")
MASTER_PASSWORD = os.environ.get("MASTER_PASSWORD", "master_pass")
QAGENT_JID = os.environ.get("QAGENT_JID", f"qagent@{XMPP_SERVER}")
RANDOM_JID = os.environ.get("RANDOM_JID", f"randomagent@{XMPP_SERVER}")
HUMAN_JID = os.environ.get("HUMAN_JID", f"human@{XMPP_SERVER}")

PLAYERS = ["human", "qagent", "randomagent"]

JID_TO_PLAYER = {
    HUMAN_JID: "human",
    QAGENT_JID: "qagent",
    RANDOM_JID: "randomagent",
}

PLAYER_TO_JID = {v: k for k, v in JID_TO_PLAYER.items()}


class GameState:
    def __init__(self):
        self.env = Hazz2Env()
        self.hands = {}
        self.turn_order = []
        self.current_turn_idx = 0
        self.game_active = False
        self.penalty_stack = 0
        self.skip_next = False
        self.winner = None
        self.total_turns = 0
        self.deck = []
        self.discard_pile = []
        self.current_suit = None

    def reset(self):
        deck = self.env.full_deck.copy()
        random.shuffle(deck)

        self.hands = {p: [] for p in PLAYERS}
        for _ in range(4):
            for p in PLAYERS:
                self.hands[p].append(deck.pop())

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
        self.winner = None
        self.turn_order = PLAYERS.copy()
        random.shuffle(self.turn_order)
        self.current_turn_idx = 0
        self.total_turns = 0

    @property
    def current_player(self):
        return self.turn_order[self.current_turn_idx]

    def next_turn(self):
        self.current_turn_idx = (self.current_turn_idx + 1) % len(self.turn_order)
        self.total_turns += 1

    def top_card(self):
        return self.discard_pile[-1] if self.discard_pile else None

    def is_playable(self, card: Card) -> bool:
        top = self.top_card()
        if top is None:
            return True
        if self.penalty_stack > 0:
            return card.rank == Rank.DOS
        return (card.rank == top.rank or card.suit == self.current_suit or
                card.rank == Rank.AS)

    def get_valid_card_indices(self, player: str):
        return [i for i, c in enumerate(self.hands[player]) if self.is_playable(c)]

    def apply_draw(self, player: str, count: int = 1):
        for _ in range(count):
            if self.deck:
                self.hands[player].append(self.deck.pop())

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
            self.penalty_stack += 2
            effect["penalty"] = self.penalty_stack
        elif card.rank == Rank.AS:
            effect["ace"] = True
        elif card.rank == Rank.SIETE:
            self.skip_next = True
            effect["skip"] = True

        return {"valid": True, "card": card.to_dict(), "effect": effect}

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
            "turn_order": self.turn_order,
            "opponents": {p: len(self.hands[p]) for p in PLAYERS if p != player},
            "valid_card_indices": self.get_valid_card_indices(player),
            "game_active": self.game_active,
            "total_turns": self.total_turns,
        }

    def agent_observation(self, player: str) -> dict:
        import numpy as np
        hand = self.hands[player]
        valid_ranks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        rank_to_idx = {r: i for i, r in enumerate(valid_ranks)}

        obs = np.zeros(62, dtype=np.int16)
        for card in hand:
            obs[rank_to_idx.get(card.rank, 0)] += 1

        top = self.top_card()
        if top:
            obs[10 + rank_to_idx.get(top.rank, 0)] = 1
            obs[20 + top.suit] = 1

        valid_idxs = self.get_valid_card_indices(player)
        for i in valid_idxs:
            if i < 25:
                obs[24 + i] = 1
        draw_action = len(hand)
        if draw_action < 25:
            obs[24 + draw_action] = 1

        other_players = [p for p in PLAYERS if p != player]
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
                    logger.info(f"Player registered: {sender} ({len(self.agent.connected_players)}/3)")

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
                    self.agent.game_started = True
                    logger.info("Game started by human command.")
                    await self.agent.start_game(self)

                elif cmd == "stop":
                    if not self.agent.game_started:
                        reply = Message(to=sender)
                        reply.set_metadata("performative", "inform")
                        reply.body = json.dumps({"info": "No game is currently running."})
                        await self.send(reply)
                        return
                    self.agent.game_started = False
                    self.agent.game_state.game_active = False
                    logger.info("Game stopped by human command.")
                    await self.agent.broadcast_game_over("stopped", self)

    class ActionBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=2)
            if msg is None:
                return
            if msg.get_metadata("performative") != "action":
                return

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

            if player != gs.current_player:
                reply = Message(to=sender)
                reply.set_metadata("performative", "reject")
                reply.body = json.dumps({"error": "not_your_turn", "current_player": gs.current_player})
                await self.send(reply)
                return

            action_type = data.get("action")
            result = {}

            if action_type == "draw":
                if gs.penalty_stack > 0:
                    count = gs.penalty_stack
                    gs.apply_draw(player, count)
                    gs.penalty_stack = 0
                    result = {"action": "draw", "count": count, "player": player}
                else:
                    gs.apply_draw(player, 1)
                    result = {"action": "draw", "count": 1, "player": player}

            elif action_type == "play":
                card_idx = data.get("card_index")
                if card_idx is None:
                    reply = Message(to=sender)
                    reply.set_metadata("performative", "reject")
                    reply.body = json.dumps({"error": "missing_card_index"})
                    await self.send(reply)
                    return

                play_result = gs.apply_play(player, card_idx)
                if not play_result["valid"]:
                    reply = Message(to=sender)
                    reply.set_metadata("performative", "reject")
                    reply.body = json.dumps({"error": play_result["error"]})
                    await self.send(reply)
                    return

                result = {
                    "action": "play",
                    "card": play_result["card"],
                    "effect": play_result["effect"],
                    "player": player,
                }
            else:
                reply = Message(to=sender)
                reply.set_metadata("performative", "reject")
                reply.body = json.dumps({"error": "unknown_action"})
                await self.send(reply)
                return

            logger.info(f"Action: {player} -> {action_type}")

            if not gs.hands[player]:
                gs.game_active = False
                self.agent.game_started = False
                await self.agent.broadcast_game_over(player, self)
                return

            if gs.skip_next:
                gs.skip_next = False
                gs.next_turn()

            gs.next_turn()

            if gs.total_turns > 300:
                gs.game_active = False
                self.agent.game_started = False
                await self.agent.broadcast_game_over("draw", self)
                return

            await self.agent.broadcast_state(result, self)
            await self.agent.request_action(self)

    async def start_game(self, behaviour):
        self.game_state.reset()
        logger.info(f"Turn order: {self.game_state.turn_order}")
        await self.broadcast_state({"action": "game_start", "turn_order": self.game_state.turn_order}, behaviour)
        await self.request_action(behaviour)

    async def broadcast_state(self, last_action: dict, behaviour):
        for player, jid in PLAYER_TO_JID.items():
            msg = Message(to=jid)
            msg.set_metadata("performative", "inform")
            state = self.game_state.player_state_view(player)
            state["last_action"] = last_action
            msg.body = json.dumps(state)
            await behaviour.send(msg)

    async def request_action(self, behaviour):
        gs = self.game_state
        if not gs.game_active:
            return

        current = gs.current_player
        jid = PLAYER_TO_JID[current]
        msg = Message(to=jid)
        msg.set_metadata("performative", "request")

        if current in ("qagent", "randomagent"):
            obs_data = gs.agent_observation(current)
            msg.body = json.dumps({"request": "action", "player": current, **obs_data})
        else:
            state = gs.player_state_view(current)
            state["request"] = "action"
            msg.body = json.dumps(state)

        await behaviour.send(msg)
        logger.info(f"Requested action from: {current}")

    async def broadcast_game_over(self, winner: str, behaviour):
        for jid in PLAYER_TO_JID.values():
            msg = Message(to=jid)
            msg.set_metadata("performative", "inform")
            msg.body = json.dumps({"game_over": True, "winner": winner})
            await behaviour.send(msg)
        logger.info(f"Game over. Winner: {winner}")

    async def setup(self):
        logger.info(f"Master Agent starting: {self.jid}")

        subscribe_template = Template()
        subscribe_template.set_metadata("performative", "subscribe")
        command_template = Template()
        command_template.set_metadata("performative", "command")
        reg_template = subscribe_template | command_template

        action_template = Template()
        action_template.set_metadata("performative", "action")

        self.add_behaviour(self.RegistrationBehaviour(), reg_template)
        self.add_behaviour(self.ActionBehaviour(), action_template)


async def main():
    agent = MasterAgent(MASTER_JID, MASTER_PASSWORD)
    await agent.start(auto_register=True)
    logger.info("Master Agent running. Waiting for players to connect.")

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