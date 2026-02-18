import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, '/app/shared')

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

logging.basicConfig(level=logging.INFO, format='%(asctime)s [HUMAN] %(message)s')
logger = logging.getLogger(__name__)

XMPP_SERVER = os.environ.get("XMPP_SERVER", "ejabberd")
HUMAN_JID = os.environ.get("HUMAN_JID", f"human@{XMPP_SERVER}")
HUMAN_PASSWORD = os.environ.get("HUMAN_PASSWORD", "human_pass")
MASTER_JID = os.environ.get("MASTER_JID", f"master@{XMPP_SERVER}")

SUIT_NAMES = {0: "Coins", 1: "Cups", 2: "Swords", 3: "Clubs"}
RANK_NAMES = {1: "Ace", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
              6: "Six", 7: "Seven", 10: "Jack", 11: "Knight", 12: "King"}


def format_card(card_dict):
    rank = card_dict["rank"]
    suit = card_dict["suit"]
    return f"{RANK_NAMES.get(rank, rank)} of {SUIT_NAMES.get(suit, suit)}"


def display_state(data: dict):
    print("\n" + "=" * 60)
    print(f"  Current Turn: {data.get('current_player', '?').upper()}")
    print("=" * 60)
    top = data.get("top_card")
    if top:
        print(f"  Top Card:     {format_card(top)}")
    print(f"  Active Suit:  {SUIT_NAMES.get(data.get('current_suit', 0), '?')}")
    penalty = data.get("penalty_stack", 0)
    if penalty:
        print(f"  Penalty:      {penalty} cards stacked")
    print(f"  Deck:         {data.get('deck_size', 0)} cards remaining")
    print(f"  Turn:         {data.get('total_turns', 0)}")
    print("\n  Opponents:")
    for opp, count in data.get("opponents", {}).items():
        print(f"    {opp}: {count} cards")
    hand = data.get("hand", [])
    valid = data.get("valid_card_indices", [])
    print(f"\n  Your Hand ({len(hand)} cards):")
    for i, card in enumerate(hand):
        marker = " [playable]" if i in valid else ""
        print(f"    {i}: {format_card(card)}{marker}")
    last = data.get("last_action", {})
    if last and last.get("action") and last.get("action") != "game_start":
        player = last.get("player", "?")
        action = last.get("action")
        if action == "play":
            print(f"\n  Last Move: {player} played {format_card(last['card'])}")
        elif action == "draw":
            print(f"\n  Last Move: {player} drew {last.get('count', 1)} card(s)")
    print("=" * 60)


def print_help():
    print("\n  Commands:")
    print("    start        - Start the game")
    print("    stop         - Stop the game")
    print("    play <index> - Play a card by index")
    print("    draw         - Draw a card")
    print("    help         - Show this help")
    print()


class HumanClientAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registered = False
        self.my_turn = False
        self.current_state = None

    class RegisterBehaviour(OneShotBehaviour):
        async def run(self):
            msg = Message(to=MASTER_JID)
            msg.set_metadata("performative", "subscribe")
            msg.body = json.dumps({"player": "human", "jid": HUMAN_JID})
            await self.send(msg)

    class InputBehaviour(CyclicBehaviour):
        async def run(self):
            loop = asyncio.get_event_loop()
            try:
                line = await loop.run_in_executor(None, lambda: input("> ").strip().lower())
            except (EOFError, KeyboardInterrupt):
                return

            if not line:
                return

            if line == "help":
                print_help()
                return

            if line in ("start", "stop"):
                msg = Message(to=MASTER_JID)
                msg.set_metadata("performative", "command")
                msg.body = json.dumps({"command": line})
                await self.send(msg)
                return

            if not self.agent.my_turn:
                print("  Not your turn.")
                return

            if line == "draw":
                msg = Message(to=MASTER_JID)
                msg.set_metadata("performative", "action")
                msg.body = json.dumps({"action": "draw"})
                await self.send(msg)
                self.agent.my_turn = False

            elif line.startswith("play"):
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    print("  Usage: play <index>")
                    return
                idx = int(parts[1])
                state = self.agent.current_state
                if state:
                    valid = state.get("valid_card_indices", [])
                    if idx not in valid:
                        print(f"  Card {idx} is not playable. Valid indices: {valid}")
                        return
                msg = Message(to=MASTER_JID)
                msg.set_metadata("performative", "action")
                msg.body = json.dumps({"action": "play", "card_index": idx})
                await self.send(msg)
                self.agent.my_turn = False
            else:
                print("  Unknown command. Type 'help' for options.")

    class GameBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg is None:
                return

            performative = msg.get_metadata("performative")

            if performative == "confirm":
                if not self.agent.registered:
                    self.agent.registered = True
                    print("\n  Connected to Master Agent.")
                    print("  All agents must be connected before starting.")
                    print("  Type 'start' to begin the game or 'help' for commands.")

            elif performative == "inform":
                try:
                    data = json.loads(msg.body)
                except Exception:
                    return

                if data.get("game_over"):
                    winner = data.get("winner", "unknown")
                    print("\n" + "=" * 60)
                    if winner == "human":
                        print("  YOU WIN!")
                    elif winner == "stopped":
                        print("  Game stopped.")
                    elif winner == "draw":
                        print("  DRAW - turn limit reached.")
                    else:
                        print(f"  {winner.upper()} WINS.")
                    print("  Type 'start' to play again.")
                    print("=" * 60)
                    self.agent.my_turn = False
                elif data.get("info"):
                    print(f"\n  [Info] {data['info']}")
                else:
                    last = data.get("last_action", {})
                    if last.get("action") == "game_start":
                        order = last.get("turn_order", [])
                        print(f"\n  Game started! Turn order: {' -> '.join(order)}")
                    else:
                        display_state(data)

            elif performative == "request":
                try:
                    data = json.loads(msg.body)
                except Exception:
                    return
                if data.get("request") != "action":
                    return

                self.agent.current_state = data
                self.agent.my_turn = True
                display_state(data)
                print("  YOUR TURN — play <index> or draw")

            elif performative == "reject":
                try:
                    data = json.loads(msg.body)
                    print(f"\n  [Rejected] {data.get('error', 'unknown error')}")
                    self.agent.my_turn = True
                except Exception:
                    pass

    async def setup(self):
        confirm_template = Template()
        confirm_template.set_metadata("performative", "confirm")
        inform_template = Template()
        inform_template.set_metadata("performative", "inform")
        request_template = Template()
        request_template.set_metadata("performative", "request")
        reject_template = Template()
        reject_template.set_metadata("performative", "reject")
        game_template = confirm_template | inform_template | request_template | reject_template

        self.add_behaviour(self.RegisterBehaviour())
        self.add_behaviour(self.GameBehaviour(), game_template)
        self.add_behaviour(self.InputBehaviour())


async def main():
    print("=" * 60)
    print("  HAZZ2 Card Game - Human Client")
    print("=" * 60)
    print(f"  Connecting to {XMPP_SERVER}...")

    agent = HumanClientAgent(HUMAN_JID, HUMAN_PASSWORD)
    await agent.start(auto_register=True)

    try:
        while True:
            await asyncio.sleep(1)
            if not agent.is_alive():
                break
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())