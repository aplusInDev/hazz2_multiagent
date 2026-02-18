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
RANK_NAMES = {
    1: "Ace", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 10: "Jack", 11: "Knight", 12: "King"
}

# ============================================================
# GAME RULES (Hazz2):
#   A card is playable if its rank OR suit matches the top card.
#   Under penalty stack, ONLY Two (2) can be played.
#   Special effects:
#     Two   (2) -> next player draws +2 (stackable)
#     Seven (7) -> you choose the active suit after playing it
#     Ace   (1) -> next player's turn is skipped
# ============================================================


def format_card(card_dict):
    rank = card_dict.get("rank_name") or RANK_NAMES.get(card_dict["rank"], str(card_dict["rank"]))
    suit = card_dict.get("suit_name") or SUIT_NAMES.get(card_dict["suit"], str(card_dict["suit"]))
    return f"{rank} of {suit}"


def display_state(data: dict):
    last = data.get("last_action", {})
    print("\n" + "=" * 60)

    # Last move always shown first
    if last and last.get("action") and last.get("action") != "game_start":
        player = last.get("player", "?")
        action = last.get("action")
        if action == "play":
            effect = last.get("effect", {})
            card_str = format_card(last["card"])
            extras = ""
            if effect.get("penalty"):
                extras = f" [penalty stack: {effect['penalty']}]"
            elif effect.get("skip"):
                extras = " [next player skipped]"
            elif effect.get("seven"):
                suit = effect.get("chosen_suit")
                if suit is not None:
                    extras = f" [suit changed to {SUIT_NAMES.get(suit, suit)}]"
            print(f"  Last Move: {player} played {card_str}{extras}")
        elif action == "draw":
            drawn = last.get("drawn", [])
            if player == "human" and drawn:
                drawn_str = ", ".join(format_card(c) for c in drawn)
                print(f"  Last Move: You drew: {drawn_str}")
            else:
                print(f"  Last Move: {player} drew {last.get('count', 1)} card(s)")
        elif action == "suit_chosen":
            suit = last.get("suit", 0)
            print(f"  Last Move: {player} chose suit {SUIT_NAMES.get(suit, suit)}")
        if last.get("finished"):
            print(f"  >>> {player} finished in position {last.get('position')} <<<")

    print("-" * 60)
    print(f"  Round:        {data.get('round', '?')}  |  Turn: {data.get('total_turns', 0)}")
    print(f"  Current Turn: {data.get('current_player', '?').upper()}")
    print(f"  Active:       {', '.join(data.get('active_players', []))}")

    top = data.get("top_card")
    if top:
        print(f"  Top Card:     {format_card(top)}")
    print(f"  Active Suit:  {SUIT_NAMES.get(data.get('current_suit', 0), '?')}")

    penalty = data.get("penalty_stack", 0)
    if penalty:
        print(f"  Penalty:      {penalty} cards stacked! (you must play Two or draw)")

    print(f"  Deck:         {data.get('deck_size', 0)} cards remaining")

    finished = data.get("finish_order", [])
    if finished:
        print(f"  Finished:     {' > '.join(f'{i+1}.{p}' for i, p in enumerate(finished))}")

    print("\n  Opponents:")
    for opp, count in data.get("opponents", {}).items():
        print(f"    {opp}: {count} cards")

    hand = data.get("hand", [])
    valid = data.get("valid_card_indices", [])
    print(f"\n  Your Hand ({len(hand)} cards):")
    for i, card in enumerate(hand):
        marker = " [playable]" if i in valid else ""
        print(f"    {i}: {format_card(card)}{marker}")

    print("=" * 60)


def display_round_over(data: dict):
    order = data.get("finish_order", [])
    round_num = data.get("round", "?")
    print("\n" + "=" * 60)
    print(f"  ROUND {round_num} OVER")
    print("=" * 60)
    for i, player in enumerate(order):
        if i == 0:
            label = "1st (WINNER)"
        elif i == len(order) - 1:
            label = f"{i+1}th (LOSER)"
        else:
            label = f"{i+1}nd/rd"
        print(f"  {label}: {player}")
    print("=" * 60)
    if not data.get("stop_requested"):
        print("  Next round starting in 3 seconds...")
        print("  Type 'stop' to end the session and see the full report.")


def display_report(data: dict):
    rounds = data.get("all_rounds", [])
    total = data.get("total_rounds", 0)
    print("\n" + "=" * 60)
    print("  GAME SESSION REPORT")
    print("=" * 60)
    print(f"  Total Rounds Played: {total}")
    print()

    win_counts = {}
    lose_counts = {}
    for r in rounds:
        order = r.get("finish_order", [])
        if order:
            win_counts[order[0]] = win_counts.get(order[0], 0) + 1
            lose_counts[order[-1]] = lose_counts.get(order[-1], 0) + 1

    print("  Results per round:")
    for r in rounds:
        order = r.get("finish_order", [])
        turns = r.get("turns", 0)
        order_str = " > ".join(order)
        print(f"    Round {r['round']:2d}: {order_str}  ({turns} turns)")

    print()
    print("  Overall standings:")
    for p in ["human", "qagent", "randomagent", "heuristic"]:
        wins = win_counts.get(p, 0)
        losses = lose_counts.get(p, 0)
        print(f"    {p:15s}: {wins} win(s), {losses} loss(es)")
    print("=" * 60)


def print_help():
    print("\n  Commands:")
    print("    start          - Start the game session")
    print("    stop           - Stop immediately and show session report")
    print("    play <index>   - Play a card by its index number")
    print("    draw           - Draw a card from the deck")
    print("    suit <0-3>     - Choose suit after playing Seven")
    print("                     0=Coins  1=Cups  2=Swords  3=Clubs")
    print("    help           - Show this help message")
    print()
    print("  Card rules:")
    print("    Two   (2) -> next player draws +2 (stackable)")
    print("    Seven (7) -> you choose the active suit")
    print("    Ace   (1) -> next player's turn is skipped")
    print("    All cards playable only on matching rank or suit")
    print()


class HumanClientAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registered = False
        self.my_turn = False
        self.awaiting_suit = False
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

            if line.startswith("suit"):
                if not self.agent.awaiting_suit:
                    print("  No suit choice needed right now.")
                    return
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit() or int(parts[1]) not in range(4):
                    print("  Usage: suit <0-3>  (0=Coins  1=Cups  2=Swords  3=Clubs)")
                    return
                suit = int(parts[1])
                msg = Message(to=MASTER_JID)
                msg.set_metadata("performative", "suit_choice")
                msg.body = json.dumps({"suit": suit})
                await self.send(msg)
                self.agent.awaiting_suit = False
                print(f"  Suit chosen: {['Coins', 'Cups', 'Swords', 'Clubs'][suit]}")
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
                    print("  Type 'start' to begin or 'help' for commands.")

            elif performative == "inform":
                try:
                    data = json.loads(msg.body)
                except Exception:
                    return

                if data.get("game_stopped"):
                    display_report(data)
                    return

                if data.get("round_over"):
                    display_round_over(data)
                    return

                if data.get("info"):
                    print(f"\n  [Info] {data['info']}")
                    return

                last = data.get("last_action", {})
                if last.get("action") == "game_start":
                    round_num = data.get("round", "?")
                    order = last.get("turn_order", [])
                    print(f"\n  Round {round_num} started! Turn order: {' -> '.join(order)}")
                else:
                    display_state(data)

            elif performative == "request":
                try:
                    data = json.loads(msg.body)
                except Exception:
                    return

                if data.get("request") == "suit_choice":
                    self.agent.awaiting_suit = True
                    print("\n  You played a Seven! Choose the active suit:")
                    print("  suit 0=Coins  suit 1=Cups  suit 2=Swords  suit 3=Clubs")
                    return

                if data.get("request") != "action":
                    return

                self.agent.current_state = data
                self.agent.my_turn = True
                display_state(data)
                print("  YOUR TURN — play <index>, draw, or help")

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