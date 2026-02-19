import asyncio
import json
import logging
import os
import sys
from collections import Counter

sys.path.insert(0, '/app/shared')

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

logging.basicConfig(level=logging.INFO, format='%(asctime)s [HEURISTIC] %(message)s')
logger = logging.getLogger(__name__)

XMPP_SERVER = os.environ.get("XMPP_SERVER", "ejabberd")
HEURISTIC_JID = os.environ.get("HEURISTIC_JID", f"heuristic@{XMPP_SERVER}")
HEURISTIC_PASSWORD = os.environ.get("HEURISTIC_PASSWORD", "heuristic_pass")
MASTER_JID = os.environ.get("MASTER_JID", f"master@{XMPP_SERVER}")

# Card playability is enforced by the Master Agent.
# This agent receives valid_card_indices and picks the one with
# the highest frequency score (most common rank + most common suit in hand).
#
# Special card effects (Two=penalty, Seven=suit choice, Ace=skip)
# are handled entirely by the Master Agent.
# Suit choice after playing Seven is also handled by the Master Agent
# (auto-selects most frequent suit in this agent's hand).


def select_heuristic_action(hand: list, valid_card_indices: list) -> dict:
    """
    Heuristic: play the card whose rank and suit appear most frequently
    in the agent's current hand. Falls back to draw if no valid cards.
    """
    if not valid_card_indices:
        return {"action": "draw"}

    rank_counts = Counter(c["rank"] for c in hand)
    suit_counts = Counter(c["suit"] for c in hand)

    def card_score(card: dict) -> int:
        return rank_counts[card["rank"]] + suit_counts[card["suit"]]

    best_idx = max(valid_card_indices, key=lambda i: card_score(hand[i]))
    return {"action": "play", "card_index": best_idx}


class HeuristicAgentSPADE(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registered = False

    class RegisterBehaviour(OneShotBehaviour):
        async def run(self):
            msg = Message(to=MASTER_JID)
            msg.set_metadata("performative", "subscribe")
            msg.body = json.dumps({"player": "heuristic", "jid": HEURISTIC_JID})
            await self.send(msg)
            logger.info("Registration message sent to Master Agent.")

    class GameBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg is None:
                return

            performative = msg.get_metadata("performative")

            if performative == "confirm":
                if not self.agent.registered:
                    self.agent.registered = True
                    logger.info("Registered with Master Agent. Waiting for game start command.")

            elif performative == "inform":
                try:
                    data = json.loads(msg.body)
                except Exception:
                    return
                if data.get("game_stopped"):
                    logger.info("Game session stopped.")
                elif data.get("round_over"):
                    order = data.get("finish_order", [])
                    logger.info(f"Round {data.get('round')} over. Order: {order}. Loser: {data.get('loser')}")
                else:
                    last = data.get("last_action", {})
                    if last.get("action") and last.get("action") != "game_start":
                        logger.info(f"Round {data.get('round')} — {last.get('player')} -> {last.get('action')}")

            elif performative == "request":
                try:
                    data = json.loads(msg.body)
                except Exception:
                    return
                if data.get("request") != "action":
                    return

                hand = data.get("hand", [])
                valid_card_indices = data.get("valid_card_indices", [])

                payload = select_heuristic_action(hand, valid_card_indices)

                reply = Message(to=MASTER_JID)
                reply.set_metadata("performative", "action")
                reply.body = json.dumps(payload)
                await self.send(reply)
                logger.info(f"Action sent: {payload}")

            elif performative == "reject":
                try:
                    data = json.loads(msg.body)
                    logger.warning(f"Action rejected: {data.get('error')}")
                except Exception:
                    pass

    async def setup(self):
        logger.info(f"Heuristic Agent starting: {self.jid}")

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


async def main():
    agent = HeuristicAgentSPADE(HEURISTIC_JID, HEURISTIC_PASSWORD)
    await agent.start(auto_register=True)
    logger.info("Heuristic Agent running.")

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
