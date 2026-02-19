import asyncio
import json
import logging
import os
import random
import sys

sys.path.insert(0, '/app/shared')

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

logging.basicConfig(level=logging.INFO, format='%(asctime)s [RANDOM] %(message)s')
logger = logging.getLogger(__name__)

XMPP_SERVER = os.environ.get("XMPP_SERVER", "ejabberd")
RANDOM_JID = os.environ.get("RANDOM_JID", f"randomagent@{XMPP_SERVER}")
RANDOM_PASSWORD = os.environ.get("RANDOM_PASSWORD", "random_pass")
MASTER_JID = os.environ.get("MASTER_JID", f"master@{XMPP_SERVER}")

# Card playability is enforced by the Master Agent.
# This agent only picks a random card from the valid_card_indices list.
# Special card effects (Two=penalty, Seven=suit choice, Ace=skip)
# are also handled entirely by the Master Agent.


class RandomAgentSPADE(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registered = False

    class RegisterBehaviour(OneShotBehaviour):
        async def run(self):
            msg = Message(to=MASTER_JID)
            msg.set_metadata("performative", "subscribe")
            msg.body = json.dumps({"player": "randomagent", "jid": RANDOM_JID})
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

                valid_card_indices = data.get("valid_card_indices", [])

                if not valid_card_indices:
                    payload = {"action": "draw"}
                else:
                    idx = random.choice(valid_card_indices)
                    payload = {"action": "play", "card_index": idx}

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
        logger.info(f"Random Agent starting: {self.jid}")

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
    agent = RandomAgentSPADE(RANDOM_JID, RANDOM_PASSWORD)
    await agent.start(auto_register=True)
    logger.info("Random Agent running.")

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