import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, '/app/shared')

import numpy as np
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from collections import defaultdict
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s [QAGENT] %(message)s')
logger = logging.getLogger(__name__)

XMPP_SERVER = os.environ.get("XMPP_SERVER", "ejabberd")
QAGENT_JID = os.environ.get("QAGENT_JID", f"qagent@{XMPP_SERVER}")
QAGENT_PASSWORD = os.environ.get("QAGENT_PASSWORD", "qagent_pass")
MASTER_JID = os.environ.get("MASTER_JID", f"master@{XMPP_SERVER}")
MODELS_PATH = os.environ.get("MODELS_PATH", "models")
MODEL_FILE = os.path.join(MODELS_PATH, "qtable_improved.npz")
MODEL_FILE_PKL = os.path.join(MODELS_PATH, "qtable_improved.pkl")


class QLearningAgent:
    """
    Q-Learning agent — inference only, no training during the game.
    Card playability is enforced by the Master Agent.
    This agent only picks the best action from the valid_actions list.
    """

    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(25, dtype=np.float32))

    def _state_to_key(self, obs: np.ndarray) -> tuple:
        return tuple(obs.astype(np.int16).tolist())

    def get_action(self, obs: np.ndarray, valid_actions: List[int]) -> int:
        """Greedy — always pick the highest Q-value among valid actions."""
        state_key = self._state_to_key(obs)
        q_values = self.q_table[state_key]
        masked_q = np.full(25, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def load_npz(self, path: str):
        """
        Load Q-table from numpy .npz format.
        Loads in ~2 seconds vs ~70 seconds for pickle,
        because numpy arrays deserialize directly into contiguous
        memory without reconstructing millions of Python objects.
        """
        data = np.load(path)
        keys = data['keys']    # shape (N, obs_size), dtype int16
        values = data['values'] # shape (N, 25),      dtype float32
        self.q_table = defaultdict(
            lambda: np.zeros(25, dtype=np.float32),
            {tuple(k): v for k, v in zip(keys, values)}
        )
        logger.info(f"Q-table loaded from npz: {len(self.q_table):,} states.")

    def load_pkl(self, path: str):
        """Fallback: load from original pickle format."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        q_table_data = data['q_table'] if isinstance(data, dict) and 'q_table' in data else data
        self.q_table = defaultdict(lambda: np.zeros(25, dtype=np.float32), q_table_data)
        logger.info(f"Q-table loaded from pkl: {len(self.q_table):,} states.")


class QLearningAgentSPADE(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ql_agent = QLearningAgent()
        self.registered = False

    def _load_model(self):
        try:
            if os.path.exists(MODEL_FILE):
                logger.info(f"Loading Q-table from {MODEL_FILE} ...")
                self.ql_agent.load_npz(MODEL_FILE)
            elif os.path.exists(MODEL_FILE_PKL):
                logger.warning(f".npz not found, falling back to pickle {MODEL_FILE_PKL} ...")
                self.ql_agent.load_pkl(MODEL_FILE_PKL)
            else:
                logger.warning("No model file found. Using untrained (random) fallback.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Using untrained fallback.")

    def select_action(self, observation: list, valid_actions: list) -> int:
        if not valid_actions:
            return 0
        obs = np.array(observation, dtype=np.int16)
        return self.ql_agent.get_action(obs, valid_actions)

    class RegisterBehaviour(OneShotBehaviour):
        async def run(self):
            msg = Message(to=MASTER_JID)
            msg.set_metadata("performative", "subscribe")
            msg.body = json.dumps({"player": "qagent", "jid": QAGENT_JID})
            await self.send(msg)
            logger.info("Registration message sent to Master Agent.")

    class LoadModelBehaviour(OneShotBehaviour):
        """
        Loads the Q-table in a background thread so it doesn't block XMPP.
        With .npz format this completes in ~2 seconds, well within ejabberd's
        keepalive window — no more session drops mid-game.
        """
        async def run(self):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.agent._load_model)

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

                observation = data.get("observation", [])
                valid_actions = data.get("valid_actions", [])
                hand_size = data.get("hand_size", 0)

                action = self.agent.select_action(observation, valid_actions)
                logger.info(f"Selected action: {action} from valid: {valid_actions}")

                if action == hand_size:
                    payload = {"action": "draw"}
                else:
                    payload = {"action": "play", "card_index": action}

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
        logger.info(f"Q-Learning Agent starting: {self.jid}")

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
        self.add_behaviour(self.LoadModelBehaviour())


async def main():
    agent = QLearningAgentSPADE(QAGENT_JID, QAGENT_PASSWORD)
    await agent.start(auto_register=True)
    logger.info("Q-Learning Agent running.")

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