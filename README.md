# Hazz2 Multi-Agent System

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network: hazz2net                  │
│                                                                   │
│  ┌─────────────┐    XMPP     ┌──────────────────────────────┐   │
│  │  ejabberd   │◄────────────┤       master_agent           │   │
│  │  (XMPP)     │             │  - Enforces game rules        │   │
│  │  :5222/:5280│             │  - Manages turn order         │   │
│  └──────┬──────┘             │  - Validates actions          │   │
│         │                    │  - Routes all messages        │   │
│         │ XMPP               └──────────────────────────────┘   │
│    ┌────┴────────────┐                                           │
│    │                 │                                           │
│  ┌─▼──────────┐  ┌──▼──────────────┐                           │
│  │qlearning   │  │  random_agent   │                           │
│  │_agent      │  │  - Random picks  │                           │
│  │- Loads     │  │  - No learning   │                           │
│  │  Q-table   │  └─────────────────┘                           │
│  │- Inference │                                                   │
│  │  only      │                                                   │
│  └────────────┘                                                   │
└──────────────────────────────────┬──────────────────────────────┘
                                   │ XMPP :5222 (external)
                              ┌────▼────┐
                              │  Human  │
                              │  Client │
                              │  (CLI)  │
                              └─────────┘
```

### Communication Rules
- All messages route through Master Agent only
- Agents never communicate directly with each other
- Human client connects externally via XMPP port 5222
- Master Agent is the sole rule enforcer and game controller

### Message Flow

```
Human/Agent ──[action]──► Master Agent ──[validate]──► apply
Master Agent ──[state]───► All Players
Master Agent ──[request]──► Current Player
```

## Setup

### 1. Place pre-trained model
```bash
cp /path/to/qtable_improved.pkl models/
```

### 2. Start server containers
```bash
docker-compose up -d
```

### 3. Register XMPP users
```bash
chmod +x register_users.sh
./register_users.sh
```

### 4. Connect as human player
```bash
# Install dependencies
pip install spade gymnasium numpy

# Run client
XMPP_SERVER=localhost \
HUMAN_JID=human@ejabberd \
HUMAN_PASSWORD=human_pass \
MASTER_JID=master@ejabberd \
python human_client/human_client.py
```

## Environment Variables

| Variable          | Default                  | Description                  |
|-------------------|--------------------------|------------------------------|
| XMPP_SERVER       | ejabberd                 | XMPP hostname                |
| MASTER_JID        | master@ejabberd          | Master agent JID             |
| MASTER_PASSWORD   | master_pass              | Master agent password        |
| QAGENT_JID        | qagent@ejabberd          | Q-Learning agent JID         |
| QAGENT_PASSWORD   | qagent_pass              | Q-Learning agent password    |
| RANDOM_JID        | randomagent@ejabberd     | Random agent JID             |
| RANDOM_PASSWORD   | random_pass              | Random agent password        |
| HUMAN_JID         | human@ejabberd           | Human player JID             |
| HUMAN_PASSWORD    | human_pass               | Human player password        |
| MODELS_PATH       | models                   | Path to pre-trained models   |

## Directory Structure

```
hazz2_multiagent/
├── docker-compose.yml
├── requirements.txt
├── register_users.sh
├── models/
│   └── qtable_improved.pkl      ← place pre-trained model here
├── ejabberd/
│   └── ejabberd.yml
├── shared/
│   └── game_env.py              ← shared card/game/agent definitions
├── master_agent/
│   ├── Dockerfile
│   └── master_agent.py
├── qlearning_agent/
│   ├── Dockerfile
│   └── qlearning_agent.py
├── random_agent/
│   ├── Dockerfile
│   └── random_agent.py
└── human_client/
    ├── Dockerfile
    └── human_client.py
```

## XMPP Message Protocol

### Performatives

| Performative | Direction            | Description                          |
|--------------|----------------------|--------------------------------------|
| subscribe    | Agent → Master       | Register/join the game               |
| confirm      | Master → Agent       | Registration acknowledged            |
| inform       | Master → Agent       | Game state broadcast                 |
| request      | Master → Agent       | Request an action from current player|
| action       | Agent → Master       | Submit chosen action                 |
| reject       | Master → Agent       | Action rejected with error reason    |

### Action Payload (Agent → Master)

Draw:
```json
{"action": "draw"}
```

Play card:
```json
{"action": "play", "card_index": 2}
```

## Game Rules Enforced by Master Agent

- Only the current player may submit an action
- Card must match top card by rank, suit, or be an As
- During a penalty stack, only Dos cards or drawing the penalty is valid
- Siete skips the next player's turn
- Dos stacks penalty (+2 per Dos played)
- First player to empty their hand wins
- Game ends in draw after 300 turns
