# Hazz2 Multi-Agent Card Game System

A distributed, containerized multi-agent system where autonomous agents and a human player compete in the Hazz2 card game under strict rule enforcement, with clear separation between learning, inference, and human interaction.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Docker Network: hazz2net                            │
│                                                                             │
│   ┌──────────────┐     XMPP   ┌────────────────────────────────────┐        │
│   │   ejabberd   │◄───────────┤              master_agent          │        │
│   │  XMPP Broker │            │  - Enforces all game rules         │        │
│   │  :5222/:5280 │            │  - Manages turn order              │        │
│   └──────┬───────┘            │  - Validates every action          │        │
│          │                    │  - Routes all messages             │        │
│          │                    │  - Manages rounds & session report │        │
│          │                    └────────────────────────────────────┘        │
│          │                XMPP                                              │
│          └────────────────────────────┐───────────────────────┐             │
│             │                         │                       │             │
│  ┌──────────▼───────────┐  ┌──────────▼─────────┐  ┌──────────▼───────────┐ │
│  │    qlearning_agent   │  │    random_agent    │  │   heuristic_agent    │ │
│  │- Pre-trained Q-table │  │- Random card picks │  │- Frequency heuristic │ │
│  │- Inference only      │  │- No learning       │  └──────────────────────┘ │
│  └──────────────────────┘  └────────────────────┘                           │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │ XMPP :5222 (external)
                                  ┌────────▼──────┐
                                  │  Human Client │
                                  │     (CLI)     │
                                  └───────────────┘
```

### Core Design Principles

- **All messages route exclusively through the Master Agent** — agents never communicate directly with each other
- **Learning is offline only** — the Q-Learning agent loads a pre-trained Q-table and never updates it during gameplay
- **Strict responsibility separation** — Master Agent enforces rules, agents only decide actions, human only inputs commands
- **Human connects externally** — the human client runs outside Docker and connects via XMPP port 5222

---

## Game Rules

The game uses a Spanish card deck (40 cards: 4 suits × 10 ranks).

### Suits
| Value | Name   |
|-------|--------|
| 0     | Coins  |
| 1     | Cups   |
| 2     | Swords |
| 3     | Clubs  |

### Ranks
| Value | Name      |
|-------|-----------|
| 1     | Ace       |
| 2     | Two       |
| 3–6   | Three–Six |
| 7     | Seven     |
| 10    | Jack      |
| 11    | Knight    |
| 12    | King      |

### Playability Rule
A card is playable if:
- Its **rank** matches the top card's rank, **OR**
- Its **suit** matches the current active suit.

Under a **penalty stack**, only **Two (2)** can be played.

This rule applies equally to **all** cards including Ace and Seven.

### Special Card Effects
| Card          | Effect |
|---------------|--------|
| **Two (2)**   | Next player must draw **+2 cards** (stackable — multiple Twos stack the penalty) |
| **Seven (7)** | The player who played it **chooses the active suit** |
| **Ace (1)**   | **Skips** the next player's turn |

### Win / Lose Condition
- First player to **empty their hand** finishes in 1st place (winner).
- The game continues until only **one player remains** — that player is the **loser**.
- Rounds are played automatically back-to-back until the human types `stop`.

### Deck Recycling
When the draw deck is exhausted, all played cards **except the top card** are shuffled and reused as a new deck. The top card remains on the table.

---

## Agents

### Master Agent
- Central coordinator and sole rule enforcer
- Validates every action before applying it
- Manages turn order, penalty stacks, skip logic, and suit choices after Seven
- Broadcasts game state to all players after each action
- Tracks rounds, finish order, and session statistics

### Q-Learning Agent
- Loads a pre-trained Q-table from `models/qtable_improved.pkl`
- **No learning occurs during gameplay** — inference only
- Receives an observation vector and selects the highest Q-value action among valid actions
- Q-table is loaded asynchronously to avoid blocking the XMPP connection

### Random Agent
- Picks a random card from the valid playable cards each turn
- Falls back to draw if no valid card exists

### Heuristic Agent
- Selects the card whose **rank and suit appear most frequently** in its current hand
- Maximises chaining potential by prioritising cards that share rank/suit with others in hand
- Falls back to draw if no valid card exists

### Human Client (CLI)
- Connects externally via XMPP
- Displays full game state after every move with **Last Move** shown first
- Shows drawn cards immediately (e.g. `You drew: Seven of Coins`)
- Prompts for suit choice when Seven is played
- Displays round results and full session report on `stop`

---

## XMPP Message Protocol

### Performatives

| Performative  | Direction             | Description                               |
|---------------|-----------------------|-------------------------------------------|
| `subscribe`   | Agent → Master        | Register and join the session             |
| `confirm`     | Master → Agent        | Registration acknowledged                 |
| `command`     | Human → Master        | Game control (`start` / `stop`)           |
| `inform`      | Master → All          | State broadcast after each action         |
| `request`     | Master → Current      | Request action from the current player    |
| `action`      | Agent → Master        | Submit chosen action                      |
| `suit_choice` | Human → Master        | Choose active suit after playing Seven    |
| `reject`      | Master → Agent        | Action rejected with error reason         |

### Action Payloads

Draw a card:
```json
{"action": "draw"}
```

Play a card:
```json
{"action": "play", "card_index": 2}
```

Choose suit after Seven (human sends as `suit_choice` performative):
```json
{"suit": 1}
```

---

## Directory Structure

```
hazz2_multiagent/
├── docker-compose.yml           ← all services definition
├── register_users.sh            ← manual XMPP user registration script
├── requirements.txt
├── models/
│   └── qtable_improved.pkl      ← place pre-trained Q-table here
├── ejabberd/
│   └── ejabberd.yml             ← ejabberd XMPP server config
├── shared/
│   └── game_env.py              ← Card, Suit, Rank, Hazz2Env, QLearningAgent
├── master_agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── master_agent.py
├── qlearning_agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── qlearning_agent.py
├── random_agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── random_agent.py
├── heuristic_agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── heuristic_agent.py
└── human_client/
    ├── Dockerfile
    ├── requirements.txt
    └── human_client.py
```

---

## Setup & Deployment

### Prerequisites
- Docker and Docker Compose installed
- Pre-trained Q-table file: `qtable_improved.pkl`

### Step 1 — Place the pre-trained model
```bash
cp /path/to/qtable_improved.pkl models/
```

### Step 2 — Start all containers
```bash
docker-compose up -d --build
```

This starts ejabberd, registers all XMPP users automatically via the `register` service, then starts all four agent containers.

### Step 3 — Connect as the human player
```bash
# Install dependencies
pip install spade gymnasium numpy

# Run the human client
XMPP_SERVER=localhost \
HUMAN_JID=human@ejabberd \
HUMAN_PASSWORD=human_pass \
MASTER_JID=master@ejabberd \
python human_client/human_client.py
```

> **Note:** Use `docker attach human_agent` instead if running the human client inside Docker. Detach safely with **Ctrl+P then Ctrl+Q** (not Ctrl+C).

### Step 4 — Play
```
> start        ← start round 1 (requires all 4 agents connected)
> play 2       ← play the card at index 2
> draw         ← draw a card from the deck
> suit 1       ← choose Cups after playing a Seven
> stop         ← stop immediately and display the session report
> help         ← show all commands and card rules
```

---

## Human Client Commands

| Command        | Description                                             |
|----------------|---------------------------------------------------------|
| `start`        | Start the game session (all 4 agents must be connected) |
| `stop`         | Stop immediately and display the full session report    |
| `play <index>` | Play the card at index `index` from your hand           |
| `draw`         | Draw a card from the deck                               |
| `suit <0-3>`   | Choose active suit after playing a Seven (0=Coins 1=Cups 2=Swords 3=Clubs) |
| `help`         | Show all available commands and card rules              |

---

## Session Report

When `stop` is typed, a full session report is displayed:

```
============================================================
  GAME SESSION REPORT
============================================================
  Total Rounds Played: 5

  Results per round:
    Round  1: human > qagent > heuristic > randomagent  (47 turns)
    Round  2: qagent > randomagent > human > heuristic  (63 turns)
    Round  3: heuristic > human > qagent > randomagent  (38 turns)
    ...

  Overall standings:
    human          : 2 win(s), 1 loss(es)
    qagent         : 1 win(s), 0 loss(es)
    randomagent    : 0 win(s), 2 loss(es)
    heuristic      : 2 win(s), 2 loss(es)
============================================================
```

---

## Environment Variables

| Variable              | Default                | Description                        |
|-----------------------|------------------------|------------------------------------|
| `XMPP_SERVER`         | `ejabberd`             | XMPP server hostname               |
| `MASTER_JID`          | `master@ejabberd`      | Master agent JID                   |
| `MASTER_PASSWORD`     | `master_pass`          | Master agent password              |
| `QAGENT_JID`          | `qagent@ejabberd`      | Q-Learning agent JID               |
| `QAGENT_PASSWORD`     | `qagent_pass`          | Q-Learning agent password          |
| `RANDOM_JID`          | `randomagent@ejabberd` | Random agent JID                   |
| `RANDOM_PASSWORD`     | `random_pass`          | Random agent password              |
| `HEURISTIC_JID`       | `heuristic@ejabberd`   | Heuristic agent JID                |
| `HEURISTIC_PASSWORD`  | `heuristic_pass`       | Heuristic agent password           |
| `HUMAN_JID`           | `human@ejabberd`       | Human player JID                   |
| `HUMAN_PASSWORD`      | `human_pass`           | Human player password              |
| `MODELS_PATH`         | `models`               | Path to pre-trained Q-table        |

---

## Monitoring

View logs per container:
```bash
docker logs master_agent       # game events, turn actions, round results
docker logs qlearning_agent    # Q-table loading, action selection
docker logs random_agent       # turn actions
docker logs heuristic_agent    # turn actions
```

Stop all containers:
```bash
docker-compose down
```