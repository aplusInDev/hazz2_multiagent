# Hazz2 Multi-Agent Card Game System

A distributed, containerized multi-agent system where autonomous agents and a human player compete in the Hazz2 card game under strict rule enforcement, with clear separation between learning, inference, and human interaction.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Game Rules](#game-rules)
3. [Agents](#agents)
4. [XMPP Message Protocol](#xmpp-message-protocol)
5. [Directory Structure](#directory-structure)
6. [Setup & Deployment](#setup--deployment)
7. [Usage Example](#usage-example)
8. [Human Client Commands](#human-client-commands)
9. [Session Report](#session-report)
10. [Environment Variables](#environment-variables)
11. [Monitoring](#monitoring)

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
│          │         XMPP                                                     │
│          └────────────────────────────┐───────────────────────┐             │
│          │                            │                       │             │
│  ┌───────▼──────────────┐  ┌──────────▼─────────┐  ┌──────────▼───────────┐ │
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
| 2–7   | Two–Seven |
| 10    | Jack      |
| 11    | Knight    |
| 12    | King      |

### Playability Rule
A card is playable if:
- Its **rank** matches the top card's rank, **OR**
- Its **suit** matches the current active suit.

Under a **penalty stack**, only **Two (2)** can be played. This rule applies equally to all cards including Ace and Seven.

### Special Card Effects
| Card          | Effect |
|---------------|--------|
| **Two (2)**   | Next player must draw **+2 cards** (stackable — multiple Twos add up) |
| **Seven (7)** | The player who played it **chooses the active suit** |
| **Ace (1)**   | **Skips** the next player's turn |

### Win / Lose Condition
- First player to **empty their hand** finishes in 1st place (winner).
- The game continues until only **one player remains** — that player is the **loser**.
- Rounds play back-to-back automatically until the human types `stop`.

### Deck Recycling
When the draw deck is exhausted, all played cards **except the top card** are reshuffled into a new deck automatically.

---

## Agents

### Master Agent
- Central coordinator and sole rule enforcer
- Validates every action before applying it
- Manages turn order, penalty stacks, skip logic, and suit choices after Seven
- Broadcasts game state to all players after each action
- Tracks rounds, finish order, and session statistics

### Q-Learning Agent
- Loads a pre-trained Q-table from `models/qtable_improved.npz` (~2s) with `.pkl` fallback
- Q-table is loaded **once before the XMPP connection starts** — no background threads, no race conditions
- **No learning occurs during gameplay** — inference only
- Selects the highest Q-value action among valid moves

### Random Agent
- Picks a random valid card each turn
- Falls back to draw if no valid card exists

### Heuristic Agent
- Selects the card whose rank and suit appear most frequently in hand
- Maximises chaining potential by prioritising cards that match others in hand
- Falls back to draw if no valid card exists

### Human Client (CLI)
- Connects externally via XMPP
- Can **participate** as a player (`start`) or **spectate** agent-only rounds (`watch <n>`)
- In spectator mode: sees all hand sizes but not private hands; rounds remaining displayed
- Displays full game state after every move with Last Move shown first
- Prompts for suit choice when Seven is played
- Displays round results and full session report on `stop`

---

## XMPP Message Protocol

### Performatives

| Performative  | Direction             | Description                                        |
|---------------|-----------------------|----------------------------------------------------|
| `subscribe`   | Agent → Master        | Register and join the session                      |
| `confirm`     | Master → Agent        | Registration acknowledged                          |
| `command`     | Human → Master        | Game control (`start` / `watch` / `stop`)          |
| `inform`      | Master → All          | State broadcast after each action                  |
| `request`     | Master → Current      | Request action from the current player             |
| `action`      | Agent → Master        | Submit chosen action                               |
| `suit_choice` | Human → Master        | Choose active suit after playing Seven             |
| `reject`      | Master → Agent        | Action rejected with error reason; master re-requests |

### Action Payloads

Draw a card:
```json
{"action": "draw"}
```

Play a card:
```json
{"action": "play", "card_index": 2}
```

Watch n rounds (spectator mode):
```json
{"command": "watch", "rounds": 10}
```

Choose suit after Seven:
```json
{"suit": 1}
```

---

## Directory Structure

```
hazz2_multiagent/
├── docker-compose.yml           ← all services definition
├── run.sh                       ← one-command startup (Ubuntu)
├── requirements.txt
├── models/
│   ├── qtable_improved.npz      ← pre-trained Q-table (recommended, ~2s load)
│   └── qtable_improved.pkl      ← original pickle fallback (~70s load)
├── ejabberd/
│   └── ejabberd.yml             ← ejabberd XMPP server config
├── shared/
│   └── game_env.py              ← Card, Suit, Rank, Hazz2Env
├── master_agent/
│   ├── Dockerfile
│   └── master_agent.py
├── qlearning_agent/
│   ├── Dockerfile
│   └── qlearning_agent.py
├── random_agent/
│   ├── Dockerfile
│   └── random_agent.py
├── heuristic_agent/
│   ├── Dockerfile
│   └── heuristic_agent.py
└── human_client/
    ├── Dockerfile
    └── human_client.py
```

---

## Setup & Deployment

### Prerequisites
- Docker and Docker Compose installed
- Pre-trained Q-table: `qtable_improved.npz` (or `qtable_improved.pkl`)

### Step 1 — Start the system

#### Ubuntu — one command
```bash
chmod +x run.sh && ./run.sh
```

#### Windows — run each command manually

```bash
# Start the XMPP broker first
docker compose up -d ejabberd

# Register all player accounts on ejabberd
# (each agent authenticates with its own JID and password)
docker exec ejabberd ejabberdctl register master      ejabberd master_pass
docker exec ejabberd ejabberdctl register qagent      ejabberd qagent_pass
docker exec ejabberd ejabberdctl register randomagent ejabberd random_pass
docker exec ejabberd ejabberdctl register human       ejabberd human_pass
docker exec ejabberd ejabberdctl register heuristic   ejabberd heuristic_pass

# Start the Master Agent first — it must be ready before others connect
docker compose up -d --build master_agent

# Start all other agents — they will auto-register with the Master Agent
docker compose up -d --build qlearning_agent random_agent human_agent heuristic_agent
```

### Step 2 — Connect as the human player

Attach to the human client container:
```bash
docker attach human_agent
```

Detach safely with **Ctrl+P then Ctrl+Q** (not Ctrl+C).

---

## Usage Example

```
$ docker attach human_agent

> help

  Commands:
    start            - Start the game (you participate)
    watch <n>        - Watch agents play n rounds (you spectate)
    stop             - Stop immediately and show session report
    play <index>     - Play a card by its index number
    draw             - Draw a card from the deck
    suit <0-3>       - Choose suit after playing Seven
                       0=Coins  1=Cups  2=Swords  3=Clubs
    help             - Show this help message

  Card rules:
    Two   (2) -> next player draws +2 (stackable)
    Seven (7) -> you choose the active suit
    Ace   (1) -> next player's turn is skipped
    All cards playable only on matching rank or suit

> watch 10
  Spectator mode: watching 10 round(s). Agents will play without you.

  [SPECTATOR] Round 1 started! Turn order: qagent -> heuristic -> randomagent
  Watching 10 round(s). Sit back and observe.

  ============================================================
  [SPECTATOR MODE] Rounds remaining: 9
  Last Move: qagent played Two of Coins [penalty stack: 2]
  ------------------------------------------------------------
  Round:        1  |  Turn: 5
  Current Turn: HEURISTIC
  Active:       qagent, heuristic, randomagent
  Top Card:     Two of Coins
  Active Suit:  Coins
  Deck:         28 cards remaining

  Hand sizes:
    qagent     : 3 cards
    heuristic  : 4 cards  <-- current
    randomagent: 5 cards
  ============================================================

> stop

  ============================================================
    GAME SESSION REPORT
  ============================================================
  ...
```

---

## Human Client Commands

| Command        | Description                                                              |
|----------------|--------------------------------------------------------------------------|
| `start`        | Start the game — you participate as a player                             |
| `watch <n>`    | Watch agents play `n` rounds in spectator mode — you do not participate  |
| `stop`         | Stop immediately and display the full session report                     |
| `play <index>` | Play the card at position `index` from your hand                         |
| `draw`         | Draw a card from the deck                                                |
| `suit <0-3>`   | Choose active suit after playing a Seven (0=Coins 1=Cups 2=Swords 3=Clubs) |
| `help`         | Show all available commands and card rules                               |

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

Order shown is 1st place (winner) → last place (loser). In `watch` mode, human does not appear in round results.

---

## Environment Variables

| Variable             | Default                | Description                       |
|----------------------|------------------------|-----------------------------------|
| `XMPP_SERVER`        | `ejabberd`             | XMPP server hostname              |
| `MASTER_JID`         | `master@ejabberd`      | Master agent JID                  |
| `MASTER_PASSWORD`    | `master_pass`          | Master agent password             |
| `QAGENT_JID`         | `qagent@ejabberd`      | Q-Learning agent JID              |
| `QAGENT_PASSWORD`    | `qagent_pass`          | Q-Learning agent password         |
| `RANDOM_JID`         | `randomagent@ejabberd` | Random agent JID                  |
| `RANDOM_PASSWORD`    | `random_pass`          | Random agent password             |
| `HEURISTIC_JID`      | `heuristic@ejabberd`   | Heuristic agent JID               |
| `HEURISTIC_PASSWORD` | `heuristic_pass`       | Heuristic agent password          |
| `HUMAN_JID`          | `human@ejabberd`       | Human player JID                  |
| `HUMAN_PASSWORD`     | `human_pass`           | Human player password             |
| `MODELS_PATH`        | `models`               | Directory containing Q-table file |

---

## Monitoring

View logs per container:
```bash
docker logs master_agent      # game events, turn actions, round results
docker logs qlearning_agent   # Q-table loading, action selection
docker logs random_agent      # turn actions
docker logs heuristic_agent   # turn actions
```

Stop all containers:
```bash
docker compose down
```