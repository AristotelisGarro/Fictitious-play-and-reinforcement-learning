"""
Stochastic Snakes & Ladders — Multi-Agent RL & FP Simulation
=============================================================
Game:   Two-player competitive Snakes & Ladders on a configurable NxN board.
        Each turn a player rolls a die (stochastic). Landing on a snake head
        sends you back; landing on a ladder base advances you. First to reach
        or exceed cell N² wins.

Stochasticity:
  - Die roll: uniform [1..die_faces]  → environment randomness
  - Optional: "power tiles" let a player choose between two die rolls → decision points
    This is what gives agents something to LEARN (otherwise the game is pure luck).

Agent types:
  - FP  : Fictitious Play — tracks opponent's empirical tile-visit distribution,
          best-responds to the induced state distribution.
  - IQL : Independent Q-Learning (ε-greedy) — treats opponent as environment.
  - MMQ : MiniMax-Q — joint state, adversarial value estimation.

Matchups: all combinations of the three agent types.

Reports:
  - Win rates per agent over episodes
  - Rolling win rate convergence
  - Nash Distance (empirical strategy vs Nash EQ)
  - Strategy heatmap (action choice frequency per board tile)
  - Comparison bar charts
"""

import numpy as np
import random
import argparse
import json
import csv
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════
#  BOARD DEFINITION
# ══════════════════════════════════════════════════════════════════

# Default 10×10 board (cells 1..100)
DEFAULT_SNAKES = {
    97: 78, 95: 56, 88: 24, 62: 18, 48: 26, 36: 6, 32: 10
}
DEFAULT_LADDERS = {
    1: 38, 4: 14, 9: 31, 20: 42, 28: 76, 40: 59, 51: 67, 63: 81, 71: 91
}
# "Power tiles": cells where the player may choose between rolling die once or twice
# This introduces decision-making into what would otherwise be a pure-luck game
DEFAULT_POWER_TILES = {15, 30, 45, 60, 75, 90}

# Actions: 0 = "roll once" (standard), 1 = "roll twice, keep better"
ACTIONS = [0, 1]
ACTION_NAMES = {0: "Single Roll", 1: "Best-of-Two"}


@dataclass
class BoardConfig:
    size: int = 100                        # total cells (goal = size)
    die_faces: int = 6
    snakes: Dict[int, int] = field(default_factory=lambda: dict(DEFAULT_SNAKES))
    ladders: Dict[int, int] = field(default_factory=lambda: dict(DEFAULT_LADDERS))
    power_tiles: set = field(default_factory=lambda: set(DEFAULT_POWER_TILES))

    def apply_tile(self, cell: int) -> int:
        """Apply snake/ladder effect."""
        if cell in self.snakes:
            return self.snakes[cell]
        if cell in self.ladders:
            return self.ladders[cell]
        return cell

    def roll(self) -> int:
        return random.randint(1, self.die_faces)

    def move(self, pos: int, action: int) -> Tuple[int, int]:
        """
        Execute action from pos.
        action=0: single roll
        action=1: best of two rolls (only meaningful on power tiles)
        Returns (new_pos_after_tile, raw_roll_used)
        """
        if action == 1:
            r1, r2 = self.roll(), self.roll()
            roll = max(r1, r2)
        else:
            roll = self.roll()
        new_pos = min(pos + roll, self.size)
        new_pos = self.apply_tile(new_pos)
        return new_pos, roll


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    n_episodes: int = 2000
    # RL
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    # FP
    fp_smoothing: float = 1.0
    # MiniMax-Q
    minimax_lr: float = 0.1
    minimax_gamma: float = 0.95
    # Reporting
    report_interval: int = 200
    plot_window: int = 100    

    def to_dict(self):
        return self.__dict__.copy()


# ══════════════════════════════════════════════════════════════════
#  NASH DISTANCE
# ══════════════════════════════════════════════════════════════════
# Stage-game Nash EQ of the power-tile sub-game:
# Both players should choose action=1 (best-of-two) on every power tile
# because it strictly dominates — always gives a higher or equal roll.
# Nash EQ: p_action1 = 1.0 for both agents.
NASH_P_AGGRESSIVE = 1.0   # prob of choosing action=1 (dominant strategy)

def compute_nash_distance(p0_agg: float, p1_agg: float) -> float:
    return float(np.sqrt((p0_agg - NASH_P_AGGRESSIVE)**2 +
                         (p1_agg - NASH_P_AGGRESSIVE)**2))


# ══════════════════════════════════════════════════════════════════
#  BASE AGENT
# ══════════════════════════════════════════════════════════════════

class Agent:
    def __init__(self, name: str, agent_id: int, board: BoardConfig):
        self.name = name
        self.agent_id = agent_id
        self.board = board
        self.pos = 0
        self.wins = 0
        self.episode_rewards: List[float] = []   # +1 win / -1 loss per episode
        self.power_action_log: List[int] = []    # actions taken on power tiles
        self.tile_visit_log: List[int] = []      # tiles visited

    def select_action(self, my_pos: int, opp_pos: int) -> int:
        """Return action (0 or 1). Only matters on power tiles."""
        raise NotImplementedError

    def update(self, my_pos: int, opp_pos: int, action: int,
               new_pos: int, reward: float, done: bool):
        pass

    def reset_episode(self):
        self.pos = 0

    def reset_all(self):
        self.pos = 0
        self.wins = 0
        self.episode_rewards = []
        self.power_action_log = []
        self.tile_visit_log = []

    def win_rate(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return sum(1 for r in self.episode_rewards if r > 0) / len(self.episode_rewards)

    def aggression_rate(self) -> float:
        """Fraction of power-tile turns where agent chose action=1."""
        if not self.power_action_log:
            return 0.0
        return sum(self.power_action_log) / len(self.power_action_log)

    def __repr__(self):
        return f"{self.name}(id={self.agent_id})"


# ══════════════════════════════════════════════════════════════════
#  FICTITIOUS PLAY AGENT
# ══════════════════════════════════════════════════════════════════

class FictitiousPlayAgent(Agent):
    """
    FP tracks the empirical distribution of opponent positions and
    chooses the action that maximises expected progress given that
    the opponent is at their average position.
    On a power tile: estimate expected advance for action 0 vs 1
    given the current board state, and pick the better one.
    """
    def __init__(self, agent_id: int, board: BoardConfig, config: Config):
        super().__init__("FP", agent_id, board)
        self.config = config
        # Track empirical opponent position distribution
        self.opp_pos_counts: Dict[int, float] = defaultdict(lambda: config.fp_smoothing)
        self.opp_total = config.fp_smoothing * (board.size + 1)

    def _expected_advance(self, pos: int, action: int, n_samples: int = 200) -> float:
        """Monte-Carlo estimate of expected new position for (pos, action)."""
        total = 0.0
        for _ in range(n_samples):
            new_pos, _ = self.board.move(pos, action)
            total += new_pos
        return total / n_samples

    def select_action(self, my_pos: int, opp_pos: int) -> int:
        if my_pos not in self.board.power_tiles:
            return 0   # no choice on regular tiles
        ev0 = self._expected_advance(my_pos, 0, n_samples=50)
        ev1 = self._expected_advance(my_pos, 1, n_samples=50)
        return 1 if ev1 >= ev0 else 0

    def update(self, my_pos: int, opp_pos: int, action: int,
               new_pos: int, reward: float, done: bool):
        self.opp_pos_counts[opp_pos] += 1
        self.opp_total += 1

    def reset_all(self):
        super().reset_all()
        self.opp_pos_counts = defaultdict(lambda: self.config.fp_smoothing)
        self.opp_total = self.config.fp_smoothing * (self.board.size + 1)


# ══════════════════════════════════════════════════════════════════
#  INDEPENDENT Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════════

class IQLAgent(Agent):
    """
    State = (my_pos, opp_pos) — treats opponent as part of environment.
    Q[state][action] updated with standard Q-learning.
    On non-power tiles, always returns action=0.
    ε-greedy exploration.
    """
    def __init__(self, agent_id: int, board: BoardConfig, config: Config):
        super().__init__("IQL", agent_id, board)
        self.config = config
        self.epsilon = config.epsilon_start
        self.Q: Dict = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self._last_state = None
        self._last_action = None

    def _state(self, my_pos: int, opp_pos: int):
        # Bucket positions to reduce state space (groups of 5)
        return (my_pos // 5, opp_pos // 5)

    def select_action(self, my_pos: int, opp_pos: int) -> int:
        if my_pos not in self.board.power_tiles:
            return 0
        s = self._state(my_pos, opp_pos)
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            q = self.Q[s]
            mx = max(q.values())
            action = random.choice([a for a, v in q.items() if v == mx])
        self._last_state = s
        self._last_action = action
        return action

    def update(self, my_pos: int, opp_pos: int, action: int,
               new_pos: int, reward: float, done: bool):
        if self._last_state is None or my_pos not in self.board.power_tiles:
            return
        ns = self._state(new_pos, opp_pos)
        max_next = max(self.Q[ns].values()) if not done else 0.0
        td = reward + self.config.gamma * max_next - self.Q[self._last_state][action]
        self.Q[self._last_state][action] += self.config.alpha * td
        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)
        self._last_state = None

    def reset_all(self):
        super().reset_all()
        self.epsilon = self.config.epsilon_start
        self.Q = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self._last_state = None


# ══════════════════════════════════════════════════════════════════
#  MINIMAX-Q AGENT
# ══════════════════════════════════════════════════════════════════

class MinimaxQAgent(Agent):
    """
    Joint state (my_pos_bucket, opp_pos_bucket).
    Q[s][a_self][a_opp] — adversarial.
    Solves 2×2 maximin analytically.
    """
    def __init__(self, agent_id: int, board: BoardConfig, config: Config):
        super().__init__("MMQ", agent_id, board)
        self.config = config
        self.epsilon = config.epsilon_start
        self.Q: Dict = defaultdict(
            lambda: {a: {b: 0.0 for b in ACTIONS} for a in ACTIONS}
        )
        self.V: Dict = defaultdict(float)
        self.pi: Dict = defaultdict(lambda: {0: 0.5, 1: 0.5})
        self._last_state = None
        self._last_action = None

    def _state(self, my_pos: int, opp_pos: int):
        return (my_pos // 5, opp_pos // 5)

    def _solve_maximin(self, s) -> Dict[int, float]:
        Q = self.Q[s]
        q00, q01 = Q[0][0], Q[0][1]
        q10, q11 = Q[1][0], Q[1][1]
        denom = q00 - q10 - q01 + q11
        if abs(denom) < 1e-10:
            ev0 = 0.5*q00 + 0.5*q01
            ev1 = 0.5*q10 + 0.5*q11
            if ev0 > ev1:
                return {0: 1.0, 1: 0.0}
            elif ev1 > ev0:
                return {0: 0.0, 1: 1.0}
            return {0: 0.5, 1: 0.5}
        p = max(0.0, min(1.0, (q11 - q10) / denom))
        return {0: p, 1: 1.0 - p}

    def select_action(self, my_pos: int, opp_pos: int) -> int:
        if my_pos not in self.board.power_tiles:
            return 0
        s = self._state(my_pos, opp_pos)
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            pi = self.pi[s]
            action = 0 if random.random() < pi[0] else 1
        self._last_state = s
        self._last_action = action
        return action

    def update(self, my_pos: int, opp_pos: int, action: int,
               new_pos: int, reward: float, done: bool):
        if self._last_state is None or my_pos not in self.board.power_tiles:
            return
        # Assume opponent plays 0 (conservative assumption)
        opp_action = 0
        ns = self._state(new_pos, opp_pos)
        td = (reward + self.config.minimax_gamma * self.V[ns]
              - self.Q[self._last_state][action][opp_action])
        self.Q[self._last_state][action][opp_action] += self.config.minimax_lr * td
        self.pi[self._last_state] = self._solve_maximin(self._last_state)
        pi = self.pi[self._last_state]
        self.V[self._last_state] = min(
            sum(pi[a] * self.Q[self._last_state][a][b] for a in ACTIONS)
            for b in ACTIONS
        )
        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)
        self._last_state = None

    def reset_all(self):
        super().reset_all()
        self.epsilon = self.config.epsilon_start
        self.Q = defaultdict(lambda: {a: {b: 0.0 for b in ACTIONS} for a in ACTIONS})
        self.V = defaultdict(float)
        self.pi = defaultdict(lambda: {0: 0.5, 1: 0.5})
        self._last_state = None


# ══════════════════════════════════════════════════════════════════
#  GAME ENGINE
# ══════════════════════════════════════════════════════════════════

class SnakesLaddersGame:
    MAX_TURNS = 500   # safety limit per episode

    def __init__(self, agent0: Agent, agent1: Agent,
                 board: BoardConfig, config: Config):
        self.agent0 = agent0
        self.agent1 = agent1
        self.board = board
        self.config = config
        self.episode_log: List[Dict] = []   # one entry per episode

    def run_episode(self) -> Dict:
        self.agent0.reset_episode()
        self.agent1.reset_episode()
        p0, p1 = 0, 0
        turns = 0
        winner = None

        while turns < self.MAX_TURNS:
            # ── Agent 0 turn ─────────────────────────
            a0 = self.agent0.select_action(p0, p1)
            np0, roll0 = self.board.move(p0, a0)
            if p0 in self.board.power_tiles:
                self.agent0.power_action_log.append(a0)
            self.agent0.tile_visit_log.append(np0)

            if np0 >= self.board.size:
                winner = 0
                self.agent0.update(p0, p1, a0, np0, +1.0, True)
                self.agent1.update(p1, p0, 0, p1, -1.0, True)
                break

            self.agent0.update(p0, p1, a0, np0, 0.0, False)
            p0 = np0

            # ── Agent 1 turn ─────────────────────────
            a1 = self.agent1.select_action(p1, p0)
            np1, roll1 = self.board.move(p1, a1)
            if p1 in self.board.power_tiles:
                self.agent1.power_action_log.append(a1)
            self.agent1.tile_visit_log.append(np1)

            if np1 >= self.board.size:
                winner = 1
                self.agent1.update(p1, p0, a1, np1, +1.0, True)
                self.agent0.update(p0, p1, a0, p0, -1.0, True)
                break

            self.agent1.update(p1, p0, a1, np1, 0.0, False)
            p1 = np1
            turns += 1

        if winner is None:   # draw on timeout
            winner = 0 if p0 >= p1 else 1

        self.agent0.episode_rewards.append(1.0 if winner == 0 else -1.0)
        self.agent1.episode_rewards.append(1.0 if winner == 1 else -1.0)
        if winner == 0:
            self.agent0.wins += 1
        else:
            self.agent1.wins += 1

        return {"winner": winner, "turns": turns, "p0_final": p0, "p1_final": p1}

    def run(self) -> Dict:
        self.agent0.reset_all()
        self.agent1.reset_all()
        self.episode_log = []

        for ep in range(self.config.n_episodes):
            result = self.run_episode()
            result["episode"] = ep + 1
            self.episode_log.append(result)

        return self._summary()

    def _summary(self) -> Dict:
        n = self.config.n_episodes
        agg0 = self.agent0.aggression_rate()
        agg1 = self.agent1.aggression_rate()
        nd = compute_nash_distance(agg0, agg1)

        # Tail (last 20%)
        tail_n = max(1, int(n * 0.2))
        tail_log = self.episode_log[-tail_n:]
        tail_w0 = sum(1 for e in tail_log if e["winner"] == 0) / tail_n
        tail_w1 = 1 - tail_w0

        r0 = self.agent0.episode_rewards
        r1 = self.agent1.episode_rewards
        tail_agg0 = np.mean([a for a, e in
                             zip(self.agent0.power_action_log,
                                 range(len(self.agent0.power_action_log)))
                             ]) if self.agent0.power_action_log else 0.0
        # Simpler tail aggression from full log
        half = len(self.agent0.power_action_log) // 2
        tail_agg0 = np.mean(self.agent0.power_action_log[half:]) if self.agent0.power_action_log else 0.0
        tail_agg1 = np.mean(self.agent1.power_action_log[half:]) if self.agent1.power_action_log else 0.0
        nd_tail = compute_nash_distance(tail_agg0, tail_agg1)

        return {
            "agent0": str(self.agent0),
            "agent1": str(self.agent1),
            "n_episodes": n,
            "wins_0": self.agent0.wins,
            "wins_1": self.agent1.wins,
            "win_rate_0": self.agent0.wins / n,
            "win_rate_1": self.agent1.wins / n,
            "win_rate_0_tail": tail_w0,
            "win_rate_1_tail": tail_w1,
            "aggression_rate_0": agg0,
            "aggression_rate_1": agg1,
            "nash_distance_overall": nd,
            "nash_distance_tail": nd_tail,
            "avg_turns": np.mean([e["turns"] for e in self.episode_log]),
        }


# ══════════════════════════════════════════════════════════════════
#  FACTORIES
# ══════════════════════════════════════════════════════════════════

def make_agent(t: str, agent_id: int, board: BoardConfig, config: Config) -> Agent:
    t = t.lower()
    if t == "fp":
        return FictitiousPlayAgent(agent_id, board, config)
    elif t == "iql":
        return IQLAgent(agent_id, board, config)
    elif t in ("mmq", "minimax"):
        return MinimaxQAgent(agent_id, board, config)
    raise ValueError(f"Unknown agent type: {t}")


# ══════════════════════════════════════════════════════════════════
#  ROLLING HELPERS
# ══════════════════════════════════════════════════════════════════

def rolling_win_rate(rewards: List[float], window: int) -> List[float]:
    out = []
    for i in range(len(rewards)):
        s = max(0, i - window + 1)
        chunk = rewards[s:i+1]
        out.append(sum(1 for r in chunk if r > 0) / len(chunk))
    return out

def rolling_aggression(action_log: List[int], window: int) -> List[float]:
    out = []
    for i in range(len(action_log)):
        s = max(0, i - window + 1)
        chunk = action_log[s:i+1]
        out.append(sum(chunk) / len(chunk))
    return out


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════

MATCHUP_COLORS = {
    "FP vs FP":   ("#2196F3", "#64B5F6"),
    "IQL vs IQL": ("#E53935", "#EF9A9A"),
    "MMQ vs MMQ": ("#43A047", "#A5D6A7"),
    "IQL vs FP":  ("#FB8C00", "#FFCC80"),
    "MMQ vs FP":  ("#8E24AA", "#CE93D8"),
    "IQL vs MMQ": ("#00897B", "#80CBC4"),
}

SNAKE_COLOR  = "#C62828"
LADDER_COLOR = "#2E7D32"
POWER_COLOR  = "#F9A825"


def _board_coords(cell: int, size: int = 10) -> Tuple[float, float]:
    """Convert 1-indexed cell to (col, row) for display (snake-pattern)."""
    idx = cell - 1
    row = idx // size
    col = idx % size
    if row % 2 == 1:
        col = size - 1 - col
    return col + 0.5, row + 0.5


def plot_board(board: BoardConfig, visit_counts: Optional[Dict[int, int]],
               title: str, save_path: str):
    """
    Draw the Snakes & Ladders board with:
    - Heatmap overlay of tile visit frequency
    - Snake and ladder markers
    - Power tile markers
    """
    N = 10
    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    # Draw grid
    for r in range(N):
        for c in range(N):
            cell = r * N + c + 1
            ax.add_patch(plt.Rectangle((c, r), 1, 1, linewidth=0.5,
                                       edgecolor="#BDBDBD", facecolor="none", zorder=1))

    # Visit heatmap
    if visit_counts:
        max_v = max(visit_counts.values()) if visit_counts else 1
        cmap = plt.cm.YlOrRd
        for cell, cnt in visit_counts.items():
            cx, cy = _board_coords(cell, N)
            r, c = int(cy - 0.5), int(cx - 0.5)
            alpha = 0.15 + 0.7 * (cnt / max_v)
            color = cmap(cnt / max_v)
            ax.add_patch(plt.Rectangle((c, r), 1, 1, linewidth=0,
                                       facecolor=color, alpha=alpha, zorder=2))

    # Power tiles
    for pt in board.power_tiles:
        cx, cy = _board_coords(pt, N)
        ax.add_patch(plt.Rectangle((int(cx-0.5), int(cy-0.5)), 1, 1,
                                   linewidth=2, edgecolor=POWER_COLOR,
                                   facecolor=POWER_COLOR, alpha=0.25, zorder=3))

    # Snakes (arrows head→tail)
    for head, tail in board.snakes.items():
        hx, hy = _board_coords(head, N)
        tx, ty = _board_coords(tail, N)
        ax.annotate("", xy=(tx, ty), xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="-|>", color=SNAKE_COLOR,
                                   lw=2.5, mutation_scale=18), zorder=5)
        ax.plot(hx, hy, "o", color=SNAKE_COLOR, ms=8, zorder=6)

    # Ladders (arrows bottom→top)
    for base, top in board.ladders.items():
        bx, by = _board_coords(base, N)
        tx, ty = _board_coords(top, N)
        ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="-|>", color=LADDER_COLOR,
                                   lw=2.5, mutation_scale=18), zorder=5)
        ax.plot(bx, by, "s", color=LADDER_COLOR, ms=8, zorder=6)

    # Cell numbers
    for r in range(N):
        for c in range(N):
            cell = r * N + c + 1
            cx, cy = _board_coords(cell, N)
            ax.text(cx, cy + 0.32, str(cell), ha="center", va="center",
                    fontsize=6.5, color="#424242", fontweight="bold", zorder=7)

    # Labels
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold", color="#212121", pad=10)

    legend_elements = [
        mpatches.Patch(facecolor=SNAKE_COLOR, alpha=0.7, label="Snake (↓)"),
        mpatches.Patch(facecolor=LADDER_COLOR, alpha=0.7, label="Ladder (↑)"),
        mpatches.Patch(facecolor=POWER_COLOR, alpha=0.5, label="Power Tile"),
    ]
    if visit_counts:
        legend_elements.append(
            mpatches.Patch(facecolor=plt.cm.YlOrRd(0.7), alpha=0.7, label="Tile visit heatmap")
        )
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8, framealpha=0.85, bbox_to_anchor=(0, -0.06))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matchup(result: Dict, agent0: Agent, agent1: Agent,
                 config: Config, board: BoardConfig, save_path: str):
    """
    6-panel figure per matchup:
      [0] Rolling win rate over episodes
      [1] Cumulative wins
      [2] Rolling aggression rate (action=1 on power tiles)
      [3] Nash distance over power-tile decisions
      [4] Board heatmap (agent0 visits)
      [5] Board heatmap (agent1 visits)
    """
    matchup = result["matchup"]
    colors = MATCHUP_COLORS.get(matchup, ("#1565C0", "#C62828"))
    c0, c1 = colors
    n_ep = result["n_episodes"]
    window = config.plot_window

    r0 = agent0.episode_rewards
    r1 = agent1.episode_rewards
    roll_wr0 = rolling_win_rate(r0, window)
    roll_wr1 = rolling_win_rate(r1, window)
    episodes = list(range(1, n_ep + 1))

    cum_w0 = list(np.cumsum([1 if r > 0 else 0 for r in r0]))
    cum_w1 = list(np.cumsum([1 if r > 0 else 0 for r in r1]))

    # Aggression series (only power-tile turns)
    agg0 = rolling_aggression(agent0.power_action_log, min(window, max(len(agent0.power_action_log), 1)))
    agg1 = rolling_aggression(agent1.power_action_log, min(window, max(len(agent1.power_action_log), 1)))
    agg_x0 = list(range(1, len(agg0) + 1))
    agg_x1 = list(range(1, len(agg1) + 1))

    # Nash distance series (per power-tile decision pair)
    min_agg_len = min(len(agg0), len(agg1))
    nd_series = [compute_nash_distance(agg0[i], agg1[i]) for i in range(min_agg_len)]
    nd_x = list(range(1, min_agg_len + 1))

    # Visit counts
    visits0: Dict[int, int] = defaultdict(int)
    visits1: Dict[int, int] = defaultdict(int)
    for cell in agent0.tile_visit_log:
        visits0[cell] += 1
    for cell in agent1.tile_visit_log:
        visits1[cell] += 1

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    title_kw = dict(fontsize=10, fontweight="bold", color="#212121", pad=8)
    label_kw = dict(fontsize=8.5, color="#424242")

    # ── [0] Rolling Win Rate ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor("#F5F5F5")
    ax.plot(episodes, roll_wr0, color=c0, lw=1.6, label=f"Agent0 ({result['agent0']})")
    ax.plot(episodes, roll_wr1, color=c1, lw=1.6, ls="--", label=f"Agent1 ({result['agent1']})")
    ax.axhline(0.5, color="#9E9E9E", lw=0.8, ls=":", alpha=0.7, label="50% (equal)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Rolling Win Rate (window={window})", **title_kw)
    ax.set_xlabel("Episode", **label_kw)
    ax.set_ylabel("Win Rate", **label_kw)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3, ls=":")
    # Final annotation
    ax.annotate(f"{roll_wr0[-1]:.2%}", xy=(n_ep, roll_wr0[-1]),
                xytext=(-45, 6), textcoords="offset points",
                fontsize=8, color=c0)
    ax.annotate(f"{roll_wr1[-1]:.2%}", xy=(n_ep, roll_wr1[-1]),
                xytext=(-45, -14), textcoords="offset points",
                fontsize=8, color=c1)

    # ── [1] Cumulative Wins ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2:])
    ax.set_facecolor("#F5F5F5")
    ax.plot(episodes, cum_w0, color=c0, lw=1.6, label=f"Agent0 ({result['agent0']})")
    ax.plot(episodes, cum_w1, color=c1, lw=1.6, ls="--", label=f"Agent1 ({result['agent1']})")
    ax.set_title("Cumulative Wins", **title_kw)
    ax.set_xlabel("Episode", **label_kw)
    ax.set_ylabel("Total Wins", **label_kw)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3, ls=":")
    ax.annotate(f"{cum_w0[-1]}", xy=(n_ep, cum_w0[-1]),
                xytext=(-35, 5), textcoords="offset points", fontsize=8, color=c0)
    ax.annotate(f"{cum_w1[-1]}", xy=(n_ep, cum_w1[-1]),
                xytext=(-35, -14), textcoords="offset points", fontsize=8, color=c1)

    # ── [2] Rolling Aggression Rate ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.set_facecolor("#F5F5F5")
    if agg0:
        ax.plot(agg_x0, agg0, color=c0, lw=1.4, label=f"Agent0 ({result['agent0']})")
    if agg1:
        ax.plot(agg_x1, agg1, color=c1, lw=1.4, ls="--", label=f"Agent1 ({result['agent1']})")
    ax.axhline(NASH_P_AGGRESSIVE, color="#B71C1C", lw=1.0, ls="--",
               alpha=0.7, label=f"Nash (p={NASH_P_AGGRESSIVE:.0f})")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("Rolling Aggression Rate on Power Tiles\n(P(choose Best-of-Two roll))", **title_kw)
    ax.set_xlabel("Power-Tile Decision #", **label_kw)
    ax.set_ylabel("P(Action = Best-of-Two)", **label_kw)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3, ls=":")

    # ── [3] Nash Distance ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    ax.set_facecolor("#F5F5F5")
    nd_color = "#6A1B9A"
    if nd_series:
        ax.plot(nd_x, nd_series, color=nd_color, lw=1.4, alpha=0.85)
        ax.fill_between(nd_x, nd_series, alpha=0.15, color=nd_color)
        final_nd = nd_series[-1]
        ax.annotate(f"Final: {final_nd:.3f}", xy=(nd_x[-1], final_nd),
                    xytext=(-70, 8), textcoords="offset points",
                    fontsize=8, color=nd_color,
                    arrowprops=dict(arrowstyle="->", color=nd_color, lw=0.8))
    ax.axhline(0.0, color="#B71C1C", lw=1.0, ls="--", alpha=0.8, label="Nash EQ (dist=0)")
    ax.axhline(np.sqrt(2), color="#1B5E20", lw=0.8, ls=":", alpha=0.6, label="Max dist (√2)")
    ax.set_ylim(-0.05, np.sqrt(2) + 0.15)
    ax.set_title("Nash Distance\n(deviation from dominant strategy: always Best-of-Two)", **title_kw)
    ax.set_xlabel("Power-Tile Decision #", **label_kw)
    ax.set_ylabel("Euclidean Dist to Nash EQ", **label_kw)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3, ls=":")

    # ── [4,5] Board Heatmaps ──────────────────────────────────────────────
    N = 10
    for ax_idx, (agent_label, visits, color) in enumerate([
        (f"Agent0 ({result['agent0']})", visits0, c0),
        (f"Agent1 ({result['agent1']})", visits1, c1),
    ]):
        ax = fig.add_subplot(gs[2, ax_idx*2: ax_idx*2+2])
        ax.set_facecolor("#EEEEEE")

        # Draw grid + heatmap
        max_v = max(visits.values()) if visits else 1
        cmap = plt.cm.Blues if ax_idx == 0 else plt.cm.Reds
        for r in range(N):
            for c in range(N):
                cell = r * N + c + 1
                cx_b, cy_b = _board_coords(cell, N)
                rc, cc = int(cy_b - 0.5), int(cx_b - 0.5)
                cnt = visits.get(cell, 0)
                facecolor = cmap(0.1 + 0.85 * cnt / max_v) if cnt > 0 else "#F5F5F5"
                ax.add_patch(plt.Rectangle((cc, rc), 1, 1, lw=0.4,
                                           edgecolor="#BDBDBD", facecolor=facecolor, zorder=1))
                ax.text(cx_b, cy_b + 0.28, str(cell),
                        ha="center", va="center", fontsize=5.5,
                        color="white" if cnt / max_v > 0.5 else "#424242",
                        fontweight="bold", zorder=4)

        # Power tiles
        for pt in board.power_tiles:
            cx_b, cy_b = _board_coords(pt, N)
            ax.add_patch(plt.Rectangle((int(cx_b-0.5), int(cy_b-0.5)), 1, 1,
                                       lw=2, edgecolor=POWER_COLOR,
                                       facecolor="none", zorder=3))

        # Snakes
        for head, tail in board.snakes.items():
            hx, hy = _board_coords(head, N)
            tx, ty = _board_coords(tail, N)
            ax.annotate("", xy=(tx, ty), xytext=(hx, hy),
                        arrowprops=dict(arrowstyle="-|>", color=SNAKE_COLOR,
                                       lw=1.8, mutation_scale=12), zorder=5)

        # Ladders
        for base, top in board.ladders.items():
            bx, by = _board_coords(base, N)
            tx, ty = _board_coords(top, N)
            ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="-|>", color=LADDER_COLOR,
                                       lw=1.8, mutation_scale=12), zorder=5)

        ax.set_xlim(0, N); ax.set_ylim(0, N)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Tile Visit Heatmap — {agent_label}", **title_kw)

    # Super title
    nd_o = result["nash_distance_overall"]
    nd_t = result["nash_distance_tail"]
    fig.suptitle(
        f"Snakes & Ladders — Matchup: {matchup}  |  Episodes: {n_ep}  |  "
        f"Win Rate A0={result['win_rate_0']:.2%} / A1={result['win_rate_1']:.2%}  |  "
        f"Nash Dist (overall/tail): {nd_o:.3f} / {nd_t:.3f}",
        fontsize=12, fontweight="bold", color="#212121", y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Chart saved: {save_path}")


def plot_comparison(all_results: List[Dict], config: Config, save_path: str):
    matchups = [r["matchup"] for r in all_results]
    n = len(matchups)
    x = np.arange(n)
    bar_w = 0.32
    short = [m.replace(" vs ", "\nvs\n") for m in matchups]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor("#FAFAFA")
    title_kw = dict(fontsize=10, fontweight="bold", color="#212121", pad=8)

    pal0 = [MATCHUP_COLORS.get(m, ("#1565C0", "#C62828"))[0] for m in matchups]
    pal1 = [MATCHUP_COLORS.get(m, ("#1565C0", "#C62828"))[1] for m in matchups]

    def bar_labels(ax, bars):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # ── Win Rate ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x - bar_w/2, [r["win_rate_0"] for r in all_results], bar_w,
                label="Agent0", color=pal0, alpha=0.85, edgecolor="white")
    b1 = ax.bar(x + bar_w/2, [r["win_rate_1"] for r in all_results], bar_w,
                label="Agent1", color=pal1, alpha=0.85, edgecolor="white")
    bar_labels(ax, b0); bar_labels(ax, b1)
    ax.axhline(0.5, color="#B71C1C", lw=1, ls="--", alpha=0.6, label="50%")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Overall Win Rate", **title_kw)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── Win Rate (Tail) ─────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x - bar_w/2, [r["win_rate_0_tail"] for r in all_results], bar_w,
                label="Agent0 (tail)", color=pal0, alpha=0.85, edgecolor="white")
    b1 = ax.bar(x + bar_w/2, [r["win_rate_1_tail"] for r in all_results], bar_w,
                label="Agent1 (tail)", color=pal1, alpha=0.85, edgecolor="white")
    bar_labels(ax, b0); bar_labels(ax, b1)
    ax.axhline(0.5, color="#B71C1C", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Win Rate (Last 20% Episodes)", **title_kw)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── Aggression Rate ─────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x - bar_w/2, [r["aggression_rate_0"] for r in all_results], bar_w,
                label="Agent0", color=pal0, alpha=0.85, edgecolor="white")
    b1 = ax.bar(x + bar_w/2, [r["aggression_rate_1"] for r in all_results], bar_w,
                label="Agent1", color=pal1, alpha=0.85, edgecolor="white")
    bar_labels(ax, b0); bar_labels(ax, b1)
    ax.axhline(1.0, color="#1B5E20", lw=1, ls="--", alpha=0.6, label="Nash (p=1.0)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Aggression Rate on Power Tiles\n(P(Best-of-Two))", **title_kw)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── Nash Distance ────────────────────────────────────────────
    ax = axes[3]
    ax.set_facecolor("#F5F5F5")
    w = 0.3
    bo = ax.bar(x - w/2, [r["nash_distance_overall"] for r in all_results], w,
                label="Overall", color="#7B1FA2", alpha=0.85, edgecolor="white")
    bt = ax.bar(x + w/2, [r["nash_distance_tail"] for r in all_results], w,
                label="Tail 20%", color="#AB47BC", alpha=0.85, edgecolor="white")
    bar_labels(ax, bo); bar_labels(ax, bt)
    ax.axhline(0.0, color="#B71C1C", lw=1.2, ls="--", alpha=0.8, label="Nash EQ (dist=0)")
    ax.axhline(np.sqrt(2), color="#1B5E20", lw=0.8, ls=":", alpha=0.6, label="Max (√2)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Nash Equilibrium Distance\n(dominant: always Best-of-Two)", **title_kw)
    ax.set_ylim(0, np.sqrt(2) + 0.2)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    fig.suptitle(f"Snakes & Ladders — All Matchups Comparison  (episodes={config.n_episodes})",
                 fontsize=13, fontweight="bold", color="#212121", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Comparison chart saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
#  REPORTER
# ══════════════════════════════════════════════════════════════════

class Reporter:
    def __init__(self, config: Config, board: BoardConfig):
        self.config = config
        self.board = board
        self.results: List[Dict] = []
        self.agents: List[Tuple[Agent, Agent]] = []

    def add(self, matchup: str, result: Dict, a0: Agent, a1: Agent):
        result["matchup"] = matchup
        self.results.append(result)
        self.agents.append((a0, a1))

    def _interpret_nash(self, nd: float) -> str:
        if nd < 0.05:  return "Converged to Nash EQ (dominant strategy)"
        if nd < 0.30:  return "Near Nash EQ"
        if nd < 0.80:  return "Partial adoption of dominant strategy"
        return              "Far from Nash — suboptimal on power tiles"

    def print_summary(self):
        sep = "=" * 78
        print(f"\n{sep}")
        print("  SNAKES & LADDERS — SIMULATION REPORT")
        print(sep)
        print(f"  Episodes : {self.config.n_episodes}")
        print(f"  Board    : {self.board.size} cells, "
              f"{len(self.board.snakes)} snakes, {len(self.board.ladders)} ladders, "
              f"{len(self.board.power_tiles)} power tiles")
        print(f"  Nash EQ  : Always choose Best-of-Two on power tiles (dominant strategy)")
        print(sep)
        for r in self.results:
            nd_o = r["nash_distance_overall"]
            nd_t = r["nash_distance_tail"]
            print(f"\n  ┌─ {r['matchup']}")
            print(f"  │  Agent0 ({r['agent0']:>5})  Wins={r['wins_0']:>5}  "
                  f"WinRate={r['win_rate_0']:.2%}  (tail={r['win_rate_0_tail']:.2%})  "
                  f"Aggr={r['aggression_rate_0']:.2%}")
            print(f"  │  Agent1 ({r['agent1']:>5})  Wins={r['wins_1']:>5}  "
                  f"WinRate={r['win_rate_1']:.2%}  (tail={r['win_rate_1_tail']:.2%})  "
                  f"Aggr={r['aggression_rate_1']:.2%}")
            print(f"  │  Avg turns/episode : {r['avg_turns']:.1f}")
            print(f"  │  Nash Dist (overall): {nd_o:.4f}")
            print(f"  └  Nash Dist (tail)  : {nd_t:.4f}  → {self._interpret_nash(nd_t)}")
        print(f"\n{sep}\n")

    def print_interval(self, matchup: str, ep: int, wins0: int, wins1: int, total: int):
        wr0 = wins0 / total; wr1 = wins1 / total
        print(f"  [{matchup}] Ep {ep:>5} | WR0={wr0:.2%} WR1={wr1:.2%}")

    def generate_plots(self, plots_dir: str):
        os.makedirs(plots_dir, exist_ok=True)
        print(f"\n  Generating charts → {plots_dir}/")

        # Board overview (no visits)
        board_path = os.path.join(plots_dir, "board_layout.png")
        plot_board(self.board, None, "Snakes & Ladders — Board Layout", board_path)
        print(f"  → Board layout saved: {board_path}")

        for result, (a0, a1) in zip(self.results, self.agents):
            safe = result["matchup"].replace(" vs ", "_vs_").lower()
            path = os.path.join(plots_dir, f"{safe}.png")
            plot_matchup(result, a0, a1, self.config, self.board, path)

        if len(self.results) > 1:
            path = os.path.join(plots_dir, "comparison.png")
            plot_comparison(self.results, self.config, path)


# ══════════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════

def run_matchup(t0: str, t1: str, board: BoardConfig,
                config: Config, reporter: Reporter) -> str:
    name_map = {"fp": "FP", "iql": "IQL", "mmq": "MMQ", "minimax": "MMQ"}
    n0 = name_map.get(t0.lower(), t0.upper())
    n1 = name_map.get(t1.lower(), t1.upper())
    matchup = f"{n0} vs {n1}"
    print(f"\n  Running: {matchup}")

    a0 = make_agent(t0, 0, board, config)
    a1 = make_agent(t1, 1, board, config)
    game = SnakesLaddersGame(a0, a1, board, config)

    a0.reset_all(); a1.reset_all()
    game.episode_log = []

    for ep in range(1, config.n_episodes + 1):
        res = game.run_episode()
        res["episode"] = ep
        game.episode_log.append(res)
        if ep % config.report_interval == 0 or ep == config.n_episodes:
            reporter.print_interval(matchup, ep, a0.wins, a1.wins, ep)

    result = game._summary()
    reporter.add(matchup, result, a0, a1)
    return matchup


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stochastic Snakes & Ladders — FP / IQL / MiniMax-Q",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--matchups", nargs="+",
                        choices=["fp_vs_fp", "iql_vs_iql", "mmq_vs_mmq",
                                 "iql_vs_fp", "mmq_vs_fp", "iql_vs_mmq", "all"],
                        default=["all"])
    # RL
    parser.add_argument("--alpha",         type=float, default=0.1)
    parser.add_argument("--gamma",         type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end",   type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    # FP
    parser.add_argument("--fp-smoothing",  type=float, default=1.0)
    # MMQ
    parser.add_argument("--minimax-lr",    type=float, default=0.1)
    parser.add_argument("--minimax-gamma", type=float, default=0.95)
    # Board
    parser.add_argument("--board-size",    type=int, default=100)
    parser.add_argument("--die-faces",     type=int, default=6)
    # Reporting
    parser.add_argument("--report-interval", type=int, default=200)
    parser.add_argument("--plot-window",     type=int, default=100)    
    parser.add_argument("--no-plots",        action="store_true")    

    args = parser.parse_args()

    board = BoardConfig(size=args.board_size, die_faces=args.die_faces)
    config = Config(
        n_episodes=args.episodes,
        alpha=args.alpha, gamma=args.gamma,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        fp_smoothing=args.fp_smoothing,
        minimax_lr=args.minimax_lr, minimax_gamma=args.minimax_gamma,
        report_interval=args.report_interval,
        plot_window=args.plot_window    
    )

    ALL = {
        "fp_vs_fp":   ("fp",  "fp"),
        "iql_vs_iql": ("iql", "iql"),
        "mmq_vs_mmq": ("mmq", "mmq"),
        "iql_vs_fp":  ("iql", "fp"),
        "mmq_vs_fp":  ("mmq", "fp"),
        "iql_vs_mmq": ("iql", "mmq"),
    }
    selected = list(ALL.keys()) if "all" in args.matchups else args.matchups

    reporter = Reporter(config, board)

    print("\n" + "=" * 78)
    print("  STOCHASTIC SNAKES & LADDERS SIMULATION")
    print(f"  Board: {board.size} cells | {len(board.snakes)} snakes | "
          f"{len(board.ladders)} ladders | {len(board.power_tiles)} power tiles")
    print(f"  Power tiles: {sorted(board.power_tiles)}")
    print(f"  Nash EQ: Always choose Best-of-Two on power tiles (dominant strategy)")
    print("=" * 78)

    for key in selected:
        t0, t1 = ALL[key]
        run_matchup(t0, t1, board, config, reporter)

    reporter.print_summary()

    if not args.no_plots:
        reporter.generate_plots(os.getcwd())

if __name__ == "__main__":
    main()