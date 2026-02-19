"""
Matching Pennies — Zero-Sum Multi-Agent Simulation
====================================================
Game:
  Two players simultaneously choose Heads (0) or Tails (1).
  - Matcher (Agent 0) wins if both choose the SAME side  → payoff (+1, -1)
  - Mismatcher (Agent 1) wins if they choose DIFFERENT   → payoff (-1, +1)

This is a classic zero-sum game with a unique MIXED Nash Equilibrium:
  Both players randomise 50/50 (p_H = 0.5).

Nash Equilibrium:
  σ* = (0.5, 0.5) for both players.
  Nash distance = Euclidean distance from (p_H_0, p_H_1) to (0.5, 0.5).

Stochasticity source:
  - Agents may use mixed (randomised) strategies.
  - RL agents explore via ε-greedy.

Agent types:
  FP      — Fictitious Play: best-responds to empirical opponent frequency.
             Converges to Nash in zero-sum games (classical result).
  IQL     — Independent Q-Learning with ε-greedy.
  MMQ     — MiniMax-Q: maintains joint Q-table, solves maximin analytically.

Matchups: FP vs FP, IQL vs IQL, MMQ vs MMQ, IQL vs FP, MMQ vs FP, IQL vs MMQ.

Reports:
  - Win rates per agent (rolling + overall)
  - Nash distance convergence
  - Empirical mixed strategy evolution (p_Heads over time)
  - Strategy heatmap (action choice frequency)
  - Comparison bar charts across matchups
"""

import numpy as np
import random
import argparse
import json
import csv
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════
#  PAYOFF MATRIX
# ══════════════════════════════════════════════════════════════════
# Actions: 0 = Heads (H), 1 = Tails (T)
# Rows = Agent0 (Matcher), Cols = Agent1 (Mismatcher)
#
#            H (1)      T (1)
#   H (0)  (+1, -1)  (-1, +1)
#   T (1)  (-1, +1)  (+1, -1)
#
PAYOFF = {
    (0, 0): (+1, -1),
    (0, 1): (-1, +1),
    (1, 0): (-1, +1),
    (1, 1): (+1, -1),
}
ACTIONS = [0, 1]
ACTION_NAMES = {0: "H (Heads)", 1: "T (Tails)"}

# Nash Equilibrium: both play 50/50
NASH_P_HEADS = 0.5

MATCHUP_COLORS = {
    "FP vs FP":   ("#2196F3", "#64B5F6"),
    "IQL vs IQL": ("#E53935", "#EF9A9A"),
    "MMQ vs MMQ": ("#43A047", "#A5D6A7"),
    "IQL vs FP":  ("#FB8C00", "#FFCC80"),
    "MMQ vs FP":  ("#8E24AA", "#CE93D8"),
    "IQL vs MMQ": ("#00897B", "#80CBC4"),
}


# ══════════════════════════════════════════════════════════════════
#  NASH UTILITIES
# ══════════════════════════════════════════════════════════════════

def compute_nash_distance(p0_heads: float, p1_heads: float) -> float:
    """Euclidean distance from empirical (p0_H, p1_H) to Nash (0.5, 0.5)."""
    return float(np.sqrt((p0_heads - NASH_P_HEADS)**2 +
                         (p1_heads - NASH_P_HEADS)**2))


def compute_exploitability(p0_heads: float, p1_heads: float) -> Tuple[float, float]:
    """
    Best-response exploitability for each player.
    How much can each player gain by deviating from their current strategy?
    Agent0 (Matcher): EV of best response minus EV of current strategy.
    """
    # Agent0's EV if opponent plays p1_heads: p1*q0 + (1-p1)*(1-q0) where we optimise q0
    # EV(H) = p1_H*1 + p1_T*(-1) = 2*p1_H - 1
    # EV(T) = p1_H*(-1) + p1_T*(1) = 1 - 2*p1_H
    ev0_H = 2 * p1_heads - 1
    ev0_T = 1 - 2 * p1_heads
    best_ev0 = max(ev0_H, ev0_T)
    curr_ev0 = p0_heads * ev0_H + (1 - p0_heads) * ev0_T
    exploit0 = best_ev0 - curr_ev0   # >= 0

    # Agent1 (Mismatcher): EV(H) = -p0_H + p0_T = 1 - 2*p0_H
    #                       EV(T) = p0_H - p0_T = 2*p0_H - 1
    ev1_H = 1 - 2 * p0_heads
    ev1_T = 2 * p0_heads - 1
    best_ev1 = max(ev1_H, ev1_T)
    curr_ev1 = p1_heads * ev1_H + (1 - p1_heads) * ev1_T
    exploit1 = best_ev1 - curr_ev1

    return exploit0, exploit1


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    n_rounds: int = 2000
    # IQL
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    # FP
    fp_smoothing: float = 1.0
    # MMQ
    minimax_lr: float = 0.1
    minimax_gamma: float = 0.9
    # Reporting
    report_interval: int = 300
    plot_window: int = 100    

    def to_dict(self):
        return self.__dict__.copy()


# ══════════════════════════════════════════════════════════════════
#  BASE AGENT
# ══════════════════════════════════════════════════════════════════

class Agent:
    def __init__(self, name: str, agent_id: int):
        self.name = name
        self.agent_id = agent_id
        # per-round tracking
        self.rewards: List[float] = []
        self.actions: List[int] = []   # raw action log (0=H, 1=T)
        self.total_reward = 0.0
        self.wins = 0

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        self.rewards.append(reward)
        self.actions.append(my_action)
        self.total_reward += reward
        if reward > 0:
            self.wins += 1

    def reset(self):
        self.rewards = []
        self.actions = []
        self.total_reward = 0.0
        self.wins = 0

    def win_rate(self) -> float:
        n = len(self.rewards)
        return self.wins / n if n else 0.0

    def p_heads(self) -> float:
        if not self.actions:
            return 0.5
        return sum(1 for a in self.actions if a == 0) / len(self.actions)

    def __repr__(self):
        return f"{self.name}(id={self.agent_id})"


# ══════════════════════════════════════════════════════════════════
#  FICTITIOUS PLAY
# ══════════════════════════════════════════════════════════════════

class FictitiousPlayAgent(Agent):
    """
    Maintains empirical frequency of opponent's actions.
    Computes best response analytically.

    Agent0 (Matcher):   prefers SAME side as opponent.
      If opp plays H more → play H; if T more → play T.
    Agent1 (Mismatcher): prefers DIFFERENT side.
      If opp plays H more → play T; if T more → play H.
    """
    def __init__(self, agent_id: int, config: Config):
        super().__init__("FP", agent_id)
        self.config = config
        self.opp_counts = [config.fp_smoothing, config.fp_smoothing]  # [H, T]

    def select_action(self) -> int:
        total = sum(self.opp_counts)
        p_H = self.opp_counts[0] / total   # opp prob of Heads
        p_T = self.opp_counts[1] / total

        if self.agent_id == 0:   # Matcher: wants same
            # EV(play H) = p_H * 1 + p_T * (-1) = 2*p_H - 1
            # EV(play T) = p_H * (-1) + p_T * 1 = 1 - 2*p_H
            if p_H > p_T:
                return 0  # H
            elif p_T > p_H:
                return 1  # T
            else:
                return random.choice(ACTIONS)
        else:   # Mismatcher: wants different
            # EV(play H) = p_H*(-1) + p_T*(1) = 1 - 2*p_H
            # EV(play T) = p_H*(1) + p_T*(-1) = 2*p_H - 1
            if p_H > p_T:
                return 1  # T (mismatch H)
            elif p_T > p_H:
                return 0  # H (mismatch T)
            else:
                return random.choice(ACTIONS)

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        super().update(my_action, opp_action, reward, round_num)
        self.opp_counts[opp_action] += 1

    def reset(self):
        super().reset()
        self.opp_counts = [self.config.fp_smoothing, self.config.fp_smoothing]

    def belief(self) -> Dict:
        total = sum(self.opp_counts)
        return {"P(H)": self.opp_counts[0]/total, "P(T)": self.opp_counts[1]/total}


# ══════════════════════════════════════════════════════════════════
#  INDEPENDENT Q-LEARNING
# ══════════════════════════════════════════════════════════════════

class IQLAgent(Agent):
    """
    State = last opponent action (or None at start).
    Q[state][my_action] updated via Q-learning.
    ε-greedy exploration.
    """
    def __init__(self, agent_id: int, config: Config):
        super().__init__("IQL", agent_id)
        self.config = config
        self.epsilon = config.epsilon_start
        self.Q: Dict = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self.state = None      # last opp action
        self._last_action = None

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            q = self.Q[self.state]
            mx = max(q.values())
            action = random.choice([a for a, v in q.items() if v == mx])
        self._last_action = action
        return action

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        super().update(my_action, opp_action, reward, round_num)
        next_state = opp_action
        max_next = max(self.Q[next_state].values())
        td = reward + self.config.gamma * max_next - self.Q[self.state][my_action]
        self.Q[self.state][my_action] += self.config.alpha * td
        self.state = next_state
        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)

    def reset(self):
        super().reset()
        self.epsilon = self.config.epsilon_start
        self.Q = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self.state = None
        self._last_action = None


# ══════════════════════════════════════════════════════════════════
#  MINIMAX-Q
# ══════════════════════════════════════════════════════════════════

class MinimaxQAgent(Agent):
    """
    Joint state (my_last, opp_last).
    Q[s][a_self][a_opp] — minimax value.
    Solves 2×2 maximin analytically.
    Mixed strategy π stored per state.
    """
    def __init__(self, agent_id: int, config: Config):
        super().__init__("MMQ", agent_id)
        self.config = config
        self.epsilon = config.epsilon_start
        self.Q: Dict = defaultdict(
            lambda: {a: {b: 0.0 for b in ACTIONS} for a in ACTIONS}
        )
        self.V: Dict = defaultdict(float)
        self.pi: Dict = defaultdict(lambda: {0: 0.5, 1: 0.5})
        self.state = None
        self._last_action = None

    def _solve_maximin(self, s) -> Dict[int, float]:
        Q = self.Q[s]
        q00, q01 = Q[0][0], Q[0][1]
        q10, q11 = Q[1][0], Q[1][1]
        denom = q00 - q10 - q01 + q11
        if abs(denom) < 1e-10:
            ev0 = 0.5*q00 + 0.5*q01
            ev1 = 0.5*q10 + 0.5*q11
            if ev0 > ev1:   return {0: 1.0, 1: 0.0}
            if ev1 > ev0:   return {0: 0.0, 1: 1.0}
            return {0: 0.5, 1: 0.5}
        p = max(0.0, min(1.0, (q11 - q10) / denom))
        return {0: p, 1: 1.0 - p}

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            pi = self.pi[self.state]
            action = 0 if random.random() < pi[0] else 1
        self._last_action = action
        return action

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        super().update(my_action, opp_action, reward, round_num)
        next_state = (my_action, opp_action)
        s = self.state
        td = (reward + self.config.minimax_gamma * self.V[next_state]
              - self.Q[s][my_action][opp_action])
        self.Q[s][my_action][opp_action] += self.config.minimax_lr * td
        self.pi[s] = self._solve_maximin(s)
        pi = self.pi[s]
        self.V[s] = min(
            sum(pi[a] * self.Q[s][a][b] for a in ACTIONS)
            for b in ACTIONS
        )
        self.state = next_state
        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)

    def reset(self):
        super().reset()
        self.epsilon = self.config.epsilon_start
        self.Q = defaultdict(lambda: {a: {b: 0.0 for b in ACTIONS} for a in ACTIONS})
        self.V = defaultdict(float)
        self.pi = defaultdict(lambda: {0: 0.5, 1: 0.5})
        self.state = None
        self._last_action = None


# ══════════════════════════════════════════════════════════════════
#  GAME
# ══════════════════════════════════════════════════════════════════

class MatchingPenniesGame:
    def __init__(self, agent0: Agent, agent1: Agent, config: Config):
        self.agent0 = agent0
        self.agent1 = agent1
        self.config = config
        self.round_log: List[Dict] = []

    def run(self) -> Dict:
        self.agent0.reset()
        self.agent1.reset()
        self.round_log = []

        for r in range(1, self.config.n_rounds + 1):
            a0 = self.agent0.select_action()
            a1 = self.agent1.select_action()
            r0, r1 = PAYOFF[(a0, a1)]
            self.agent0.update(a0, a1, r0, r)
            self.agent1.update(a1, a0, r1, r)
            self.round_log.append({"round": r, "a0": a0, "a1": a1, "r0": r0, "r1": r1})

        return self._summary()

    def _summary(self) -> Dict:
        n = self.config.n_rounds
        log = self.round_log

        p0h = self.agent0.p_heads()
        p1h = self.agent1.p_heads()
        nd = compute_nash_distance(p0h, p1h)

        # Tail 20%
        tail = log[int(n * 0.8):]
        tp0h = sum(1 for x in tail if x["a0"] == 0) / max(len(tail), 1)
        tp1h = sum(1 for x in tail if x["a1"] == 0) / max(len(tail), 1)
        nd_tail = compute_nash_distance(tp0h, tp1h)
        expl0, expl1 = compute_exploitability(tp0h, tp1h)

        return {
            "agent0": str(self.agent0),
            "agent1": str(self.agent1),
            "n_rounds": n,
            "total_reward_0": self.agent0.total_reward,
            "total_reward_1": self.agent1.total_reward,
            "avg_reward_0": self.agent0.total_reward / n,
            "avg_reward_1": self.agent1.total_reward / n,
            "win_rate_0": self.agent0.win_rate(),
            "win_rate_1": self.agent1.win_rate(),
            "p_heads_0": p0h,
            "p_heads_1": p1h,
            "p_heads_0_tail": tp0h,
            "p_heads_1_tail": tp1h,
            "nash_distance_overall": nd,
            "nash_distance_tail": nd_tail,
            "exploitability_0": expl0,
            "exploitability_1": expl1,
            "joint_HH": sum(1 for x in log if x["a0"]==0 and x["a1"]==0) / n,
            "joint_HT": sum(1 for x in log if x["a0"]==0 and x["a1"]==1) / n,
            "joint_TH": sum(1 for x in log if x["a0"]==1 and x["a1"]==0) / n,
            "joint_TT": sum(1 for x in log if x["a0"]==1 and x["a1"]==1) / n,
        }


# ══════════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════════

def make_agent(t: str, agent_id: int, config: Config) -> Agent:
    t = t.lower()
    if t == "fp":      return FictitiousPlayAgent(agent_id, config)
    if t == "iql":     return IQLAgent(agent_id, config)
    if t in ("mmq", "minimax"): return MinimaxQAgent(agent_id, config)
    raise ValueError(f"Unknown agent: {t}")


# ══════════════════════════════════════════════════════════════════
#  ROLLING HELPERS
# ══════════════════════════════════════════════════════════════════

def rolling(data: List[float], window: int) -> List[float]:
    out = []
    for i in range(len(data)):
        s = max(0, i - window + 1)
        out.append(float(np.mean(data[s:i+1])))
    return out

def rolling_win(rewards: List[float], window: int) -> List[float]:
    wins = [1.0 if r > 0 else 0.0 for r in rewards]
    return rolling(wins, window)

def rolling_p_heads(actions: List[int], window: int) -> List[float]:
    heads = [1.0 if a == 0 else 0.0 for a in actions]
    return rolling(heads, window)


# ══════════════════════════════════════════════════════════════════
#  PLOTTING — PER MATCHUP (7 panels)
# ══════════════════════════════════════════════════════════════════

def plot_matchup(result: Dict, agent0: Agent, agent1: Agent,
                 config: Config, save_path: str):
    matchup = result["matchup"]
    c0, c1 = MATCHUP_COLORS.get(matchup, ("#1565C0", "#C62828"))
    n = result["n_rounds"]
    w = config.plot_window
    rounds = list(range(1, n + 1))

    # Series
    rwr0 = rolling_win(agent0.rewards, w)
    rwr1 = rolling_win(agent1.rewards, w)
    rph0 = rolling_p_heads(agent0.actions, w)
    rph1 = rolling_p_heads(agent1.actions, w)
    rrew0 = rolling(agent0.rewards, w)
    rrew1 = rolling(agent1.rewards, w)

    # Nash distance series (per-round rolling)
    nd_series = [compute_nash_distance(rph0[i], rph1[i]) for i in range(n)]

    # Exploitability series
    expl0_s = [compute_exploitability(rph0[i], rph1[i])[0] for i in range(n)]
    expl1_s = [compute_exploitability(rph0[i], rph1[i])[1] for i in range(n)]

    # Cumulative rewards
    cum0 = list(np.cumsum(agent0.rewards))
    cum1 = list(np.cumsum(agent1.rewards))

    # Joint heatmap
    joint = np.array([
        [result["joint_HH"], result["joint_HT"]],
        [result["joint_TH"], result["joint_TT"]],
    ])

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

    tkw = dict(fontsize=10, fontweight="bold", color="#212121", pad=8)
    lkw = dict(fontsize=8.5, color="#424242")

    # ── [0] Rolling Win Rate ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor("#F5F5F5")
    ax.plot(rounds, rwr0, color=c0, lw=1.6, label=f"Agent0 — {result['agent0']} (Matcher)")
    ax.plot(rounds, rwr1, color=c1, lw=1.6, ls="--", label=f"Agent1 — {result['agent1']} (Mismatcher)")
    ax.axhline(0.5, color="#9E9E9E", lw=1.0, ls=":", alpha=0.8, label="Nash win rate (50%)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Rolling Win Rate  (window={w})", **tkw)
    ax.set_xlabel("Round", **lkw); ax.set_ylabel("Win Rate", **lkw)
    ax.legend(fontsize=8, framealpha=0.75); ax.grid(True, alpha=0.3, ls=":")
    ax.annotate(f"{rwr0[-1]:.2%}", xy=(n, rwr0[-1]),
                xytext=(-50, 6), textcoords="offset points", fontsize=8, color=c0)
    ax.annotate(f"{rwr1[-1]:.2%}", xy=(n, rwr1[-1]),
                xytext=(-50, -14), textcoords="offset points", fontsize=8, color=c1)

    # ── [1] Cumulative Reward ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2:])
    ax.set_facecolor("#F5F5F5")
    ax.plot(rounds, cum0, color=c0, lw=1.6, label=f"Agent0 — {result['agent0']}")
    ax.plot(rounds, cum1, color=c1, lw=1.6, ls="--", label=f"Agent1 — {result['agent1']}")
    ax.axhline(0, color="#9E9E9E", lw=0.9, ls=":", alpha=0.7, label="Zero-sum baseline")
    ax.set_title("Cumulative Reward", **tkw)
    ax.set_xlabel("Round", **lkw); ax.set_ylabel("Σ Reward", **lkw)
    ax.legend(fontsize=8, framealpha=0.75); ax.grid(True, alpha=0.3, ls=":")
    ax.annotate(f"{cum0[-1]:+.0f}", xy=(n, cum0[-1]),
                xytext=(-40, 5), textcoords="offset points", fontsize=8, color=c0)
    ax.annotate(f"{cum1[-1]:+.0f}", xy=(n, cum1[-1]),
                xytext=(-40, -13), textcoords="offset points", fontsize=8, color=c1)

    # ── [2] p(Heads) Evolution ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.set_facecolor("#F5F5F5")
    ax.plot(rounds, rph0, color=c0, lw=1.5, label=f"Agent0 — {result['agent0']}")
    ax.plot(rounds, rph1, color=c1, lw=1.5, ls="--", label=f"Agent1 — {result['agent1']}")
    ax.axhline(NASH_P_HEADS, color="#B71C1C", lw=1.2, ls="--",
               alpha=0.8, label=f"Nash EQ: p(H)={NASH_P_HEADS:.1f}")
    ax.fill_between(rounds,
                    [NASH_P_HEADS - 0.05]*n, [NASH_P_HEADS + 0.05]*n,
                    alpha=0.08, color="#B71C1C", label="±5% Nash band")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title(f"Empirical p(Heads) over Time  (window={w})", **tkw)
    ax.set_xlabel("Round", **lkw); ax.set_ylabel("P(Heads)", **lkw)
    ax.legend(fontsize=8, framealpha=0.75); ax.grid(True, alpha=0.3, ls=":")

    # ── [3] Nash Distance ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    ax.set_facecolor("#F5F5F5")
    nd_col = "#6A1B9A"
    ax.plot(rounds, nd_series, color=nd_col, lw=1.4, alpha=0.85, label="Nash Distance")
    ax.fill_between(rounds, nd_series, alpha=0.12, color=nd_col)
    ax.axhline(0.0, color="#B71C1C", lw=1.2, ls="--", alpha=0.8, label="Nash EQ (dist=0)")
    ax.axhline(np.sqrt(2)/2, color="#1B5E20", lw=0.8, ls=":", alpha=0.6,
               label="Max dist (√2/2 ≈ 0.707)")
    final_nd = nd_series[-1]
    ax.annotate(f"Final: {final_nd:.3f}", xy=(n, final_nd),
                xytext=(-70, 8 if final_nd < 0.5 else -16),
                textcoords="offset points", fontsize=8.5, color=nd_col,
                arrowprops=dict(arrowstyle="->", color=nd_col, lw=0.8))
    ax.set_ylim(-0.02, np.sqrt(2)/2 + 0.08)
    ax.set_title("Nash Distance  ||σ − σ*||₂", **tkw)
    ax.set_xlabel("Round", **lkw); ax.set_ylabel("Distance to Nash EQ", **lkw)
    ax.legend(fontsize=8, framealpha=0.75); ax.grid(True, alpha=0.3, ls=":")

    # ── [4] Exploitability ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    ax.set_facecolor("#F5F5F5")
    ax.plot(rounds, expl0_s, color=c0, lw=1.4,
            label=f"Exploit. Agent0 ({result['agent0']})")
    ax.plot(rounds, expl1_s, color=c1, lw=1.4, ls="--",
            label=f"Exploit. Agent1 ({result['agent1']})")
    ax.axhline(0.0, color="#B71C1C", lw=1.0, ls="--", alpha=0.7, label="Nash (0 exploitability)")
    ax.set_title("Exploitability  (best-response gap)", **tkw)
    ax.set_xlabel("Round", **lkw); ax.set_ylabel("Max gain from deviation", **lkw)
    ax.set_ylim(-0.02, 1.1)
    ax.legend(fontsize=8, framealpha=0.75); ax.grid(True, alpha=0.3, ls=":")

    # ── [5] Joint Action Heatmap ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(joint, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto")
    labels = ["H (0)", "T (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Agent1 ({result['agent1']}) Action", **lkw)
    ax.set_ylabel(f"Agent0 ({result['agent0']}) Action", **lkw)
    ax.set_title("Joint Action Distribution", **tkw)
    for i in range(2):
        for j in range(2):
            val = joint[i, j]
            ax.text(j, i, f"{val:.2%}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val > 0.3 else "#212121")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Nash EQ diagonal (H,H) and (T,T) highlighted for Matcher wins
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False,
                            edgecolor="#1B5E20", lw=2.5, label="Matcher wins"))
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False,
                            edgecolor="#1B5E20", lw=2.5))
    ax.legend(fontsize=7, loc="upper right")

    # ── [6] Rolling Average Reward ───────────────────────────────────────
    ax = fig.add_subplot(gs[2, 3])
    ax.set_facecolor("#F5F5F5")
    ax.plot(rounds, rrew0, color=c0, lw=1.4, label=f"Agent0 ({result['agent0']})")
    ax.plot(rounds, rrew1, color=c1, lw=1.4, ls="--", label=f"Agent1 ({result['agent1']})")
    ax.axhline(0.0, color="#B71C1C", lw=1.0, ls="--", alpha=0.7, label="Nash avg reward (0)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Rolling Avg Reward  (window={w})", **tkw)
    ax.set_xlabel("Round", **lkw); ax.set_ylabel("Avg Reward", **lkw)
    ax.legend(fontsize=8, framealpha=0.75); ax.grid(True, alpha=0.3, ls=":")

    # Super-title
    fig.suptitle(
        f"Matching Pennies — {matchup}  |  Rounds: {n}  |  "
        f"p(H): A0={result['p_heads_0']:.3f} A1={result['p_heads_1']:.3f}  |  "
        f"Nash Dist (overall/tail): {result['nash_distance_overall']:.3f} / "
        f"{result['nash_distance_tail']:.3f}",
        fontsize=12, fontweight="bold", color="#212121", y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Chart saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
#  PLOTTING — COMPARISON
# ══════════════════════════════════════════════════════════════════

def plot_comparison(all_results: List[Dict], config: Config, save_path: str):
    matchups = [r["matchup"] for r in all_results]
    n = len(matchups)
    x = np.arange(n)
    bw = 0.32
    short = [m.replace(" vs ", "\nvs\n") for m in matchups]
    pal0 = [MATCHUP_COLORS.get(m, ("#1565C0","#C62828"))[0] for m in matchups]
    pal1 = [MATCHUP_COLORS.get(m, ("#1565C0","#C62828"))[1] for m in matchups]

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.patch.set_facecolor("#FAFAFA")
    tkw = dict(fontsize=10, fontweight="bold", color="#212121", pad=7)

    def annotate(ax, bars):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # ── Win Rate ─────────────────────────────────────────────────────────
    ax = axes[0, 0]; ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x-bw/2, [r["win_rate_0"] for r in all_results], bw, color=pal0, alpha=0.85, edgecolor="white", label="Agent0")
    b1 = ax.bar(x+bw/2, [r["win_rate_1"] for r in all_results], bw, color=pal1, alpha=0.85, edgecolor="white", label="Agent1")
    annotate(ax, b0); annotate(ax, b1)
    ax.axhline(0.5, color="#B71C1C", lw=1, ls="--", alpha=0.6, label="Nash (50%)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Overall Win Rate", **tkw); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── p(Heads) overall ─────────────────────────────────────────────────
    ax = axes[0, 1]; ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x-bw/2, [r["p_heads_0"] for r in all_results], bw, color=pal0, alpha=0.85, edgecolor="white", label="Agent0")
    b1 = ax.bar(x+bw/2, [r["p_heads_1"] for r in all_results], bw, color=pal1, alpha=0.85, edgecolor="white", label="Agent1")
    annotate(ax, b0); annotate(ax, b1)
    ax.axhline(0.5, color="#B71C1C", lw=1.2, ls="--", alpha=0.8, label="Nash p(H)=0.5")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Empirical p(Heads) — Overall", **tkw); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── p(Heads) tail ────────────────────────────────────────────────────
    ax = axes[0, 2]; ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x-bw/2, [r["p_heads_0_tail"] for r in all_results], bw, color=pal0, alpha=0.85, edgecolor="white", label="Agent0 tail")
    b1 = ax.bar(x+bw/2, [r["p_heads_1_tail"] for r in all_results], bw, color=pal1, alpha=0.85, edgecolor="white", label="Agent1 tail")
    annotate(ax, b0); annotate(ax, b1)
    ax.axhline(0.5, color="#B71C1C", lw=1.2, ls="--", alpha=0.8, label="Nash p(H)=0.5")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Empirical p(Heads) — Last 20%", **tkw); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── Nash Distance ─────────────────────────────────────────────────────
    ax = axes[1, 0]; ax.set_facecolor("#F5F5F5")
    w2 = 0.3
    bo = ax.bar(x-w2/2, [r["nash_distance_overall"] for r in all_results], w2,
                color="#7B1FA2", alpha=0.85, edgecolor="white", label="Overall")
    bt = ax.bar(x+w2/2, [r["nash_distance_tail"] for r in all_results], w2,
                color="#AB47BC", alpha=0.85, edgecolor="white", label="Tail 20%")
    annotate(ax, bo); annotate(ax, bt)
    ax.axhline(0, color="#B71C1C", lw=1.2, ls="--", alpha=0.8, label="Nash EQ")
    ax.axhline(np.sqrt(2)/2, color="#1B5E20", lw=0.8, ls=":", alpha=0.6, label="Max (√2/2)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Nash Distance  ||σ − σ*||₂", **tkw)
    ax.set_ylim(0, np.sqrt(2)/2 + 0.12)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── Exploitability ────────────────────────────────────────────────────
    ax = axes[1, 1]; ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x-bw/2, [r["exploitability_0"] for r in all_results], bw, color=pal0, alpha=0.85, edgecolor="white", label="Agent0")
    b1 = ax.bar(x+bw/2, [r["exploitability_1"] for r in all_results], bw, color=pal1, alpha=0.85, edgecolor="white", label="Agent1")
    annotate(ax, b0); annotate(ax, b1)
    ax.axhline(0, color="#B71C1C", lw=1.2, ls="--", alpha=0.8, label="Nash (0 exploit.)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Exploitability (tail 20%)", **tkw); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    # ── Avg Reward ───────────────────────────────────────────────────────
    ax = axes[1, 2]; ax.set_facecolor("#F5F5F5")
    b0 = ax.bar(x-bw/2, [r["avg_reward_0"] for r in all_results], bw, color=pal0, alpha=0.85, edgecolor="white", label="Agent0")
    b1 = ax.bar(x+bw/2, [r["avg_reward_1"] for r in all_results], bw, color=pal1, alpha=0.85, edgecolor="white", label="Agent1")
    annotate(ax, b0); annotate(ax, b1)
    ax.axhline(0, color="#B71C1C", lw=1.0, ls="--", alpha=0.7, label="Nash avg reward (0)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_title("Average Reward per Round", **tkw); ax.set_ylim(-0.6, 0.6)
    ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.3, ls=":")

    fig.suptitle(
        f"Matching Pennies — All Matchups Comparison  (rounds={config.n_rounds})",
        fontsize=14, fontweight="bold", color="#212121", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Comparison chart saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
#  REPORTER
# ══════════════════════════════════════════════════════════════════

class Reporter:
    def __init__(self, config: Config):
        self.config = config
        self.results: List[Dict] = []
        self.agents: List[Tuple[Agent, Agent]] = []

    def add(self, matchup: str, result: Dict, a0: Agent, a1: Agent):
        result["matchup"] = matchup
        self.results.append(result)
        self.agents.append((a0, a1))

    def _interpret(self, nd: float, expl0: float, expl1: float) -> str:
        if nd < 0.03:   return "Converged to Nash EQ (50/50 mix)"
        if nd < 0.10:   return "Very near Nash EQ"
        if nd < 0.25:   return f"Near Nash — exploit: A0={expl0:.3f} A1={expl1:.3f}"
        return               f"Far from Nash — exploit: A0={expl0:.3f} A1={expl1:.3f}"

    def print_interval(self, matchup: str, rnd: int, r: Dict):
        print(f"  [{matchup}] Round {rnd:>6} | "
              f"WR0={r['win_rate_0']:.2%}  WR1={r['win_rate_1']:.2%} | "
              f"p(H)0={r['p_heads_0']:.3f}  p(H)1={r['p_heads_1']:.3f} | "
              f"NashDist={r['nash_distance_overall']:.4f}")

    def print_summary(self):
        sep = "=" * 78
        print(f"\n{sep}")
        print("  MATCHING PENNIES — SIMULATION REPORT")
        print(sep)
        print(f"  Rounds : {self.config.n_rounds}")
        print(f"  Payoff : (H,H)=(+1,-1)  (H,T)=(-1,+1)  (T,H)=(-1,+1)  (T,T)=(+1,-1)")
        print(f"  Nash EQ: Mixed — both play p(H)=0.5  |  Avg reward = 0 for both")
        print(sep)

        for r in self.results:
            nd_o = r["nash_distance_overall"]
            nd_t = r["nash_distance_tail"]
            e0, e1 = r["exploitability_0"], r["exploitability_1"]
            print(f"\n  ┌─ {r['matchup']}")
            print(f"  │  Agent0 ({r['agent0']:>4})  "
                  f"WinRate={r['win_rate_0']:.2%}  AvgRew={r['avg_reward_0']:+.4f}  "
                  f"p(H)={r['p_heads_0']:.3f}  (tail: {r['p_heads_0_tail']:.3f})")
            print(f"  │  Agent1 ({r['agent1']:>4})  "
                  f"WinRate={r['win_rate_1']:.2%}  AvgRew={r['avg_reward_1']:+.4f}  "
                  f"p(H)={r['p_heads_1']:.3f}  (tail: {r['p_heads_1_tail']:.3f})")
            print(f"  │  Joint: HH={r['joint_HH']:.2%}  HT={r['joint_HT']:.2%}  "
                  f"TH={r['joint_TH']:.2%}  TT={r['joint_TT']:.2%}")
            print(f"  │  Nash Dist (overall): {nd_o:.4f}")
            print(f"  │  Nash Dist (tail)  : {nd_t:.4f}")
            print(f"  │  Exploitability    : A0={e0:.4f}  A1={e1:.4f}")
            print(f"  └  {self._interpret(nd_t, e0, e1)}")

        print(f"\n{sep}\n")

    def generate_plots(self, plots_dir: str):
        os.makedirs(plots_dir, exist_ok=True)
        print(f"\n  Generating charts /")
        for result, (a0, a1) in zip(self.results, self.agents):
            safe = result["matchup"].replace(" vs ", "_vs_").lower()
            path = os.path.join(plots_dir, f"{safe}.png")
            plot_matchup(result, a0, a1, self.config, path)
        if len(self.results) > 1:
            path = os.path.join(plots_dir, "comparison.png")
            plot_comparison(self.results, self.config, path)


# ══════════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════

def run_matchup(t0: str, t1: str, config: Config, reporter: Reporter) -> str:
    nmap = {"fp": "FP", "iql": "IQL", "mmq": "MMQ", "minimax": "MMQ"}
    matchup = f"{nmap.get(t0.lower(), t0.upper())} vs {nmap.get(t1.lower(), t1.upper())}"
    print(f"\n  Running: {matchup}")

    a0 = make_agent(t0, 0, config)
    a1 = make_agent(t1, 1, config)
    game = MatchingPenniesGame(a0, a1, config)
    a0.reset(); a1.reset()
    game.round_log = []

    for r in range(1, config.n_rounds + 1):
        a0_act = a0.select_action()
        a1_act = a1.select_action()
        r0, r1 = PAYOFF[(a0_act, a1_act)]
        a0.update(a0_act, a1_act, r0, r)
        a1.update(a1_act, a0_act, r1, r)
        game.round_log.append({"round": r, "a0": a0_act, "a1": a1_act, "r0": r0, "r1": r1})

        if r % config.report_interval == 0 or r == config.n_rounds:
            interim = game._summary()
            reporter.print_interval(matchup, r, interim)

    result = game._summary()
    reporter.add(matchup, result, a0, a1)
    return matchup


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Matching Pennies Zero-Sum — FP / IQL / MiniMax-Q",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rounds",         type=int,   default=2000)
    parser.add_argument("--matchups",       nargs="+",
                        choices=["fp_vs_fp","iql_vs_iql","mmq_vs_mmq",
                                 "iql_vs_fp","mmq_vs_fp","iql_vs_mmq","all"],
                        default=["all"])
    parser.add_argument("--alpha",          type=float, default=0.1)
    parser.add_argument("--gamma",          type=float, default=0.9)
    parser.add_argument("--epsilon-start",  type=float, default=1.0)
    parser.add_argument("--epsilon-end",    type=float, default=0.05)
    parser.add_argument("--epsilon-decay",  type=float, default=0.997)
    parser.add_argument("--fp-smoothing",   type=float, default=1.0)
    parser.add_argument("--minimax-lr",     type=float, default=0.1)
    parser.add_argument("--minimax-gamma",  type=float, default=0.9)
    parser.add_argument("--report-interval",type=int,   default=300)
    parser.add_argument("--plot-window",    type=int,   default=100)    
    parser.add_argument("--no-plots",       action="store_true")    

    args = parser.parse_args()

    config = Config(
        n_rounds=args.rounds,
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

    reporter = Reporter(config)

    print("\n" + "=" * 78)
    print("  MATCHING PENNIES — ZERO-SUM SIMULATION")
    print("  Agent0 = Matcher   wins on (H,H) or (T,T)")
    print("  Agent1 = Mismatcher wins on (H,T) or (T,H)")
    print("  Nash EQ: both randomise 50/50  →  avg reward = 0 for both")
    print("=" * 78)

    for key in selected:
        t0, t1 = ALL[key]
        run_matchup(t0, t1, config, reporter)

    reporter.print_summary()

    if not args.no_plots:
        reporter.generate_plots(os.getcwd())

if __name__ == "__main__":
    main()