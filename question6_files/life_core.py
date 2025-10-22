# Game of Life implementation for this assignment.
# Random initialisation, state hashing, oscillation detection, glider detection.

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Dict, List


def next_state_bounded(grid: np.ndarray) -> np.ndarray:
    """
    Next generation with bounded edges (no wrap).
    Implemented via zero-padding + 8 aligned slices (no broadcasting issues).
    Returns uint8 array of shape grid.shape with 0/1 values.
    """
    g = grid.astype(np.uint8)
    P = np.pad(g, 1, mode="constant")  # pad with zeros around

    # sum 8 neighbours from the padded array, aligned to the inner (original) region
    nb = (
        P[0:-2, 0:-2] + P[0:-2, 1:-1] + P[0:-2, 2:] +
        P[1:-1, 0:-2]                 + P[1:-1, 2:] +
        P[2:  , 0:-2] + P[2:  , 1:-1] + P[2:  , 2:]
    )

    new = ((g == 1) & ((nb == 2) | (nb == 3))) | ((g == 0) & (nb == 3))
    return new.astype(np.uint8)



def random_grid(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli(p) random grid of shape (n,n), dtype=uint8."""
    return (rng.random((n, n)) < p).astype(np.uint8)


def grid_hash(grid: np.ndarray) -> int:
    """Fast-ish hash: view as bytes and hash(). Good enough for cycle detection here."""
    # only compare equality within a run.
    return hash(grid.tobytes())

# ----- Oscillation & still-life detection -----

@dataclass
class CycleDetector:
    """Tracks recent states to detect repeats. Stores first-seen step per hash."""
    window: int = 200  # large enough for our purposes
    seen: Dict[int, int] = None
    order: List[int] = None

    def __post_init__(self):
        self.seen = {}
        self.order = []

    def update(self, step: int, h: int) -> Optional[int]:
        """
        Record hash h at time 'step'.
        If h was seen before, return period = step - first_seen.
        Otherwise return None.
        evict old hashes past the sliding 'window'.
        """
        if h in self.seen:
            return step - self.seen[h]

        self.seen[h] = step
        self.order.append(h)
        if len(self.order) > self.window:
            old = self.order.pop(0)
            # only evict if the stored index matches
            if self.seen.get(old, None) is not None and self.seen[old] < step - self.window:
                self.seen.pop(old, None)
        return None

# ----- Glider detection -----
# Detect classic 5-cell gliders by matching small 3x3 masks.
# There are 4 phases in a given direction; rotating those phases covers all directions.
# Prebuild a set of unique masks to check.

def _rotate90(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, k=1)

def _all_glider_masks() -> List[np.ndarray]:
    # Phases for one glider orientation (3x3 windows); 1 = live, 0 = dead/ignore.
    # These are the 4 consecutive phases of the standard glider (moving diagonally).
    phases = []
    phases.append(np.array([[0,1,0],
                            [0,0,1],
                            [1,1,1]], dtype=np.uint8))
    phases.append(np.array([[1,0,1],
                            [0,1,1],
                            [0,1,0]], dtype=np.uint8))
    phases.append(np.array([[0,0,1],
                            [1,0,1],
                            [0,1,1]], dtype=np.uint8))
    phases.append(np.array([[0,1,0],
                            [1,0,1],
                            [0,1,1]], dtype=np.uint8))

    # Now add rotations (0,90,180,270) and dedupe.
    masks = []
    for ph in phases:
        cur = ph.copy()
        for _ in range(4):
            masks.append(cur.copy())
            cur = _rotate90(cur)

    # Deduplicate masks by bytes
    unique = {}
    for m in masks:
        unique[m.tobytes()] = m
    return list(unique.values())

_GLIDER_MASKS = _all_glider_masks()

def detect_glider_once(grid: np.ndarray) -> bool:
    """
    Return True if any 3x3 *bounded* window exactly equals a known glider phase.
    (Exact match: the 3x3 must be exactly the mask—no extra live cells.)
    """
    n = grid.shape[0]
    if n < 3:
        return False
    for i in range(n - 2):
        for j in range(n - 2):
            w = grid[i:i+3, j:j+3]
            for m in _GLIDER_MASKS:
                if np.array_equal(w, m):
                    return True
    return False


# ----- Single-run simulation with event logging -----

@dataclass
class RunOutcome:
    outcome: str              # "extinct" | "still" | "oscillating" | "active_end"
    t_event: Optional[int]    # time step when the outcome was first observed
    period: Optional[int]     # for oscillating
    glider_seen: int          # 0/1
    t_glider: Optional[int]   # first time glider detected


def simulate_once(n: int, p: float, T: int, rng: np.random.Generator) -> RunOutcome:
    g = random_grid(n, p, rng)
    cyc = CycleDetector(window=200)

    glider_seen = 0
    t_glider: Optional[int] = None

    # treat t=0 as the initial state.
    h0 = grid_hash(g)
    cyc.update(0, h0)

    if detect_glider_once(g):
        glider_seen = 1
        t_glider = 0

    for t in range(1, T+1):
        g_next = next_state_bounded(g)
        # outcomes: extinction
        if g_next.sum() == 0:
            # note: detect glider in the last alive state as well (optional)
            return RunOutcome("extinct", t, None, glider_seen, t_glider)

        # still life: next equals current
        if np.array_equal(g_next, g):
            return RunOutcome("still", t, None, glider_seen, t_glider)

        # oscillation check via cycle detector
        h = grid_hash(g_next)
        per = cyc.update(t, h)
        if per is not None and per >= 2:
            return RunOutcome("oscillating", t, per, glider_seen, t_glider)

        # glider detection
        if glider_seen == 0 and detect_glider_once(g_next):
            glider_seen = 1
            t_glider = t

        g = g_next

    # If this is reached, we ran out of time without a terminal label.
    # Lets call it "active_end" to mean it neither died, nor froze, nor became periodic within T.
    return RunOutcome("active_end", T, None, glider_seen, t_glider)
