"""Nervous System — 17 biological principles for tool orchestration.

NOT a CEO. NOT an orchestrator. A NERVOUS SYSTEM:
- Senses state (proprioception)
- Accumulates signals (summation)
- Fires at threshold (all-or-none)
- Tools excite/inhibit each other
- Habituation dampens repetition
- Sensitization amplifies novelty
- Reflexes bypass deep reasoning
- Homeostasis balances activity
- Predictions filter surprises
- Lateral inhibition sharpens focus
- Neuromodulators adjust learning
- Autonomic modes shift explore↔integrate
"""

from __future__ import annotations

import math
from collections import Counter

from sparks.state import CognitiveState, NervousSignals, Phase


# ─── Main Sense Function ───


def sense(state: CognitiveState) -> NervousSignals:
    """Read shared state → update all signal potentials. No commands."""
    signals = state.signals

    # 0. Decay all potentials toward resting (natural relaxation)
    signals.convergence_potential.decay(0.05)
    signals.contradiction_potential.decay(0.05)
    signals.diminishing_potential.decay(0.05)
    signals.anomaly_potential.decay(0.05)
    signals.sufficient_potential.decay(0.05)

    # ─── 1. Convergence: accumulate from structural similarity ───
    if state.round >= 1 and state.round - 1 in state.snapshots:
        prev = state.snapshots[state.round - 1]
        count_score = 1.0 - abs(len(state.principles) - prev.principle_count) / max(len(state.principles), prev.principle_count, 1)

        curr_words = set()
        for p in state.principles:
            curr_words.update(p.statement.lower().split())
        prev_words = set()
        for s in prev.principle_statements:
            prev_words.update(s.lower().split())

        jaccard = len(curr_words & prev_words) / max(len(curr_words | prev_words), 1) if curr_words else 0.0
        conv_signal = count_score * 0.4 + jaccard * 0.6

        # Excitation from convergence evidence
        signals.convergence_potential.contribute(conv_signal * 0.5)

        # Time summation: consecutive convergent rounds boost
        if conv_signal > 0.4:
            signals.convergence_potential.contribute(0.1)  # temporal summation

    # ─── 2. Contradiction: accumulate from unresolved conflicts ───
    unresolved = sum(1 for c in state.contradictions if not c.resolved)
    if unresolved > 0:
        signals.contradiction_potential.contribute(0.3 * min(unresolved, 3))
    else:
        signals.contradiction_potential.contribute(-0.2)  # inhibition: no contradictions

    # ─── 3. Diminishing returns: accumulate from pattern discovery rate ───
    if state.round >= 1 and state.round - 1 in state.snapshots:
        prev = state.snapshots[state.round - 1]
        new_patterns = len(state.patterns) - prev.pattern_count
        if new_patterns < 2:
            signals.diminishing_potential.contribute(0.3)  # excitation: slowing down
        else:
            signals.diminishing_potential.contribute(-0.2)  # inhibition: still finding stuff

    # ─── 4. Anomaly: accumulate from model failures ───
    for m in state.model_results:
        if m.failures:
            signals.anomaly_potential.contribute(0.2 * len(m.failures))

    # ─── Fire potentials 1-4 BEFORE using their state ───
    signals.convergence_potential.check_fire()
    signals.contradiction_potential.check_fire()
    signals.diminishing_potential.check_fire()
    signals.anomaly_potential.check_fire()

    # ─── 5. Sufficient depth: requires convergence (now current) + principles ───
    if signals.convergence_potential.fired and len(state.principles) >= 1:
        signals.sufficient_potential.contribute(0.5)
    if state.round >= 2:
        signals.sufficient_potential.contribute(0.1)  # time pressure
    signals.sufficient_potential.check_fire()

    # ─── 6. Habituation: repeated pattern types lose weight ───
    type_counts = Counter(p.type for p in state.patterns)
    for ptype, count in type_counts.items():
        weight = 1.0 / (1.0 + math.log1p(count))  # 1st: 1.0, 3rd: 0.59, 10th: 0.30
        signals.habituation[ptype] = weight

    # ─── 7. Sensitization: anomaly boosts related channel sensitivity ───
    # (applied in observe tool via state.signals.anomaly)

    # ─── 8. Neuromodulation (AFTER check_fire so .fired is current) ───
    _update_neuromodulators(state, signals)

    # ─── 9. Autonomic mode ───
    _update_autonomic_mode(state, signals)

    # ─── 10. Homeostasis: detect tool activity imbalance ───
    _check_homeostasis(state, signals)

    # ─── 11. Proprioception: update self-awareness ───
    signals.round = state.round
    signals.phase = state.phase

    return signals


# ─── Neuromodulation ───


def _update_neuromodulators(state: CognitiveState, signals: NervousSignals):
    """Dopamine (reward), NE (explore/exploit), ACh (learning rate)."""
    mod = signals.modulators

    # Dopamine: reward prediction error
    # Did this round produce more/fewer principles than expected?
    if state.round >= 1 and state.round - 1 in state.snapshots:
        prev = state.snapshots[state.round - 1]
        expected = prev.principle_count
        actual = len(state.principles)
        mod.dopamine = min(1.0, max(-1.0, (actual - expected) * 0.3))

    # Norepinephrine: exploration vs exploitation
    # Many new patterns → keep exploring, few → time to exploit
    if state.patterns:
        absent_ratio = sum(1 for p in state.patterns if p.type == "absent") / len(state.patterns)
        if absent_ratio > 0.3:
            mod.norepinephrine = min(1.0, mod.norepinephrine + 0.1)  # explore more
        elif signals.diminishing_potential.fired:
            mod.norepinephrine = max(0.0, mod.norepinephrine - 0.2)  # exploit

    # Acetylcholine: precision / learning rate
    # High when predictions are uncertain (early rounds)
    # Low when converged (late rounds)
    if signals.convergence_potential.fired:
        mod.acetylcholine = max(0.1, mod.acetylcholine - 0.2)
    else:
        mod.acetylcholine = min(0.9, mod.acetylcholine + 0.1)


# ─── Autonomic Mode ───


def _update_autonomic_mode(state: CognitiveState, signals: NervousSignals):
    """Sympathetic (explore/diverge) vs Parasympathetic (integrate/converge)."""
    auto = signals.autonomic

    if signals.anomaly_potential.fired or signals.contradiction_potential.fired:
        auto.mode = "sympathetic"  # crisis → explore, scan wide
    elif signals.convergence_potential.fired:
        auto.mode = "parasympathetic"  # stable → integrate, go deep
    elif signals.diminishing_potential.fired:
        auto.mode = "parasympathetic"  # slowing → consolidate
    else:
        auto.mode = "balanced"

    # Vagal tone = how easily the system switches modes
    # High vagal tone = flexible, healthy
    if not hasattr(state, '_mode_history'):
        state._mode_history = []
    state._mode_history.append(auto.mode)
    history = state._mode_history
    if len(history) >= 2:
        recent = history[-min(5, len(history)):]
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        auto.vagal_tone = min(1.0, changes * 0.2 + 0.3)


# ─── Homeostasis ───


def _check_homeostasis(state: CognitiveState, signals: NervousSignals):
    """Detect and flag tool activity imbalance."""
    activity = signals.tool_activity
    if not activity:
        return

    avg = sum(activity.values()) / max(len(activity), 1)
    for tool, count in activity.items():
        if count > avg * 2:
            # Overactive tool → inhibit (via negative reward)
            signals.tool_rewards[tool] = signals.tool_rewards.get(tool, 0) - 0.1
        elif count < avg * 0.5 and count > 0:
            # Underactive tool → excite
            signals.tool_rewards[tool] = signals.tool_rewards.get(tool, 0) + 0.1


# ─── Reflexes (fast path, no LLM) ───


def reflex_check(state: CognitiveState) -> list[str]:
    """Immediate reflexes — bypass deep reasoning."""
    reflexes = []

    # Budget exhaustion → stop immediately
    if hasattr(state, 'cost_tracker'):
        tracker = state.cost_tracker
        if hasattr(tracker, 'remaining') and tracker.remaining() < 0.05:
            reflexes.append("STOP:budget_exhausted")

    # Zero principles after full round → data may be insufficient
    if state.round >= 1 and len(state.principles) == 0:
        reflexes.append("WARN:no_principles_extracted")

    # High-confidence absent pattern → immediate alert
    for p in state.patterns:
        if p.type == "absent" and p.confidence > 0.9:
            reflexes.append(f"ALERT:strong_absence:{p.id}")

    return reflexes


# ─── Prediction Error Filter ───


def filter_by_prediction(
    observations: list,
    predictions: list[str],
    threshold: float = 0.3,
) -> tuple[list, list]:
    """Predictive coding: separate predicted (boring) from surprising.

    Returns: (surprising_observations, predicted_observations)
    """
    if not predictions:
        return observations, []  # no predictions yet → everything is new

    predictions_lower = " ".join(predictions).lower()
    pred_words = set(predictions_lower.split())

    surprising = []
    predicted = []

    for obs in observations:
        obs_words = set(obs.content.lower().split())
        if not obs_words:
            surprising.append(obs)
            continue

        overlap = len(obs_words & pred_words) / len(obs_words)
        if overlap > (1.0 - threshold):
            # High overlap with predictions → boring, predicted
            obs.confidence *= 0.5  # reduce weight
            predicted.append(obs)
        else:
            # Low overlap → surprising, prediction error!
            obs.confidence = min(1.0, obs.confidence * 1.3)  # boost weight
            surprising.append(obs)

    return surprising, predicted


# ─── Lateral Inhibition (Tool Competition) ───


def competitive_tool_selection(
    tools: dict,
    state: CognitiveState,
    max_active: int = 5,
) -> list[str]:
    """Winner-take-all: most relevant tools inhibit the rest.

    Each tool computes relevance → top K win → rest suppressed.
    """
    relevance = {}

    for name, tool in tools.items():
        score = 0.5  # base

        # Reward from neuromodulation
        reward = state.signals.tool_rewards.get(name, 0)
        score += reward

        # Homeostatic boost for underactive tools
        activity = state.signals.tool_activity.get(name, 0)
        avg_activity = sum(state.signals.tool_activity.values()) / max(len(state.signals.tool_activity), 1) if state.signals.tool_activity else 1
        if activity < avg_activity * 0.5:
            score += 0.2  # underactive → boost

        # Autonomic mode influence
        if state.signals.autonomic.mode == "sympathetic":
            if name in ("play", "transform", "imagine", "empathize"):
                score += 0.3  # explore tools boosted
        elif state.signals.autonomic.mode == "parasympathetic":
            if name in ("abstract", "synthesize", "model", "analogize"):
                score += 0.3  # integration tools boosted

        relevance[name] = score

    # Winner-take-all: top K, but always keep synthesize
    sorted_tools = sorted(relevance, key=relevance.get, reverse=True)
    selected = sorted_tools[:max_active]

    # Synthesize is always active (it's the final integration)
    if "synthesize" not in selected and "synthesize" in tools:
        selected.append("synthesize")

    return selected


# ─── 12. Hebbian Plasticity: "fire together, wire together" ───


def update_synapses(state: CognitiveState, tool_name: str, success: bool):
    """After a tool runs, strengthen/weaken connections based on outcome.

    STDP: if tool A fired before tool B and result was good → A→B strengthens.
    """
    signals = state.signals
    fired = signals.last_fired

    if len(fired) < 2:
        fired.append(tool_name)
        return

    prev_tool = fired[-1]
    key = f"{prev_tool}→{tool_name}"

    current = signals.synapses.get(key, 1.0)

    if success:
        # LTP: strengthen the connection (capped at 2.0)
        signals.synapses[key] = min(2.0, current * 1.1 + 0.05)
    else:
        # LTD: weaken the connection (floored at 0.1)
        signals.synapses[key] = max(0.1, current * 0.9 - 0.05)

    fired.append(tool_name)

    # Prune very weak connections (synapse elimination)
    weak = [k for k, v in signals.synapses.items() if v < 0.2]
    for k in weak:
        del signals.synapses[k]


# ─── 13. Population Coding: signal tracks who contributed ───


def record_signal_contributor(signals: NervousSignals, signal_name: str, tool_name: str):
    """Track which tools contributed to each signal — not just the sum."""
    contributors = signals.signal_contributors.get(signal_name, [])
    if tool_name not in contributors:
        contributors.append(tool_name)
    signals.signal_contributors[signal_name] = contributors


# ─── 14. Recurrent Feedback: tool re-activation mid-round ───


def check_feedback_needed(state: CognitiveState) -> list[str]:
    """Check if any tool should be re-activated based on current results.

    Like cortex feedback: higher area tells lower area "look again at this".
    """
    reactivate = []
    fired = state.signals.last_fired

    # If model failed → re-observe the failure areas
    for m in state.model_results:
        if m.failures and (not fired or "observe" not in fired[-2:]):
            reactivate.append("observe")
            break

    # If abstract produced too few principles → need more patterns
    if state.principles and len(state.principles) < 2:
        if not fired or "recognize_patterns" not in fired[-2:]:
            reactivate.append("recognize_patterns")

    # If contradictions found → analogize might help resolve
    if state.contradictions and any(not c.resolved for c in state.contradictions):
        if not fired or "analogize" not in fired[-3:]:
            reactivate.append("analogize")

    return reactivate


# ─── 15. Sleep/Consolidation: selective cleanup between rounds ───


def consolidate(state: CognitiveState):
    """Sleep-like consolidation between rounds.

    NOT clean_slate (delete everything). Selective:
    - Strengthen strong connections (replay)
    - Weaken weak connections (pruning)
    - Remove noise observations (cleanup)
    - Merge similar patterns (compression)
    """
    signals = state.signals

    # 1. Synapse replay: strengthen connections that actually fired this round
    fired = signals.last_fired
    if fired and state.principles:
        for i in range(len(fired) - 1):
            key = f"{fired[i]}→{fired[i + 1]}"
            signals.synapses[key] = min(2.0, signals.synapses.get(key, 1.0) + 0.05)

    # 2. Prune low-confidence observations (metabolic waste cleanup)
    before = len(state.observations)
    state.observations = [
        o for o in state.observations
        if o.confidence > 0.3 or o.channel == "absence"  # keep absences always
    ]
    pruned = before - len(state.observations)

    # 3. Merge duplicate patterns (compression)
    seen_descriptions = {}
    unique_patterns = []
    for p in state.patterns:
        key = p.description.lower().strip()
        if key not in seen_descriptions:
            seen_descriptions[key] = p
            unique_patterns.append(p)
        else:
            # Merge: keep higher confidence
            existing = seen_descriptions[key]
            existing.confidence = max(existing.confidence, p.confidence)
    merged = len(state.patterns) - len(unique_patterns)
    state.patterns = unique_patterns

    # 4. Reset tool activity counters (new day)
    signals.tool_activity = {}
    signals.last_fired = []

    # 5. Decay all habituation slightly (partial recovery from habituation during sleep)
    for key in signals.habituation:
        signals.habituation[key] = min(1.0, signals.habituation[key] * 1.1)

    state.signals.consolidation_needed = False

    return {"pruned_observations": pruned, "merged_patterns": merged}


# ─── 16. Stochastic Resonance: useful noise ───


def stochastic_boost(signal_value: float, noise_level: float = 0.1) -> float:
    """Add controlled noise to help detect weak signals.

    In real neurons, background noise helps sub-threshold signals cross the threshold.
    Without noise, weak-but-real patterns are invisible.
    """
    import random
    noise = random.gauss(0, noise_level)
    return signal_value + noise


# ─── 17. Oscillation-based tool grouping ───


def get_active_rhythm_group(state: CognitiveState) -> str:
    """Which rhythm group should be most active now?

    Theta (sensory) → Beta (processing) → Gamma (integration) → Alpha (exploration)
    Like brain rhythms that cycle through phases.
    """
    n_obs = len(state.observations)
    n_pat = len(state.patterns)
    n_prin = len(state.principles)

    if n_obs == 0:
        return "sensory"  # need data first
    elif n_pat < 3:
        return "sensory" if n_obs < 20 else "processing"
    elif n_prin == 0:
        return "processing"
    elif state.signals.convergence_potential.fired:
        return "integration"
    elif state.signals.diminishing_potential.fired:
        return "exploration"  # stuck → try creative tools
    else:
        return "processing"  # default: keep analyzing


def boost_rhythm_group(state: CognitiveState, tools: dict) -> dict[str, float]:
    """Boost tools in the active rhythm group, dampen others."""
    active_group = get_active_rhythm_group(state)
    group_tools = state.signals.rhythm_groups.get(active_group, [])

    boosts = {}
    for name in tools:
        if name in group_tools:
            boosts[name] = 0.3  # in-rhythm boost
        else:
            boosts[name] = -0.1  # out-of-rhythm dampening

    return boosts


# ─── Should Continue ───


def should_continue(state: CognitiveState, max_rounds: int) -> bool:
    """Should we do another iteration round?"""
    # Reflex: hard limits
    if state.round >= max_rounds:
        return False

    # Threshold-based: sufficient depth fired?
    if state.signals.sufficient_depth:
        return False

    # Convergence fired AND past minimum rounds
    if state.signals.convergence and state.round >= 2:
        return False

    return True
