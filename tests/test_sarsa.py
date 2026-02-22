"""Tests for SARSA scheduler — Q-table updates, action selection, serialization."""

import json
import os
import tempfile

import pytest
from math_mirror.mcp.mirror_train import SARSAScheduler


@pytest.fixture
def scheduler():
    return SARSAScheduler(alpha=0.1, gamma=0.99, epsilon=0.1)


# ── Q-table basics ────────────────────────────────────────

class TestQTable:
    def test_initial_q_zero(self, scheduler):
        assert scheduler._q('s0', 'continue') == 0.0
        assert scheduler._q('nonexistent', 'branch') == 0.0

    def test_set_q(self, scheduler):
        scheduler._set_q('s0', 'deploy', 1.5)
        assert scheduler._q('s0', 'deploy') == 1.5

    def test_different_state_action_pairs(self, scheduler):
        scheduler._set_q('s0', 'continue', 1.0)
        scheduler._set_q('s0', 'branch', 2.0)
        scheduler._set_q('s1', 'continue', 3.0)
        assert scheduler._q('s0', 'continue') == 1.0
        assert scheduler._q('s0', 'branch') == 2.0
        assert scheduler._q('s1', 'continue') == 3.0


# ── SARSA update ──────────────────────────────────────────

class TestSARSAUpdate:
    def test_basic_update(self, scheduler):
        """Q(s,a) += alpha * [r + gamma*Q(s',a') - Q(s,a)]"""
        scheduler.sarsa_update('s0', 'continue', 1.0, 's1', 'continue')
        # Q was 0, reward=1, Q(s1,continue)=0
        # new Q = 0 + 0.1 * (1.0 + 0.99*0 - 0) = 0.1
        assert abs(scheduler._q('s0', 'continue') - 0.1) < 1e-10

    def test_update_with_existing_q(self, scheduler):
        scheduler._set_q('s0', 'continue', 0.5)
        scheduler._set_q('s1', 'branch', 0.3)
        scheduler.sarsa_update('s0', 'continue', 1.0, 's1', 'branch')
        # Q = 0.5 + 0.1 * (1.0 + 0.99*0.3 - 0.5)
        # Q = 0.5 + 0.1 * (1.0 + 0.297 - 0.5) = 0.5 + 0.1*0.797 = 0.5797
        assert abs(scheduler._q('s0', 'continue') - 0.5797) < 1e-10

    def test_convergence_to_stable(self, scheduler):
        """Repeated updates with same reward should converge."""
        for _ in range(1000):
            scheduler.sarsa_update('s', 'a', 1.0, 's', 'a')
        q = scheduler._q('s', 'a')
        # Should converge to r / (1 - gamma) = 1 / 0.01 = 100
        assert q > 50  # approaching 100

    def test_zero_reward(self, scheduler):
        scheduler.sarsa_update('s0', 'deploy', 0.0, 's1', 'deploy')
        assert scheduler._q('s0', 'deploy') == 0.0


# ── Action selection ──────────────────────────────────────

class TestActionSelection:
    def test_greedy_selects_best(self):
        """With epsilon=0, always selects best action."""
        scheduler = SARSAScheduler(epsilon=0.0)
        scheduler._set_q('s0', 'continue', 0.1)
        scheduler._set_q('s0', 'branch', 0.5)
        scheduler._set_q('s0', 'deploy', 0.3)
        assert scheduler.select_action('s0') == 'branch'

    def test_random_with_full_epsilon(self):
        """With epsilon=1.0, always explores."""
        scheduler = SARSAScheduler(epsilon=1.0)
        scheduler._set_q('s0', 'deploy', 100.0)
        # Over many trials, should see different actions
        actions = set()
        for _ in range(100):
            actions.add(scheduler.select_action('s0'))
        assert len(actions) > 1  # not always the same

    def test_action_in_valid_set(self, scheduler):
        action = scheduler.select_action('s0')
        assert action in scheduler.actions

    def test_unknown_state_still_works(self, scheduler):
        # No Q-values set for this state — should still return valid action
        action = scheduler.select_action('never_seen')
        assert action in scheduler.actions


# ── Serialization ─────────────────────────────────────────

class TestSerialization:
    def test_save_load_roundtrip(self, scheduler):
        scheduler._set_q('s0', 'continue', 1.5)
        scheduler._set_q('s1', 'deploy', 2.7)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            scheduler.save_q_table(path)

            # Load into new scheduler
            new_scheduler = SARSAScheduler()
            new_scheduler.load_q_table(path)

            assert abs(new_scheduler._q('s0', 'continue') - 1.5) < 1e-10
            assert abs(new_scheduler._q('s1', 'deploy') - 2.7) < 1e-10
            assert new_scheduler._q('s0', 'branch') == 0.0  # not set
        finally:
            os.unlink(path)

    def test_save_format_is_json(self, scheduler):
        scheduler._set_q('s0', 'continue', 1.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            scheduler.save_q_table(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert 's0|continue' in data
        finally:
            os.unlink(path)

    def test_empty_q_table(self, scheduler):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            scheduler.save_q_table(path)
            new_scheduler = SARSAScheduler()
            new_scheduler.load_q_table(path)
            assert len(new_scheduler.q_table) == 0
        finally:
            os.unlink(path)


# ── Actions list ──────────────────────────────────────────

class TestActions:
    def test_actions_exist(self, scheduler):
        assert 'continue' in scheduler.actions
        assert 'branch' in scheduler.actions
        assert 'deploy' in scheduler.actions
        assert len(scheduler.actions) == 3
