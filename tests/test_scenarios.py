"""Tests for server.scenarios — procedural scenario generation."""
from __future__ import annotations

import pytest

from server.scenarios import (
    CRITICAL_BY_FAMILY,
    DIFFICULTIES,
    FAMILIES,
    FIELD_VOCAB,
    MAX_QUESTIONS,
    MAX_STEPS_BY_DIFFICULTY,
    N_FIELDS_RANGE_BY_DIFFICULTY,
    REQUIRED_KEYS_BY_FAMILY,
    TASK_FIELDS,
    generate,
)


class TestFieldVocab:
    def test_every_task_field_has_vocab(self):
        for family, fields in TASK_FIELDS.items():
            for f in fields:
                assert f in FIELD_VOCAB, f"{family}.{f} missing from FIELD_VOCAB"

    def test_every_critical_field_in_task_fields(self):
        for family, crit in CRITICAL_BY_FAMILY.items():
            valid = set(TASK_FIELDS[family])
            for f in crit:
                assert f in valid, f"critical {f} not in TASK_FIELDS[{family}]"

    def test_every_required_key_in_task_fields(self):
        for family, rk in REQUIRED_KEYS_BY_FAMILY.items():
            valid = set(TASK_FIELDS[family])
            for f in rk:
                assert f in valid, f"required_key {f} not in TASK_FIELDS[{family}]"


class TestGenerate:
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_all_difficulties_work(self, difficulty):
        sc = generate(seed=42, task_id=difficulty)
        assert sc["task_id"] == difficulty
        assert sc["family"] in FAMILIES
        assert sc["max_steps"] == MAX_STEPS_BY_DIFFICULTY[difficulty]
        assert sc["max_questions"] == MAX_QUESTIONS

    @pytest.mark.parametrize("family_idx", range(len(FAMILIES)))
    def test_all_families_reachable(self, family_idx):
        for seed in range(200):
            sc = generate(seed=seed, task_id="medium")
            if sc["family"] == FAMILIES[family_idx]:
                return
        pytest.fail(f"Family {FAMILIES[family_idx]} never generated in 200 seeds")

    def test_reproducibility(self):
        a = generate(seed=123, task_id="medium")
        b = generate(seed=123, task_id="medium")
        assert a == b

    def test_different_seeds_differ(self):
        a = generate(seed=1, task_id="medium")
        b = generate(seed=2, task_id="medium")
        assert a != b

    def test_profile_values_from_vocab(self):
        for seed in range(50):
            for diff in DIFFICULTIES:
                sc = generate(seed=seed, task_id=diff)
                for k, v in sc["hidden_profile"].items():
                    assert v in FIELD_VOCAB[k], f"seed={seed} {diff}: {k}={v} not in vocab"

    def test_required_keys_always_in_profile_medium_hard(self):
        for seed in range(100):
            for diff in ("medium", "hard"):
                sc = generate(seed=seed, task_id=diff)
                rk = set(REQUIRED_KEYS_BY_FAMILY[sc["family"]])
                profile_keys = set(sc["hidden_profile"].keys())
                missing = rk - profile_keys
                assert not missing, (
                    f"seed={seed} {diff} {sc['family']}: required keys {missing} missing from profile"
                )

    def test_field_count_in_range(self):
        for seed in range(100):
            for diff in DIFFICULTIES:
                sc = generate(seed=seed, task_id=diff)
                n = len(sc["hidden_profile"])
                lo, hi = N_FIELDS_RANGE_BY_DIFFICULTY[diff]
                pool_size = len(TASK_FIELDS[sc["family"]])
                expected_lo = min(lo, pool_size)
                assert n >= expected_lo, f"seed={seed} {diff} {sc['family']}: {n} fields < {expected_lo}"
                assert n <= pool_size, f"seed={seed} {diff}: {n} fields > pool {pool_size}"

    def test_critical_fields_subset_of_profile(self):
        for seed in range(50):
            for diff in DIFFICULTIES:
                sc = generate(seed=seed, task_id=diff)
                for cf in sc["critical_fields"]:
                    assert cf in sc["hidden_profile"], (
                        f"seed={seed} {diff}: critical field {cf} not in profile"
                    )

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            generate(seed=0, task_id="nightmare")
