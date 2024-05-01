"""Heuristics for policy search."""
from __future__ import annotations

import abc
import logging
from typing import Sequence, Set

from pg3 import utils
from pg3.structs import GroundAtom, LiftedDecisionList, Object, \
    PlanningFailure, Task
from pg3.trajectory_gen import _TrajectoryGenerator


class _PG3Heuristic(abc.ABC):
    """Given an LDL policy, produce a score, with lower better."""
    _num_calls = 0

    def __init__(
        self,
        trajectory_gen: _TrajectoryGenerator,
        train_tasks: Sequence[Task],
        horizon: int,
    ) -> None:
        self._trajectory_gen = trajectory_gen
        self._train_tasks = train_tasks
        self._horizon = horizon

    def __call__(self, ldl: LiftedDecisionList) -> float:
        """Compute the heuristic value for the given LDL policy."""
        _PG3Heuristic._num_calls += 1
        score = 0.0
        for task in self._train_tasks:
            score += self._get_score_for_task(ldl, task)
        logging.debug(f"Scoring:\n{ldl}\nScore: {score}")
        return score

    @abc.abstractmethod
    def _get_score_for_task(self, ldl: LiftedDecisionList,
                            task: Task) -> float:
        """Produce a score, with lower better."""
        raise NotImplementedError("Override me!")


class _PolicyEvaluationPG3Heuristic(_PG3Heuristic):
    """Score a policy based on the number of train tasks it solves at the
    abstract level."""

    def _get_score_for_task(self, ldl: LiftedDecisionList,
                            task: Task) -> float:
        if self._ldl_solves_abstract_task(ldl, task.init, task.objects,
                                          task.goal):
            return 0.0
        return 1.0

    def _ldl_solves_abstract_task(self, ldl: LiftedDecisionList,
                                  atoms: Set[GroundAtom], objects: Set[Object],
                                  goal: Set[GroundAtom]) -> bool:
        for _ in range(self._horizon):
            if goal.issubset(atoms):
                return True
            ground_op = utils.query_ldl(ldl, atoms, objects, goal)
            if ground_op is None:
                return False
            atoms = utils.apply_operator(ground_op, atoms)
        return goal.issubset(atoms)


class _PlanComparisonPG3Heuristic(_PG3Heuristic):
    """Score a policy based on agreement with certain plans.

    Which plans are used to compute agreement is defined by subclasses.
    """
    _plan_compare_inapplicable_cost: float = 0.99

    def _get_score_for_task(self, ldl: LiftedDecisionList,
                            task: Task) -> float:
        try:
            _, atom_plan, _ = self._trajectory_gen.get_trajectory_for_task(
                task, ldl)
        except PlanningFailure:
            return self._horizon  # worst possible score
        # Note: we need the goal because it's an input to the LDL policy.
        return self._count_missed_steps(ldl, atom_plan, task.objects,
                                        task.goal)

    def _count_missed_steps(self, ldl: LiftedDecisionList,
                            atoms_seq: Sequence[Set[GroundAtom]],
                            objects: Set[Object],
                            goal: Set[GroundAtom]) -> float:
        missed_steps = 0.0
        for t in range(len(atoms_seq) - 1):
            ground_op = utils.query_ldl(ldl, atoms_seq[t], objects, goal)
            if ground_op is None:
                missed_steps += self._plan_compare_inapplicable_cost
            else:
                predicted_atoms = utils.apply_operator(ground_op, atoms_seq[t])
                if predicted_atoms != atoms_seq[t + 1]:
                    missed_steps += 1
        return missed_steps
