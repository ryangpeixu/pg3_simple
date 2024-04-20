"""Policy search operators."""
from __future__ import annotations

import abc
import functools
from typing import FrozenSet, Iterator, List, Set

from pg3 import utils
from pg3.structs import LDLRule, LiftedAtom, LiftedDecisionList, Predicate, \
    STRIPSOperator, Variable


class _PG3SearchOperator(abc.ABC):
    """Given an LDL policy, generate zero or more successor LDL policies."""

    def __init__(self,
                 predicates: Set[Predicate],
                 operators: Set[STRIPSOperator],
                 allow_new_vars: bool = True) -> None:
        self._predicates = predicates
        self._operators = operators
        self._allow_new_vars = allow_new_vars

    @abc.abstractmethod
    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        """Generate zero or more successor LDL policies."""
        raise NotImplementedError("Override me!")


class _AddRulePG3SearchOperator(_PG3SearchOperator):
    """An operator that adds new rules to an existing LDL policy.

    Only add new rules to the end of a policy. This does not change
    completeness and helps with reducing the branching factor of policy
    search.
    """

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule in self._get_candidate_rules():
            new_rules = list(ldl.rules) + [rule]
            yield LiftedDecisionList(new_rules)

    @functools.lru_cache(maxsize=None)
    def _get_candidate_rules(self) -> List[LDLRule]:
        return [self._operator_to_rule(op) for op in sorted(self._operators)]

    @staticmethod
    def _operator_to_rule(operator: STRIPSOperator) -> LDLRule:
        """Initialize an LDLRule from an operator."""
        return LDLRule(
            name=operator.name,
            parameters=list(operator.parameters),
            pos_state_preconditions=set(operator.preconditions),
            neg_state_preconditions=set(),
            goal_preconditions=set(),
            operator=operator,
        )


class _AddConditionPG3SearchOperator(_PG3SearchOperator):
    """An operator that adds new preconditions to existing LDL rules."""

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule_idx, rule in enumerate(ldl.rules):
            rule_vars = frozenset(rule.parameters)
            for condition in self._get_candidate_conditions(rule_vars):
                # Consider adding new condition to positive preconditions,
                # negative preconditions, or goal preconditions.
                for destination in ["pos", "neg", "goal"]:
                    new_pos = set(rule.pos_state_preconditions)
                    new_neg = set(rule.neg_state_preconditions)
                    new_goal = set(rule.goal_preconditions)
                    if destination == "pos":
                        dest_set = new_pos
                    elif destination == "neg":
                        dest_set = new_neg
                    else:
                        assert destination == "goal"
                        dest_set = new_goal
                    # If the condition already exists, skip.
                    if condition in dest_set:
                        continue
                    # Special case: if the condition already exists in the
                    # positive preconditions, don't add to the negative
                    # preconditions, and vice versa.
                    if destination in ("pos", "neg") and condition in \
                        new_pos | new_neg:
                        continue
                    dest_set.add(condition)
                    parameters = sorted({
                        v
                        for c in new_pos | new_neg | new_goal
                        for v in c.variables
                    } | set(rule.operator.parameters))
                    # Create the new rule.
                    new_rule = LDLRule(
                        name=rule.name,
                        parameters=parameters,
                        pos_state_preconditions=new_pos,
                        neg_state_preconditions=new_neg,
                        goal_preconditions=new_goal,
                        operator=rule.operator,
                    )
                    # Create the new LDL.
                    new_rules = list(ldl.rules)
                    new_rules[rule_idx] = new_rule
                    yield LiftedDecisionList(new_rules)

    @functools.lru_cache(maxsize=None)
    def _get_candidate_conditions(
            self, variables: FrozenSet[Variable]) -> List[LiftedAtom]:
        conditions = []
        for pred in sorted(self._predicates):
            # Allow or disallow creation of fresh variables.
            condition_vars = variables
            if self._allow_new_vars:
                new_vars = utils.create_new_variables(pred.types, variables)
                condition_vars |= frozenset(new_vars)
            for condition in utils.get_all_lifted_atoms_for_predicate(
                    pred, condition_vars):
                unique_vars = set([v.name for v in condition.variables])
                if len(unique_vars) != len(condition.variables):
                    continue
                conditions.append(condition)
        return conditions


class _DeleteConditionPG3SearchOperator(_PG3SearchOperator):
    """An operator that removes conditions from existing LDL rules."""

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule_idx, rule in enumerate(ldl.rules):
            for condition in sorted(rule.pos_state_preconditions | \
                rule.neg_state_preconditions | rule.goal_preconditions):

                # If the condition to be removed is a
                # precondition of an operator, don't remove it.
                if condition in rule.operator.preconditions:
                    continue

                # Recreate new preconditions.
                # Assumes that a condition can appear only in one set
                new_pos = rule.pos_state_preconditions - {condition}
                new_neg = rule.neg_state_preconditions - {condition}
                new_goal = rule.goal_preconditions - {condition}

                # Reconstruct parameters from the other
                # components of the LDL.
                all_atoms = new_pos | new_neg | new_goal
                new_rule_params_set = \
                    {v for a in all_atoms for v in a.variables}
                new_rule_params_set.update(rule.operator.parameters)
                new_rule_params = sorted(new_rule_params_set)

                # Create the new rule.
                new_rule = LDLRule(
                    name=rule.name,
                    parameters=new_rule_params,
                    pos_state_preconditions=new_pos,
                    neg_state_preconditions=new_neg,
                    goal_preconditions=new_goal,
                    operator=rule.operator,
                )
                # Create the new LDL.
                new_rules = list(ldl.rules)
                new_rules[rule_idx] = new_rule
                yield LiftedDecisionList(new_rules)


class _DeleteRulePG3SearchOperator(_PG3SearchOperator):
    """An operator that removes entire rules from existing LDL rules."""

    def get_successors(
            self, ldl: LiftedDecisionList) -> Iterator[LiftedDecisionList]:
        for rule_idx in range(len(ldl.rules)):
            new_rules = [r for i, r in enumerate(ldl.rules) if i != rule_idx]
            yield LiftedDecisionList(new_rules)
