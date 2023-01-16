"""Utility functions."""

import functools
import itertools
from typing import Collection, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set

from pg3.structs import GroundAtom, LDLRule, LiftedAtom, LiftedDecisionList, \
    Object, ObjectOrVariable, Predicate, Type, Variable, _GroundLDLRule, \
    _GroundSTRIPSOperator


def all_ground_ldl_rules(
    rule: LDLRule,
    objects: Collection[Object],
    static_predicates: Optional[Collection[Predicate]] = None,
    init_atoms: Optional[Collection[GroundAtom]] = None
) -> List[_GroundLDLRule]:
    """Get all possible groundings of the given rule with the given objects.

    If provided, use the static predicates and init_atoms to avoid
    grounding rules that will never have satisfied preconditions in any
    state.
    """
    if static_predicates is None:
        static_predicates = set()
    if init_atoms is None:
        init_atoms = set()
    return _cached_all_ground_ldl_rules(rule, frozenset(objects),
                                        frozenset(static_predicates),
                                        frozenset(init_atoms))


@functools.lru_cache(maxsize=None)
def _cached_all_ground_ldl_rules(
        rule: LDLRule, objects: FrozenSet[Object],
        static_predicates: FrozenSet[Predicate],
        init_atoms: FrozenSet[GroundAtom]) -> List[_GroundLDLRule]:
    """Helper for all_ground_ldl_rules() that caches the outputs."""
    ground_rules = []
    # Use static preconds to reduce the map of parameters to possible objects.
    # For example, if IsBall(?x) is a positive state precondition, then only
    # the objects that appear in init_atoms with IsBall could bind to ?x.
    # For now, we just check unary static predicates, since that covers the
    # common case where such predicates are used in place of types.
    # Create map from each param to unary static predicates.
    param_to_pos_preds: Dict[Variable, Set[Predicate]] = {
        p: set()
        for p in rule.parameters
    }
    param_to_neg_preds: Dict[Variable, Set[Predicate]] = {
        p: set()
        for p in rule.parameters
    }
    for (preconditions, param_to_preds) in [
        (rule.pos_state_preconditions, param_to_pos_preds),
        (rule.neg_state_preconditions, param_to_neg_preds),
    ]:
        for atom in preconditions:
            pred = atom.predicate
            if pred in static_predicates and pred.arity == 1:
                param = atom.variables[0]
                param_to_preds[param].add(pred)
    # Create the param choices, filtering based on the unary static atoms.
    param_choices = []  # list of lists of possible objects for each param
    # Preprocess the atom sets for faster lookups.
    init_atom_tups = {(a.predicate, tuple(a.objects)) for a in init_atoms}
    for param in rule.parameters:
        choices = []
        for obj in objects:
            # Types must match, as usual.
            if obj.type != param.type:
                continue
            # Check the static conditions.
            binding_valid = True
            for pred in param_to_pos_preds[param]:
                if (pred, (obj, )) not in init_atom_tups:
                    binding_valid = False
                    break
            for pred in param_to_neg_preds[param]:
                if (pred, (obj, )) in init_atom_tups:
                    binding_valid = False
                    break
            if binding_valid:
                choices.append(obj)
        # Must be sorted for consistency with other grounding code.
        param_choices.append(sorted(choices))
    for choice in itertools.product(*param_choices):
        ground_rule = rule.ground(choice)
        ground_rules.append(ground_rule)
    return ground_rules


def query_ldl(
    ldl: LiftedDecisionList,
    atoms: Set[GroundAtom],
    objects: Set[Object],
    goal: Set[GroundAtom],
    static_predicates: Optional[Set[Predicate]] = None,
    init_atoms: Optional[Collection[GroundAtom]] = None
) -> Optional[_GroundSTRIPSOperator]:
    """Queries a lifted decision list representing a goal-conditioned policy.

    Given an abstract state and goal, the rules are grounded in order.
    The first applicable ground rule is used to return a ground
    operator. If static_predicates is provided, it is used to avoid
    grounding rules with nonsense preconditions like IsBall(robot). If
    no rule is applicable, returns None.
    """
    for rule in ldl.rules:
        for ground_rule in all_ground_ldl_rules(
                rule,
                objects,
                static_predicates=static_predicates,
                init_atoms=init_atoms):
            if ground_rule.pos_state_preconditions.issubset(atoms) and \
               not ground_rule.neg_state_preconditions & atoms and \
               ground_rule.goal_preconditions.issubset(goal):
                return ground_rule.ground_operator
    return None


def _get_entity_combinations(
        entities: Collection[ObjectOrVariable],
        types: Sequence[Type]) -> Iterator[List[ObjectOrVariable]]:
    """Get all combinations of entities satisfying the given types sequence."""
    sorted_entities = sorted(entities)
    choices = []
    for vt in types:
        this_choices = []
        for ent in sorted_entities:
            if ent.is_instance(vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


def get_variable_combinations(
        variables: Collection[Variable],
        types: Sequence[Type]) -> Iterator[List[Variable]]:
    """Get all combinations of objects satisfying the given types sequence."""
    return _get_entity_combinations(variables, types)


def get_all_lifted_atoms_for_predicate(
        predicate: Predicate,
        variables: FrozenSet[Variable]) -> Set[LiftedAtom]:
    """Get all groundings of the predicate given variables.

    Note: we don't want lru_cache() on this function because we might want
    to call it with stripped predicates, and we wouldn't want it to return
    cached values.
    """
    lifted_atoms = set()
    for args in get_variable_combinations(variables, predicate.types):
        lifted_atom = LiftedAtom(predicate, args)
        lifted_atoms.add(lifted_atom)
    return lifted_atoms


def create_new_variables(
    types: Sequence[Type],
    existing_vars: Optional[Collection[Variable]] = None,
    var_prefix: str = "?x",
) -> List[Variable]:
    """Create new variables of the given types, avoiding name collisions with
    existing variables. By convention, all new variables are of the form.

    <var_prefix><number>.
    """
    pre_len = len(var_prefix)
    existing_var_nums = set()
    if existing_vars:
        for v in existing_vars:
            if v.name.startswith(var_prefix) and v.name[pre_len:].isdigit():
                existing_var_nums.add(int(v.name[pre_len:]))
    if existing_var_nums:
        counter = itertools.count(max(existing_var_nums) + 1)
    else:
        counter = itertools.count(0)
    new_vars = []
    for t in types:
        new_var_name = f"{var_prefix}{next(counter)}"
        new_var = Variable(new_var_name, t)
        new_vars.append(new_var)
    return new_vars
