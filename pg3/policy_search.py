"""PG3 policy search."""

from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

from typing_extensions import TypeAlias

from pg3 import utils
from pg3.heuristics import _PG3Heuristic, _PlanComparisonPG3Heuristic, \
    _PolicyEvaluationPG3Heuristic
from pg3.operators import _AddConditionPG3SearchOperator, \
    _AddRulePG3SearchOperator, _PG3SearchOperator
from pg3.search import run_gbfs, run_hill_climbing
from pg3.structs import LiftedDecisionList, Predicate, STRIPSOperator, Task, \
    Type
from pg3.trajectory_gen import _PolicyGuidedPlanningTrajectoryGenerator, \
    _StaticPlanningTrajectoryGenerator, _TrajectoryGenerator, \
    _UserSuppliedDemoTrajectoryGenerator


def learn_policy(domain_str: str,
                 problem_strs: List[str],
                 horizon: int,
                 demos: Optional[List[List[str]]] = None,
                 max_rule_params: int = 8,
                 heuristic_name: str = "policy_guided",
                 search_method: str = "hill_climbing",
                 task_planning_heuristic: str = "lmcut",
                 max_policy_guided_rollout: int = 50,
                 gbfs_max_expansions: int = 100,
                 hc_enforced_depth: int = 0,
                 allow_new_vars: bool = True,
                 initial_policy_strs: Optional[List[str]] = None) -> str:
    """Outputs a string representation of a lifted decision list."""
    types, predicates, operators = utils.parse_pddl_domain(domain_str)
    train_tasks = [
        utils.pddl_problem_str_to_task(problem_str, domain_str, types,
                                       predicates)
        for problem_str in problem_strs
    ]
    if demos is not None:
        assert len(demos) == len(problem_strs), "Supply one demo per problem."
        assert heuristic_name == "demo_plan_comparison", \
            ("Only supply demos if using demo_plan_comparison heuristic, and "
             "even then, the demos are optional.")
        user_supplied_demos = dict(zip(train_tasks, demos))
    else:
        user_supplied_demos = None
    ldl = _run_policy_search(types, predicates, operators, train_tasks,
                             horizon, user_supplied_demos, max_rule_params,
                             heuristic_name, search_method,
                             task_planning_heuristic,
                             max_policy_guided_rollout, gbfs_max_expansions,
                             hc_enforced_depth, allow_new_vars,
                             initial_policy_strs)
    return str(ldl)

def score_policy(domain_str: str,
                 problem_strs: List[str],
                 horizon: int,
                 demos: Optional[List[List[str]]] = None,
                 max_rule_params: int = 8,
                 heuristic_name: str = "policy_guided",
                 search_method: str = "hill_climbing",
                 task_planning_heuristic: str = "lmcut",
                 max_policy_guided_rollout: int = 50,
                 gbfs_max_expansions: int = 100,
                 hc_enforced_depth: int = 0,
                 allow_new_vars: bool = True,
                 initial_policy_strs: Optional[List[str]] = None) -> str:
    """Outputs a string representation of a lifted decision list."""
    types, predicates, operators = utils.parse_pddl_domain(domain_str)
    train_tasks = [
        utils.pddl_problem_str_to_task(problem_str, domain_str, types,
                                       predicates)
        for problem_str in problem_strs
    ]
    if demos is not None:
        assert len(demos) == len(problem_strs), "Supply one demo per problem."
        assert heuristic_name == "demo_plan_comparison", \
            ("Only supply demos if using demo_plan_comparison heuristic, and "
             "even then, the demos are optional.")
        user_supplied_demos = dict(zip(train_tasks, demos))
    else:
        user_supplied_demos = None
    
    trajectory_generator = _create_trajectory_generator(
        heuristic_name, predicates, operators, task_planning_heuristic,
        max_policy_guided_rollout, user_supplied_demos)
    heuristic = _create_heuristic(heuristic_name, trajectory_generator, train_tasks, horizon)
    initial_states = []
    if initial_policy_strs is None:
        initial_states = [LiftedDecisionList([])]
    else:
        initial_states = [
            utils.parse_ldl_from_str(l, types, predicates, operators)
            for l in initial_policy_strs
        ]
    policy_scores = [heuristic(policy) for policy in initial_states]
    return policy_scores

def _run_policy_search(
        types: Set[Type],
        predicates: Set[Predicate],
        operators: Set[STRIPSOperator],
        train_tasks: Sequence[Task],
        horizon: int,
        user_supplied_demos: Optional[Dict[Task, List[str]]] = None,
        max_rule_params: int = 8,
        heuristic_name: str = "policy_guided",
        search_method: str = "hill_climbing",
        task_planning_heuristic: str = "lmcut",
        max_policy_guided_rollout: int = 50,
        gbfs_max_expansions: int = 100,
        hc_enforced_depth: int = 0,
        allow_new_vars: bool = True,
        initial_policy_strs: Optional[List[str]] = None) -> LiftedDecisionList:
    """Search for a lifted decision list policy that solves the training
    tasks."""
    # Set up a search over LDL space.
    _S: TypeAlias = LiftedDecisionList
    # An "action" here is a search operator and an integer representing the
    # count of successors generated by that operator.
    _A: TypeAlias = Tuple[_PG3SearchOperator, int]

    # Create a trajectory generator.
    trajectory_generator = _create_trajectory_generator(
        heuristic_name, predicates, operators, task_planning_heuristic,
        max_policy_guided_rollout, user_supplied_demos)

    # Create the PG3 search operators.
    search_operators = _create_search_operators(predicates, operators,
                                                allow_new_vars)

    # The heuristic is what distinguishes PG3 from baseline approaches.
    heuristic = _create_heuristic(heuristic_name, trajectory_generator,
                                  train_tasks, horizon)

    # Initialize the search with an empty list.
    if initial_policy_strs is None:
        initial_states = [LiftedDecisionList([])]
    else:
        initial_states = [
            utils.parse_ldl_from_str(l, types, predicates, operators)
            for l in initial_policy_strs
        ]

    def get_successors(ldl: _S) -> Iterator[Tuple[_A, _S, float]]:
        for op in search_operators:
            for i, child in enumerate(op.get_successors(ldl)):
                if any(
                        len(rule.parameters) > max_rule_params
                        for rule in child.rules):
                    continue
                yield (op, i), child, 1.0  # cost always 1

    if search_method == "gbfs":
        # Terminate only after max expansions.
        path, _ = run_gbfs(initial_states=initial_states,
                           check_goal=lambda _: False,
                           get_successors=get_successors,
                           heuristic=heuristic,
                           max_expansions=gbfs_max_expansions,
                           lazy_expansion=True)

    elif search_method == "hill_climbing":
        # Terminate when no improvement is found.
        path, _, _ = run_hill_climbing(initial_states=initial_states,
                                       check_goal=lambda _: False,
                                       get_successors=get_successors,
                                       heuristic=heuristic,
                                       early_termination_heuristic_thresh=0,
                                       enforced_depth=hc_enforced_depth)

    else:
        raise NotImplementedError("Unrecognized search_method "
                                  f"{search_method}.")

    # Return the best seen policy.
    best_ldl = path[-1]
    return best_ldl


def _create_search_operators(
        predicates: Set[Predicate],
        operators: Set[STRIPSOperator],
        allow_new_vars: bool = True) -> List[_PG3SearchOperator]:
    search_operator_classes = [
        _AddRulePG3SearchOperator,
        _AddConditionPG3SearchOperator,
    ]
    return [
        cls(predicates, operators, allow_new_vars)
        for cls in search_operator_classes
    ]


def _create_heuristic(heuristic_name: str,
                      trajectory_gen: _TrajectoryGenerator,
                      train_tasks: Sequence[Task],
                      horizon: int) -> _PG3Heuristic:
    heuristic_name_to_cls: Dict[str, TypingType[_PG3Heuristic]] = {
        "policy_evaluation": _PolicyEvaluationPG3Heuristic,
        # Demo plan comparison and policy guided differ in the trajectory gen.
        "demo_plan_comparison": _PlanComparisonPG3Heuristic,
        "policy_guided": _PlanComparisonPG3Heuristic,
    }
    cls = heuristic_name_to_cls[heuristic_name]
    return cls(trajectory_gen, train_tasks, horizon)


def _create_trajectory_generator(
    heuristic_name: str,
    predicates: Set[Predicate],
    operators: Set[STRIPSOperator],
    task_planning_heuristic: str = "lmcut",
    max_policy_guided_rollout: int = 50,
    user_supplied_demos: Optional[Dict[Task, List[str]]] = None
) -> _TrajectoryGenerator:
    if heuristic_name == "policy_guided":
        traj_gen_cls: TypingType[
            _TrajectoryGenerator] = _PolicyGuidedPlanningTrajectoryGenerator
    elif user_supplied_demos is not None:
        traj_gen_cls = _UserSuppliedDemoTrajectoryGenerator
    else:
        traj_gen_cls = _StaticPlanningTrajectoryGenerator
    return traj_gen_cls(predicates, operators, task_planning_heuristic,
                        max_policy_guided_rollout, user_supplied_demos)
