"""Search algorithms."""
from __future__ import annotations

import heapq as hq
import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Hashable, Iterator, List, \
    Optional, Sequence, Tuple, TypeVar, cast

import pathos.multiprocessing as mp

_S = TypeVar("_S", bound=Hashable)  # state in heuristic search
_A = TypeVar("_A")  # action in heuristic search


@dataclass(frozen=True)
class _HeuristicSearchNode(Generic[_S, _A]):
    state: _S
    edge_cost: float
    cumulative_cost: float
    parent: Optional[_HeuristicSearchNode[_S, _A]] = None
    action: Optional[_A] = None


def _run_heuristic_search(
        initial_states: Sequence[_S],
        check_goal: Callable[[_S], bool],
        get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
        get_priority: Callable[[_HeuristicSearchNode[_S, _A]], Any],
        max_expansions: int = 10000000,
        max_evals: int = 10000000,
        timeout: int = 10000000,
        lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """A generic heuristic search implementation.

    Depending on get_priority, can implement A*, GBFS, or UCS. If no
    goal is found, returns the state with the best priority.
    """
    start_time = time.perf_counter()

    queue: List[Tuple[Any, int, _HeuristicSearchNode[_S, _A]]] = []
    state_to_best_path_cost: Dict[_S, float] = \
        defaultdict(lambda : float("inf"))

    best_node: Optional[_HeuristicSearchNode[_S, _A]] = None
    best_node_priority = float("inf")
    tiebreak = itertools.count()
    num_expansions = 0
    num_evals = 0
    for initial_state in initial_states:
        root_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
            initial_state, 0, 0)
        root_priority = get_priority(root_node)
        if best_node is None or root_priority < best_node_priority:
            best_node = root_node
            best_node_priority = root_priority
        hq.heappush(queue, (root_priority, next(tiebreak), root_node))
        num_evals += 1
    assert best_node is not None

    while len(queue) > 0 and time.perf_counter() - start_time < timeout and \
          num_expansions < max_expansions and num_evals < max_evals:
        _, _, node = hq.heappop(queue)
        # If we already found a better path here, don't bother.
        if state_to_best_path_cost[node.state] < node.cumulative_cost:
            continue
        # If the goal holds, return.
        if check_goal(node.state):
            return _finish_plan(node)
        num_expansions += 1
        # Generate successors.
        for action, child_state, cost in get_successors(node.state):
            if time.perf_counter() - start_time >= timeout:
                break
            child_path_cost = node.cumulative_cost + cost
            # If we already found a better path to this child, don't bother.
            if state_to_best_path_cost[child_state] <= child_path_cost:
                continue
            # Add new node.
            child_node = _HeuristicSearchNode(state=child_state,
                                              edge_cost=cost,
                                              cumulative_cost=child_path_cost,
                                              parent=node,
                                              action=action)
            priority = get_priority(child_node)
            num_evals += 1
            hq.heappush(queue, (priority, next(tiebreak), child_node))
            state_to_best_path_cost[child_state] = child_path_cost
            if priority < best_node_priority:
                best_node_priority = priority
                best_node = child_node
                # Optimization: if we've found a better child, immediately
                # explore the child without expanding the rest of the children.
                # Accomplish this by putting the parent node back on the queue.
                if lazy_expansion:
                    hq.heappush(queue, (priority, next(tiebreak), node))
                    break
            if num_evals >= max_evals:
                break

    # Did not find path to goal; return best path seen.
    return _finish_plan(best_node)


def _finish_plan(
        node: _HeuristicSearchNode[_S, _A]) -> Tuple[List[_S], List[_A]]:
    """Helper for _run_heuristic_search and run_hill_climbing."""
    rev_state_sequence: List[_S] = []
    rev_action_sequence: List[_A] = []

    while node.parent is not None:
        action = cast(_A, node.action)
        rev_action_sequence.append(action)
        rev_state_sequence.append(node.state)
        node = node.parent
    rev_state_sequence.append(node.state)

    return rev_state_sequence[::-1], rev_action_sequence[::-1]


def run_gbfs(initial_states: Sequence[_S],
             check_goal: Callable[[_S], bool],
             get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
             heuristic: Callable[[_S], float],
             max_expansions: int = 10000000,
             max_evals: int = 10000000,
             timeout: int = 10000000,
             lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """Greedy best-first search."""
    get_priority = lambda n: heuristic(n.state)
    return _run_heuristic_search(initial_states, check_goal, get_successors,
                                 get_priority, max_expansions, max_evals,
                                 timeout, lazy_expansion)


def run_astar(initial_states: Sequence[_S],
              check_goal: Callable[[_S], bool],
              get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
              heuristic: Callable[[_S], float],
              max_expansions: int = 10000000,
              max_evals: int = 10000000,
              timeout: int = 10000000,
              lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """A* search."""
    get_priority = lambda n: heuristic(n.state) + n.cumulative_cost
    return _run_heuristic_search(initial_states, check_goal, get_successors,
                                 get_priority, max_expansions, max_evals,
                                 timeout, lazy_expansion)


def run_hill_climbing(
        initial_states: Sequence[_S],
        check_goal: Callable[[_S], bool],
        get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
        heuristic: Callable[[_S], float],
        early_termination_heuristic_thresh: Optional[float] = None,
        enforced_depth: int = 0,
        parallelize: bool = False) -> Tuple[List[_S], List[_A], List[float]]:
    """Enforced hill climbing local search.

    For each node, the best child node is always selected, if that child
    is an improvement over the node. If no children improve on the node,
    look at the children's children, etc., up to enforced_depth, where
    enforced_depth 0 corresponds to simple hill climbing. Terminate when
    no improvement can be found. early_termination_heuristic_thresh
    allows for searching until heuristic reaches a specified value.
    Lower heuristic is better.
    """
    assert enforced_depth >= 0
    # Start with the best initial state.
    cur_node: Optional[_HeuristicSearchNode[_S, _A]] = None
    visited = set()
    best_heuristic = float("inf")
    for initial_state in initial_states:
        root_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
            initial_state, 0, 0)
        root_heuristic = heuristic(initial_state)
        if cur_node is None or root_heuristic < best_heuristic:
            cur_node = root_node
            best_heuristic = root_heuristic
        visited.add(initial_state)
    last_heuristic = best_heuristic
    heuristics = [last_heuristic]
    assert cur_node is not None
    logging.info(f"\n\nStarting hill climbing at state {cur_node.state} "
                 f"with heuristic {last_heuristic}")
    while True:

        # Stops when heuristic reaches specified value.
        if early_termination_heuristic_thresh is not None \
            and last_heuristic <= early_termination_heuristic_thresh:
            break

        if check_goal(cur_node.state):
            logging.info("\nTerminating hill climbing, achieved goal")
            break
        best_heuristic = float("inf")
        best_child_node = None
        current_depth_nodes = [cur_node]
        all_best_heuristics = []
        early_break = False
        for depth in range(0, enforced_depth + 1):
            logging.info(f"Searching for an improvement at depth {depth}")
            # This is a list to ensure determinism. Note that duplicates are
            # filtered out in the `child_state in visited` check.
            successors_at_depth = []
            for parent in current_depth_nodes:
                for action, child_state, cost in get_successors(parent.state):
                    if child_state in visited:
                        continue
                    visited.add(child_state)
                    child_path_cost = parent.cumulative_cost + cost
                    child_node = _HeuristicSearchNode(
                        state=child_state,
                        edge_cost=cost,
                        cumulative_cost=child_path_cost,
                        parent=parent,
                        action=action)
                    successors_at_depth.append(child_node)
                    if parallelize:
                        continue  # heuristic computation is parallelized later
                    child_heuristic = heuristic(child_node.state)
                    if child_heuristic < best_heuristic:
                        best_heuristic = child_heuristic
                        best_child_node = child_node
                    print(child_node, "SCORE", child_heuristic)
                    if early_termination_heuristic_thresh is not None \
                        and best_heuristic <= early_termination_heuristic_thresh:
                        early_break = True
                        break

            if parallelize:
                # Parallelize the expensive part (heuristic computation).
                num_cpus = mp.cpu_count()
                fn = lambda n: (heuristic(n.state), n)
                with mp.Pool(processes=num_cpus) as p:
                    for child_heuristic, child_node in p.map(
                            fn, successors_at_depth):
                        if child_heuristic < best_heuristic:
                            best_heuristic = child_heuristic
                            best_child_node = child_node
            all_best_heuristics.append(best_heuristic)
            if last_heuristic > best_heuristic:
                # Some improvement found.
                logging.info(f"Found an improvement at depth {depth}")
                break
            # Continue on to the next depth.
            current_depth_nodes = successors_at_depth
            logging.info(f"No improvement found at depth {depth}")
        if best_child_node is None:
            logging.info("\nTerminating hill climbing, no more successors")
            break
        if last_heuristic <= best_heuristic:
            logging.info(
                "\nTerminating hill climbing, could not improve score")
            break
        heuristics.extend(all_best_heuristics)
        cur_node = best_child_node
        last_heuristic = best_heuristic
        logging.info(f"\nHill climbing reached new state {cur_node.state} "
                     f"with heuristic {last_heuristic}")
        if early_break:
            break
    states, actions = _finish_plan(cur_node)
    assert len(states) == len(heuristics)
    return states, actions, heuristics


def run_policy_guided_astar(
        initial_states: Sequence[_S],
        check_goal: Callable[[_S], bool],
        get_valid_actions: Callable[[_S], Iterator[Tuple[_A, float]]],
        get_next_state: Callable[[_S, _A], _S],
        heuristic: Callable[[_S], float],
        policy: Callable[[_S], Optional[_A]],
        num_rollout_steps: int,
        rollout_step_cost: float,
        max_expansions: int = 10000000,
        max_evals: int = 10000000,
        timeout: int = 10000000,
        lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """Perform A* search, but at each node, roll out a given policy for a given
    number of timesteps, creating new successors at each step.

    Stop the rollout prematurely if the policy returns None. Note that
    unlike the other search functions, which take get_successors as
    input, this function takes get_valid_actions and get_next_state as
    two separate inputs. This is necessary because we need to anticipate
    the next state conditioned on the action output by the policy. The
    get_valid_actions generates (action, cost) tuples. For policy-
    generated transitions, the costs are ignored, and rollout_step_cost
    is used instead.
    """

    # Create a new successor function that rolls out the policy first.
    # A successor here means: from this state, if you take this sequence of
    # actions in order, you'll end up at this final state.
    def get_successors(state: _S) -> Iterator[Tuple[List[_A], _S, float]]:
        # Get policy-based successors.
        policy_state = state
        policy_action_seq = []
        policy_cost = 0.0
        for _ in range(num_rollout_steps):
            action = policy(policy_state)
            valid_actions = {a for a, _ in get_valid_actions(policy_state)}
            if action is None or action not in valid_actions:
                break
            policy_state = get_next_state(policy_state, action)
            policy_action_seq.append(action)
            policy_cost += rollout_step_cost
            yield (list(policy_action_seq), policy_state, policy_cost)

        # Get primitive successors.
        for action, cost in get_valid_actions(state):
            next_state = get_next_state(state, action)
            yield ([action], next_state, cost)

    jumpy_states, action_subseqs = run_astar(initial_states=initial_states,
                                             check_goal=check_goal,
                                             get_successors=get_successors,
                                             heuristic=heuristic,
                                             max_expansions=max_expansions,
                                             max_evals=max_evals,
                                             timeout=timeout,
                                             lazy_expansion=lazy_expansion)

    # The states are "jumpy", so we need to reconstruct the dense state
    # sequence from the action subsequences. We also need to construct a
    # flat action sequence.
    state = jumpy_states[0]
    state_seq = [state]
    action_seq = []
    for action_subseq in action_subseqs:
        for action in action_subseq:
            action_seq.append(action)
            state = get_next_state(state, action)
            state_seq.append(state)

    return state_seq, action_seq
