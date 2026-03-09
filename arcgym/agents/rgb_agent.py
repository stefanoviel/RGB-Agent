"""
RGBAgent — orchestrator that owns the game loop.

Absorbs the logic previously in ``arcgym.evaluation.harness.evaluate_single_game``.
The Swarm creates one RGBAgent per game thread and calls ``agent.run()``.
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

import requests

from arcgym.agents.claude_code_action_agent import ClaudeCodeActionAgent, QueueExhausted
from arcgym.environments import ArcAgi3Env
from arcengine import GameState
from arcgym.metrics.structures import GameMetrics, LevelMetrics, AttemptMetrics, Status

log = logging.getLogger(__name__)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # seconds


def _run_with_retries(func_to_run: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run *func_to_run* with exponential backoff on network errors."""
    retries = 0
    backoff = INITIAL_BACKOFF
    while True:
        try:
            return func_to_run(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retries >= MAX_RETRIES:
                log.error(f"Final attempt failed for {func_to_run.__name__} after {retries} retries.")
                raise
            log.warning(
                f"API error for {func_to_run.__name__}: {type(e).__name__}. "
                f"Retrying in {backoff}s (attempt {retries + 1}/{MAX_RETRIES})"
            )
            time.sleep(backoff)
            retries += 1
            backoff *= 2


def _render_post_action_grid(agent) -> str | None:
    """Render the current board from the agent's last observation."""
    _, grid_text = agent._process_frame(agent._last_observation or {})
    return grid_text or None


# ---------------------------------------------------------------------------
# RGBAgent
# ---------------------------------------------------------------------------

class RGBAgent:
    """Orchestrator that drives the inner ClaudeCodeActionAgent through a game loop.

    Not a BaseAgent subclass — this is a high-level controller that owns the
    env interaction loop, manages the analyzer, and collects metrics.
    """

    def __init__(
        self,
        *,
        env: ArcAgi3Env,
        game_id: str,
        agent_name: str,
        max_actions_per_game: int,
        run_index: int = 1,
        tags: Optional[List[str]] = None,
        prompts_log_path: Optional[Path] = None,
        analyzer=None,
        log_post_board: bool = False,
        analyzer_retries: int = 5,
        inner_agent_kwargs: Optional[dict] = None,
    ) -> None:
        self.env = env
        self.game_id = game_id
        self.agent_name = agent_name
        self.max_actions_per_game = max_actions_per_game
        self.run_index = run_index
        self.tags = tags
        self.prompts_log_path = prompts_log_path
        self.analyzer = analyzer
        self.log_post_board = log_post_board
        self.analyzer_retries = analyzer_retries

        self._agent = ClaudeCodeActionAgent(**(inner_agent_kwargs or {}))

    # -- public API ----------------------------------------------------------

    def run(self) -> GameMetrics:
        """Execute the full game loop, returning collected metrics."""

        run_metrics = GameMetrics(
            game_id=self.game_id,
            agent_name=self.agent_name,
            run_index=self.run_index,
            start_time=time.time(),
        )
        run_metrics.status = Status.IN_PROGRESS

        self._current_level_number = 1
        self._current_level_metrics = LevelMetrics(level_number=self._current_level_number)
        self._current_attempt_number = 1
        self._current_attempt_metrics = AttemptMetrics(attempt_number=self._current_attempt_number)
        self._attempt_start_time = run_metrics.start_time

        self._max_score = 0
        self._total_actions = 0
        self._arc_state: GameState | None = None
        self._arc_score = 0

        _loop = asyncio.new_event_loop()

        try:
            self._agent.reset()

            self._arc_state, self._arc_score = self._reset_game_state(run_metrics)

            # Log the initial board state
            if self.prompts_log_path:
                _init_grid = _render_post_action_grid(self._agent)
                if _init_grid:
                    with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Action 0 | Level {self._current_level_number} | Attempt {self._current_attempt_number} | INITIAL STATE\n")
                        f.write(f"Score: {self._arc_score} | State: {self._arc_state.name}\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(f"[INITIAL BOARD STATE]\n{_init_grid}\n\n")

            # --- Main game loop ---
            while self._total_actions < self.max_actions_per_game:
                try:
                    action_dict = _loop.run_until_complete(self._agent.call_llm())
                except QueueExhausted:
                    log.info(f"[RGBAgent] queue exhausted at action {self._total_actions} — firing analyzer")
                    _RETRY_NUDGE = (
                        "CRITICAL: Your previous response was missing the [ACTIONS] section. "
                        "You MUST end your response with an [ACTIONS] section containing a JSON action plan. "
                        "Do NOT write actions to a file — output them directly in your response text."
                    )
                    _analyzer_loaded = False
                    for _attempt in range(self.analyzer_retries):
                        _nudge = _RETRY_NUDGE if _attempt > 0 else ""
                        # On resumed sessions, always nudge since model tends to skip [ACTIONS]
                        if not _nudge and self._total_actions > 0:
                            _nudge = _RETRY_NUDGE
                        log.info(f"[RGBAgent] analyzer attempt {_attempt + 1}/{self.analyzer_retries} "
                                 f"action={self._total_actions} nudge={bool(_nudge)}")
                        if self._fire_analyzer_and_load_plan(self._total_actions, retry_nudge=_nudge):
                            _analyzer_loaded = True
                            break
                        log.warning(f"[RGBAgent] analyzer attempt {_attempt + 1}/{self.analyzer_retries} failed — retrying")
                    if not _analyzer_loaded:
                        raise
                    action_dict = _loop.run_until_complete(self._agent.call_llm())

                action_obj = self._agent.update_from_model()
                observation, reward, done = _run_with_retries(self.env.step, action_obj)

                self._total_actions += 1
                self._current_attempt_metrics.actions += 1

                previous_arc_score = self._arc_score
                self._arc_state = GameState[observation.get("state") or "NOT_PLAYED"]
                self._arc_score = observation.get("score", 0) or 0
                self._max_score = max(self._max_score, self._arc_score)
                run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, self._current_level_number)

                self._agent.update_from_env(observation=observation, reward=reward, done=done)

                # Log prompts/responses
                if self.prompts_log_path and self._agent.trajectory.steps:
                    last_step = self._agent.trajectory.steps[-1]
                    with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*80}\n")
                        plan_step_info = ""
                        if self._agent._plan_total > 0:
                            plan_step_info = f" | Plan Step {self._agent._plan_index}/{self._agent._plan_total}"
                        f.write(f"Action {self._total_actions} | Level {self._current_level_number} | Attempt {self._current_attempt_number}{plan_step_info}\n")
                        f.write(f"Score: {self._arc_score} | State: {self._arc_state.name}\n")
                        f.write(f"{'='*80}\n\n")

                        if last_step.chat_completions:
                            for msg in last_step.chat_completions:
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                tool_calls = msg.get('tool_calls', [])
                                f.write(f"[{role.upper()}]\n")
                                if content:
                                    f.write(f"{content}\n")
                                if tool_calls:
                                    for tc in tool_calls:
                                        fn = tc.get('function', {}) if isinstance(tc, dict) else {}
                                        f.write(f"Tool: {fn.get('name', tc)}({fn.get('arguments', '')})\n")
                                f.write("\n")

                # Log post-action board state
                if self.log_post_board and self.prompts_log_path:
                    _post_grid = _render_post_action_grid(self._agent)
                    if _post_grid:
                        with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"[POST-ACTION BOARD STATE]\nScore: {self._arc_score}\n{_post_grid}\n\n")

                # --- Handle Level Completion ---
                level_completed = (self._arc_score > previous_arc_score and
                                   self._arc_state not in (GameState.WIN, GameState.GAME_OVER))

                if level_completed:
                    attempt_end_time = self._record_attempt(Status.COMPLETED)
                    self._current_level_metrics.status = Status.COMPLETED
                    run_metrics.level_metrics[self._current_level_number] = self._current_level_metrics

                    log.info(
                        f"[{self.game_id} Run {self.run_index}] Level {self._current_level_number} COMPLETED. "
                        f"Attempt {self._current_attempt_number} actions: {self._current_attempt_metrics.actions}. Score: {self._arc_score}."
                    )

                    self._current_level_number += 1
                    run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, self._current_level_number)
                    self._current_level_metrics = LevelMetrics(level_number=self._current_level_number)
                    self._current_attempt_number = 1
                    self._current_attempt_metrics = AttemptMetrics(attempt_number=self._current_attempt_number)
                    self._attempt_start_time = attempt_end_time
                    continue

                if self._arc_state == GameState.GAME_OVER:
                    self._record_attempt(Status.GAME_OVER, game_over=True)
                    self._current_level_metrics.status = Status.GAME_OVER
                    run_metrics.level_metrics[self._current_level_number] = self._current_level_metrics
                    run_metrics.status = Status.TIMEOUT
                    log.warning(
                        f"[{self.game_id} Run {self.run_index}] Game Over on Level {self._current_level_number}, "
                        f"Attempt {self._current_attempt_number}. Actions: {self._current_attempt_metrics.actions}."
                    )
                    self._current_attempt_number += 1
                    self._current_attempt_metrics = AttemptMetrics(attempt_number=self._current_attempt_number)
                    self._attempt_start_time = time.time()

                if self._arc_state == GameState.WIN:
                    self._record_attempt(Status.COMPLETED)
                    self._current_level_metrics.status = Status.COMPLETED
                    run_metrics.level_metrics[self._current_level_number] = self._current_level_metrics
                    run_metrics.status = Status.COMPLETED_RUN
                    log.info(
                        f"[{self.game_id} Run {self.run_index}] Game COMPLETED! "
                        f"Level {self._current_level_number} actions: {self._current_attempt_metrics.actions}. Score: {self._arc_score}"
                    )
                    break

        except QueueExhausted as e:
            log.info(f"[{self.game_id} Run {self.run_index}] Episode ended (queue exhausted): {e}")
            run_metrics.status = Status.QUEUE_EXHAUSTED

        except Exception as e:
            run_metrics.status = Status.ERROR
            run_metrics.error_message = str(e)
            self._current_attempt_metrics.status = Status.ERROR
            self._current_level_metrics.status = Status.ERROR
            log.error(f"[{self.game_id} Run {self.run_index}] Exception: {e}", exc_info=True)

        finally:
            run_metrics.end_time = time.time()
            run_metrics.run_duration_seconds = run_metrics.end_time - run_metrics.start_time

            # Finalize attempt status if still in progress
            if self._current_attempt_metrics.status == Status.IN_PROGRESS:
                self._current_attempt_metrics.duration_seconds = run_metrics.end_time - self._attempt_start_time
                if run_metrics.status == Status.ERROR:
                    self._current_attempt_metrics.status = Status.ERROR
                elif self._arc_state == GameState.WIN:
                    self._current_attempt_metrics.status = Status.COMPLETED
                    run_metrics.status = Status.COMPLETED_RUN
                else:
                    self._current_attempt_metrics.status = Status.TIMEOUT
                    if run_metrics.status == Status.IN_PROGRESS:
                        run_metrics.status = Status.TIMEOUT

            if not self._current_level_metrics.attempts or self._current_level_metrics.attempts[-1].attempt_number != self._current_attempt_metrics.attempt_number:
                self._current_level_metrics.attempts.append(self._current_attempt_metrics)
            if self._current_level_metrics.status == Status.IN_PROGRESS:
                self._current_level_metrics.status = self._current_attempt_metrics.status

            run_metrics.level_metrics[self._current_level_number] = self._current_level_metrics
            run_metrics.run_total_actions = sum(lm.total_actions for lm in run_metrics.level_metrics.values())
            run_metrics.total_game_overs_across_run = sum(lm.total_game_overs for lm in run_metrics.level_metrics.values())
            run_metrics.total_state_changes_across_run = sum(lm.total_state_changes for lm in run_metrics.level_metrics.values())
            run_metrics.final_score = self._max_score

            if run_metrics.guid and not run_metrics.replay_url:
                run_metrics.replay_url = f"{ROOT_URL}/replay/{self.game_id}/{run_metrics.guid}"

            _loop.close()

        return run_metrics

    # -- private helpers -----------------------------------------------------

    def _reset_game_state(self, run_metrics: GameMetrics) -> tuple[GameState, int]:
        """Reset env, extract GUID, feed initial observation to inner agent."""
        observation = _run_with_retries(
            self.env.reset,
            task={"game_id": self.game_id, "max_actions": self.max_actions_per_game, "tags": self.tags},
        )
        arc_state = GameState[observation.get("state") or "NOT_PLAYED"]
        arc_score = observation.get("score", 0) or 0

        obs_guid = observation.get("guid")
        if obs_guid and not run_metrics.guid:
            run_metrics.guid = obs_guid
            run_metrics.replay_url = f"{ROOT_URL}/replay/{self.game_id}/{obs_guid}"
            log.info(f"[{self.game_id} Run {self.run_index}] Replay URL: {run_metrics.replay_url}")
            if self.prompts_log_path:
                guid_path = self.prompts_log_path.parent / "run_info.txt"
                guid_path.write_text(
                    f"game_id: {self.game_id}\n"
                    f"guid: {obs_guid}\n"
                    f"replay_url: {run_metrics.replay_url}\n"
                    f"scorecard_id: {getattr(self.env, '_scorecard_id', 'unknown')}\n"
                    f"command: {Path(sys.argv[0]).name} {' '.join(sys.argv[1:])}\n"
                )

        self._agent.update_from_env(observation=observation, reward=0.0, done=False)
        return arc_state, arc_score

    def _fire_analyzer_and_load_plan(self, action_num: int, retry_nudge: str = "") -> bool:
        """Fire the analyzer, parse hint, load action plan. Returns True if plan loaded."""
        if not self.analyzer:
            return False
        # Append current board state so the analyzer sees the latest
        if self.prompts_log_path and not self.log_post_board:
            _post_grid = _render_post_action_grid(self._agent)
            if _post_grid:
                with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[POST-ACTION BOARD STATE]\nScore: {self._arc_score}\n{_post_grid}\n\n")
        hint = self.analyzer(self.prompts_log_path, action_num, retry_nudge=retry_nudge)
        if not hint:
            log.warning(f"[RGBAgent] analyzer returned None at action {action_num}")
            return False
        # Normalize trailing whitespace
        hint = "\n".join(line.rstrip() for line in hint.split("\n"))
        _actions_text = None
        _actions_sep = "\n[ACTIONS]\n"
        if _actions_sep in hint:
            hint, _actions_text = hint.split(_actions_sep, 1)
            _actions_text = _actions_text.strip()
        _plan_sep = "\n[PLAN]\n"
        if _plan_sep in hint:
            full_hint, plan = hint.split(_plan_sep, 1)
            full_hint = full_hint.strip()
            plan = plan.strip()
        else:
            full_hint = hint
            plan = hint
        self._agent.set_external_hint(full_hint)
        self._agent.set_persistent_hint(plan)
        if _actions_text:
            loaded = self._agent.set_action_plan(_actions_text)
            if loaded:
                log.info(f"[RGBAgent] analyzer at action {action_num}: loaded action plan ({len(_actions_text)} chars)")
                return True
            log.warning(f"[RGBAgent] analyzer at action {action_num}: set_action_plan rejected the plan")
            return False
        log.warning(f"[RGBAgent] analyzer at action {action_num}: hint received but NO [ACTIONS] section")
        return False

    def _record_attempt(self, status: str, game_over: bool = False) -> float:
        """Finalize current attempt metrics. Returns the end time."""
        attempt_end_time = time.time()
        self._current_attempt_metrics.duration_seconds = attempt_end_time - self._attempt_start_time
        self._current_attempt_metrics.status = status
        if game_over:
            self._current_attempt_metrics.game_overs += 1
        self._current_level_metrics.attempts.append(self._current_attempt_metrics)
        return attempt_end_time
