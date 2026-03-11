from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REMOTE_SOURCE = "arc"
LOCAL_SOURCE = "rearc"
LOCAL_PREFIXES = ("rearc:", "re-arc:", "local:")
REMOTE_PREFIXES = ("arc:", "api:", "remote:")


@dataclass(frozen=True)
class GameSpec:
    source: str
    game_id: str
    requested: str


@dataclass(frozen=True)
class LocalSourceInfo:
    game_ids: tuple[str, ...]
    environments_dir: str


def _split_source_prefix(value: str) -> tuple[str | None, str]:
    normalized = str(value).strip().lower()
    for prefix in LOCAL_PREFIXES:
        if normalized.startswith(prefix):
            return LOCAL_SOURCE, normalized[len(prefix):]
    for prefix in REMOTE_PREFIXES:
        if normalized.startswith(prefix):
            return REMOTE_SOURCE, normalized[len(prefix):]
    return None, normalized


def _prefix_index(game_ids: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for game_id in game_ids:
        out.setdefault(game_id.split("-", 1)[0], game_id)
    return out


@lru_cache(maxsize=1)
def load_local_source_info() -> LocalSourceInfo:
    try:
        from re_arc import list_game_ids  # type: ignore[import-not-found]
        import re_arc  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Local RE-ARC games require the optional re-arc package. "
            "Install with `pip install -e \".[rearc]\"` or "
            "`pip install git+https://github.com/Tufalabs/re-arc-3.git@v0.1.1`."
        ) from exc

    environments_dir = Path(re_arc.__file__).resolve().parent / "environment_files"
    game_ids = tuple(sorted(str(game_id).strip().lower() for game_id in list_game_ids()))
    return LocalSourceInfo(game_ids=game_ids, environments_dir=str(environments_dir))


def build_remote_catalog(evaluation_games: dict[str, list[str]]) -> tuple[tuple[str, ...], dict[str, str]]:
    remote_ids = tuple(sorted({game_id.strip().lower() for values in evaluation_games.values() for game_id in values}))
    return remote_ids, _prefix_index(remote_ids)


def build_local_catalog(local_info: LocalSourceInfo) -> dict[str, str]:
    return _prefix_index(local_info.game_ids)


def resolve_game_specs(
    raw_games: list[str],
    *,
    evaluation_games: dict[str, list[str]],
    allow_local: bool = True,
) -> tuple[list[GameSpec], LocalSourceInfo | None]:
    remote_ids, remote_prefix_map = build_remote_catalog(evaluation_games)
    remote_set = set(remote_ids)

    local_info: LocalSourceInfo | None = load_local_source_info() if allow_local else None
    local_set = set(local_info.game_ids) if local_info else set()
    local_prefix_map = build_local_catalog(local_info) if local_info else {}

    specs: list[GameSpec] = []
    for raw_game in raw_games:
        source_hint, token = _split_source_prefix(raw_game)
        if not token:
            continue

        remote_match = token if token in remote_set else remote_prefix_map.get(token)
        local_match = token if token in local_set else local_prefix_map.get(token)

        if source_hint == REMOTE_SOURCE:
            if not remote_match:
                raise ValueError(f"Unknown ARC API game: {raw_game}")
            specs.append(GameSpec(source=REMOTE_SOURCE, game_id=remote_match, requested=raw_game))
            continue

        if source_hint == LOCAL_SOURCE:
            if local_info is None:
                raise ValueError(f"Local game source requested but re-arc is unavailable: {raw_game}")
            if not local_match:
                raise ValueError(f"Unknown RE-ARC game: {raw_game}")
            specs.append(GameSpec(source=LOCAL_SOURCE, game_id=local_match, requested=raw_game))
            continue

        if remote_match and local_match and remote_match != local_match:
            raise ValueError(
                f"Ambiguous game '{raw_game}'. Use 'arc:{token}' for {remote_match} "
                f"or 'rearc:{token}' for {local_match}."
            )
        if remote_match:
            specs.append(GameSpec(source=REMOTE_SOURCE, game_id=remote_match, requested=raw_game))
            continue
        if local_match:
            specs.append(GameSpec(source=LOCAL_SOURCE, game_id=local_match, requested=raw_game))
            continue
        raise ValueError(f"Unknown game: {raw_game}")

    return specs, local_info


def build_local_env(game_spec: GameSpec, local_info: LocalSourceInfo, *, max_actions: int, seed: int = 0):
    from arcgym.environments import ArcAgi3Env
    from arc_agi import OperationMode

    return ArcAgi3Env(
        game_id=game_spec.game_id,
        max_actions=max_actions,
        seed=seed,
        operation_mode=OperationMode.OFFLINE,
        environments_dir=local_info.environments_dir,
        manage_scorecard=False,
    )
