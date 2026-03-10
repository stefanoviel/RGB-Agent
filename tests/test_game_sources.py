from arcgym.evaluation.game_sources import LOCAL_SOURCE, REMOTE_SOURCE, resolve_game_specs


EVALUATION_GAMES = {
    "ls20": ["ls20-cb3b57cc"],
    "vc33": ["vc33-9851e02b"],
    "all": ["ls20-cb3b57cc", "vc33-9851e02b"],
}


def test_resolve_remote_exact_without_local() -> None:
    specs, local_info = resolve_game_specs(
        ["ls20"],
        evaluation_games=EVALUATION_GAMES,
        allow_local=False,
    )

    assert local_info is None
    assert specs == [type(specs[0])(source=REMOTE_SOURCE, game_id="ls20-cb3b57cc", requested="ls20")]


def test_resolve_prefixed_local_with_stubbed_catalog(monkeypatch) -> None:
    from arcgym.evaluation import game_sources

    stub = game_sources.LocalSourceInfo(
        game_ids=("ls20-0001", "snake-0001"),
        environments_dir="/tmp/re_arc/environment_files",
    )
    game_sources.load_local_source_info.cache_clear()
    monkeypatch.setattr(game_sources, "load_local_source_info", lambda: stub)

    specs, local_info = resolve_game_specs(
        ["rearc:snake", "arc:vc33"],
        evaluation_games=EVALUATION_GAMES,
    )

    assert local_info == stub
    assert [spec.source for spec in specs] == [LOCAL_SOURCE, REMOTE_SOURCE]
    assert [spec.game_id for spec in specs] == ["snake-0001", "vc33-9851e02b"]


def test_resolve_ambiguous_short_name_requires_prefix(monkeypatch) -> None:
    from arcgym.evaluation import game_sources

    stub = game_sources.LocalSourceInfo(
        game_ids=("ls20-0001",),
        environments_dir="/tmp/re_arc/environment_files",
    )
    game_sources.load_local_source_info.cache_clear()
    monkeypatch.setattr(game_sources, "load_local_source_info", lambda: stub)

    try:
        resolve_game_specs(["ls20"], evaluation_games=EVALUATION_GAMES)
    except ValueError as exc:
        assert "Ambiguous game 'ls20'" in str(exc)
    else:
        raise AssertionError("expected ambiguity error")
