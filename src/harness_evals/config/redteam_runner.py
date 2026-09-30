"""Wire a RedTeamConfig to the red-team runner."""

from __future__ import annotations

import logging

from harness_evals._async_compat import _run_async
from harness_evals.config.redteam_schema import RedTeamConfig
from harness_evals.config.runner import build_llm, build_sink, build_target
from harness_evals.errors import HarnessEvalsError
from harness_evals.plugins import attack_plugin, load_plugins
from harness_evals.redteam.packs import pack as lookup_pack
from harness_evals.redteam.plugin import AttackPlugin
from harness_evals.redteam.report import RedTeamReport
from harness_evals.redteam.runner import run_redteam

logger = logging.getLogger(__name__)


def resolve_plugins(cfg: RedTeamConfig) -> list[AttackPlugin]:
    """Instantiate every plugin named by the config's packs and plugin list.

    Explicit ``plugins`` entries override pack entries with the same id, so a
    config can pull in a whole pack and still pass params to one member of it.
    Pack ids with no registered implementation are skipped with a warning
    rather than failing the run — a pack is a coverage goal, and a half-built
    pack should still run the half that exists.
    """
    by_id: dict[str, AttackPlugin] = {}

    for pack_id in cfg.packs:
        try:
            resolved_pack = lookup_pack(pack_id)
        except KeyError as err:
            raise HarnessEvalsError(str(err)) from None
        missing = resolved_pack.unimplemented()
        if missing:
            logger.warning(
                "Pack %r has %d plugin(s) with no implementation yet, skipping: %s",
                pack_id,
                len(missing),
                ", ".join(missing),
            )
        for plugin in resolved_pack.resolve_available():
            by_id[plugin.id] = plugin

    for spec in cfg.plugins:
        cls = attack_plugin(spec.id)
        try:
            by_id[spec.id] = cls(**spec.params)
        except (TypeError, ValueError) as err:
            raise HarnessEvalsError(f"Invalid params for plugin {spec.id!r}: {err}") from None

    if not by_id:
        raise HarnessEvalsError(
            "No attack plugins resolved. Check 'redteam.packs' and 'redteam.plugins' "
            "against `harness-evals redteam list-plugins`."
        )

    return list(by_id.values())


def run_redteam_config(cfg: RedTeamConfig) -> RedTeamReport:
    """Synchronous entry point — load plugins, build the target, run the scan."""
    if cfg.plugin_modules:
        load_plugins(cfg.plugin_modules)

    plugins = resolve_plugins(cfg)
    judge_llm = build_llm(cfg.judge_llm) if cfg.judge_llm else None
    sinks = [build_sink(s) for s in cfg.sinks] if cfg.sinks else None

    async def _run() -> RedTeamReport:
        target = await build_target(cfg.target)
        async with target:
            return await run_redteam(
                cfg.name,
                plugins,
                target.ainvoke,
                capabilities=cfg.capabilities,
                preconditions=cfg.preconditions,
                judge_llm=judge_llm,
                sinks=sinks,
                concurrency=cfg.concurrency,
            )

    return _run_async(_run())
