"""Interactive-node guardrails.

This pipeline does not submit batch jobs. If a wrapper asks for job submission,
only interactive allocation modes are accepted.
"""

from __future__ import annotations


INTERACTIVE_MODES = {"local", "interactive", "salloc", "srun-interactive"}
FORBIDDEN_MODES = {"batch", "sbatch", "qsub", "noninteractive", "queue"}


def require_interactive_mode(mode: str | None) -> str:
    selected = (mode or "local").strip().lower()
    if selected in FORBIDDEN_MODES:
        raise RuntimeError(
            f"Refusing run_mode={mode!r}: RAE-2 Ryle-Vonberg jobs must run on interactive nodes only."
        )
    if selected not in INTERACTIVE_MODES:
        raise RuntimeError(
            f"Unknown run_mode={mode!r}. Allowed modes are: {', '.join(sorted(INTERACTIVE_MODES))}."
        )
    return selected

