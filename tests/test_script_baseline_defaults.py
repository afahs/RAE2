"""Science scripts must not silently default to attenuating profile baselines.

``sideband_linear`` and ``linear_all`` absorb roughly three quarters of a
predicted step into the fitted slope (see
test_source_sensitivity_regressions). They remain available as explicit
sensitivity options, but no script may select them by default.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from rylevonberg.stacking import DEFAULT_BASELINE_MODE, detrend_profile_values

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ATTENUATING = ("sideband_linear", "linear_all")
DEFAULT_PATTERNS = [
    re.compile(r"baseline_mode\s*:\s*str\s*=\s*[\"'](?P<mode>\w+)[\"']"),
    re.compile(r"--baseline-mode[\"']\s*,\s*default\s*=\s*[\"'](?P<mode>\w+)[\"']"),
    re.compile(r"\bbaseline_mode\s*=\s*[\"'](?P<mode>\w+)[\"']"),
]


def test_no_script_defaults_to_an_attenuating_baseline() -> None:
    offenders = []
    for path in sorted(SCRIPTS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in DEFAULT_PATTERNS:
            for match in pattern.finditer(text):
                if match.group("mode") in ATTENUATING:
                    line = text.count("\n", 0, match.start()) + 1
                    offenders.append(f"{path.name}:{line} {match.group(0)}")
    assert not offenders, "attenuating baseline used as a default:\n" + "\n".join(offenders)


@pytest.mark.parametrize("mode", ATTENUATING)
def test_attenuating_modes_are_documented_as_attenuating(mode: str) -> None:
    t = np.arange(-600.0, 601.0, 30.0)
    y = np.where(t < 0, 0.5, -0.5)
    profile, template, _, _ = detrend_profile_values(t, y, "disappearance", baseline_mode=mode, normalize=False)
    amplitude = float(np.dot(profile, template) / np.dot(template, template))
    assert amplitude < 0.35  # legacy modes recover only about a quarter of the step
    profile, template, _, _ = detrend_profile_values(t, y, "disappearance", baseline_mode=DEFAULT_BASELINE_MODE, normalize=False)
    assert np.isclose(float(np.dot(profile, template) / np.dot(template, template)), 1.0)
