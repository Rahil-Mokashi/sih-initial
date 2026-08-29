"""
Vessel-identity anonymization for public-facing dashboard/map builds.

THOR FREYJA and ALAWAD1 (the real top suspects in ow-0001/ow-0002) are
real, identifiable ships. Naming them as "suspects" in a spill is a real
reputational exposure for a real company's real vessel if it leaves the
judging room via a deck, recording, or screenshot -- even though the
attribution methodology is explicit that this is a ranked lead, not a
verdict (see DECISIONS.md "Attribution scoring"). Decided with the user
2026-08-26: keep real identities in the live dashboard shown directly to
judges (proof the pipeline runs on real data), but anonymize ship_name/
mmsi/imo in anything built for a deck, recording, or screenshot via
build_dashboard.py/build_map.py --anonymize. Distance, timing, score,
flag, and vessel_type are left real -- they're evidence, not identity.
"""

from __future__ import annotations

import copy
import string


def anonymize_ranking(ranking: dict | None) -> dict | None:
    if ranking is None:
        return None
    anon = copy.deepcopy(ranking)
    for v in anon["ranking"]:
        idx = v["rank"] - 1
        letter = string.ascii_uppercase[idx % 26]
        suffix = "" if idx < 26 else str(idx // 26 + 1)
        v["ship_name"] = f"Vessel {letter}{suffix}"
        v["mmsi"] = "REDACTED"
        v["imo"] = "REDACTED"
    return anon
