# -*- coding: utf-8 -*-
"""소자 파라미터 dataclass와 config.yaml 로더."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Device:
    """소자 치수와 도핑 (단위는 필드 이름에 표시)."""

    gate_length_um: float = 1.0
    source_length_um: float = 0.5
    drain_length_um: float = 0.5
    oxide_thickness_nm: float = 10.0
    silicon_thickness_um: float = 0.5
    junction_depth_um: float = 0.1
    body_doping_cm3: float = 1.0e16
    sd_doping_cm3: float = 1.0e19
    temperature_k: float = 300.0


def load_config(path: str | Path) -> tuple[Device, dict]:
    """config.yaml을 읽어 (Device, sweeps 딕셔너리)를 돌려준다."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    device = Device(**data.get("device", {}))
    sweeps = data.get("sweeps", {})
    return device, sweeps
