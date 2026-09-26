# -*- coding: utf-8 -*-
"""[2단계] CLI 진입점. ``python mosfet.py idvg``로 Id-Vg 해석을 실행한다.

idvd / cv 모드는 과제다 — mosfet_tool의 비워 둔 함수를 채우면 동작한다.
실행: STEP2_RUN.bat 더블클릭
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mosfet_tool.config import Device, load_config
from mosfet_tool.workflows import run_cv, run_idvd, run_idvg, save_csv

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D nMOS TCAD simulator (project)")
    parser.add_argument("mode", choices=("idvg", "idvd", "cv"), nargs="?", default="idvg")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    device, sweeps = load_config(args.config)

    if args.mode == "idvg":
        curve = run_idvg(device, **sweeps.get("idvg", {}))
        save_csv("idvg.csv", curve)
        print(f"idvg: {len(curve)} points -> idvg.csv")
    elif args.mode == "idvd":
        curve = run_idvd(device, **sweeps.get("idvd", {}))
        save_csv("idvd.csv", curve)
        print(f"idvd: {len(curve)} points -> idvd.csv")
    else:
        curve = run_cv(device, **sweeps.get("cv", {}))
        save_csv("cv.csv", curve)
        print(f"cv: {len(curve)} points -> cv.csv")


if __name__ == "__main__":
    main()
