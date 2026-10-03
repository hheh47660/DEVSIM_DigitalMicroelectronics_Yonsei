# -*- coding: utf-8 -*-
"""제출용 구조 파일 검사기 (단독 실행, 배포 가능).

    python check_structure_file.py cell.devsim

OK      : 메시·영역·접점·계면·도핑만 들어 있음 -> 제출 가능
REJECT  : 방정식이나 물리 모델이 들어 있음 -> 물리 등록 전에 write_devices 로 다시 저장할 것
"""
import os
import sys

# 배포 환경(.conda)의 python.exe 를 직접 불러도 devsim DLL 을 찾도록 경로를 잡아 준다 (bat 없이 실행 가능)
if os.name == "nt":
    for _d in (os.path.join(sys.prefix, "Library", "bin"), sys.prefix):
        if os.path.isdir(_d):
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(_d)
            except (AttributeError, OSError):
                pass

import devsim

# 구조 파일에 있어도 되는 노드 모델: 좌표·기하·도핑
STRUCTURE_NODE_MODELS = {
    "AtContactNode", "ContactNSurfaceNormal_x", "ContactNSurfaceNormal_y", "ContactSurfaceArea",
    "NSurfaceNormal_x", "NSurfaceNormal_y", "NodeVolume", "SurfaceArea", "coordinate_index", "node_index",
    "x", "y", "z", "SourceDoping", "DrainDoping", "NetDoping", "TapDoping",
}

# ---- 재료 표 (채점 고정값) 와 이름 정규화/가이드 ----
GATE_TABLE = {"n+poly": 4.05, "Al": 4.10, "Ta": 4.25, "Ti": 4.33, "TaN": 4.45, "W": 4.60,
              "TiN": 4.65, "Mo": 4.70, "Ni": 5.10, "p+poly": 5.15, "Pt": 5.30}
CAP_TABLE = {"SiO2": 3.9, "Al2O3": 9.0, "HfO2": 20.0, "ZrO2": 35.0}
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _norm(name):
    n = name.translate(_SUB).replace(" ", "").replace("-", "").replace("_", "").lower()
    # PPT 표기 별칭: "n+ poly-Si" / "n+polySi" / "npoly" -> n+poly, p 도 같음
    for pre in ("n", "p"):
        if n in (f"{pre}+polysi", f"{pre}+polysilicon", f"{pre}poly", f"{pre}polysi", f"{pre}+poly"):
            return f"{pre}+poly"
    return n


def resolve_material(name, table, what):
    """(정규화된 표 키, None) 또는 (None, 안내 문구). 숫자를 쓰면 같은 값의 이름을 알려준다."""
    keys = {_norm(k): k for k in table}
    n = _norm(name)
    if n in keys:
        return keys[n], None
    try:
        val = float(name)
        hits = [k for k, v in table.items() if abs(v - val) < 1e-6]
        if hits:
            return None, f"{what} material 에 숫자 '{name}' 를 썼습니다. 값이 같은 재료 이름 '{hits[0]}' 로 바꿔 주세요 (이름으로만 받습니다)"
        return None, f"{what} material 에 숫자 '{name}' 를 썼습니다. 표에 그 값이 없습니다. 허용 이름: {list(table)}"
    except ValueError:
        pass
    close = [k for k in table if _norm(k).startswith(n[:2]) or n.startswith(_norm(k)[:2])] if len(n) >= 2 else []
    hint = f" 혹시 {close} 중 하나?" if close else ""
    return None, f"{what} material '{name}' 는 표에 없습니다.{hint} 허용 이름: {list(table)}"

REQUIRED_CONTACTS = {"gate", "source", "drain", "body"}
# 변수명 규칙
#   영역  : bulk(실리콘) · oxide(게이트 산화막) · gate_metal(게이트 재료 라벨) 은 고정.
#           hk 또는 hk_* = 캡 유전막 (개수·모양 자유).  air*/metal*/label* = 방정식 없는 채움/라벨 영역 (자유).
#   접점  : gate · source · drain · body 는 고정.  storage* = 저장 노드 면, plate* = 플레이트 면 (개수 자유).
#   계면  : bulk_oxide 는 고정. 그 밖의 계면은 이름 자유이되 bulk/oxide/hk* 영역 사이에만 둘 수 있다.
EXAMPLE_REGIONS = {"bulk", "oxide", "gate_metal", "hk_l", "hk_r", "air"}
EXAMPLE_CONTACTS = REQUIRED_CONTACTS | {"storage_l", "storage_r", "plate_l", "plate_r"}
CORE_CONTACT_REGION = {"gate": "oxide", "source": "bulk", "drain": "bulk", "body": "bulk"}


def region_role(r):
    if r in ("bulk", "oxide"): return "core"
    if r == "gate_metal": return "gate_metal"
    if r == "hk" or r.startswith("hk_"): return "hk"
    if r.startswith(("air", "metal", "label")): return "inert"
    return None


def contact_role(c):
    if c in REQUIRED_CONTACTS: return "core"
    if c.startswith("storage"): return "storage"
    if c == "plate" or c.startswith("plate_"): return "plate"
    return None


def _similar(name, known):
    """대소문자·공백·하이픈만 다른 이름, 또는 앞 3글자가 같은 이름을 제안."""
    n = _norm(name)
    exact = [k for k in known if _norm(k) == n]
    if exact:
        return exact
    return sorted(k for k in known if len(n) >= 3 and (_norm(k).startswith(n[:3]) or n.startswith(_norm(k)[:3])))


def validate(device):
    problems = []
    regions = list(devsim.get_region_list(device=device))
    contacts = list(devsim.get_contact_list(device=device))
    interfaces = list(devsim.get_interface_list(device=device))

    # ---- 물리가 섞여 있지 않은가 ----
    for region in regions:
        eqs = devsim.get_equation_list(device=device, region=region)
        if eqs:
            problems.append(f"region '{region}' 에 방정식 {list(eqs)} 가 들어 있음")
        extra = sorted(set(devsim.get_node_model_list(device=device, region=region)) - STRUCTURE_NODE_MODELS)
        if extra:
            problems.append(f"region '{region}' 에 구조 외 노드 모델 {extra[:8]}{' ...' if len(extra) > 8 else ''}")

    # ---- 영역 변수명 ----
    for r in regions:
        if region_role(r) is None:
            cand = _similar(r, EXAMPLE_REGIONS - set(regions))
            hint = f" → 변수명을 '{cand[0]}' 로 바꾸세요" if cand else \
                " (규칙: bulk/oxide/gate_metal 고정, 캡 유전막은 hk 또는 hk_*, 채움·라벨 영역은 air*/metal*/label*)"
            problems.append(f"영역 '{r}' 는 규칙에 없는 변수명입니다{hint}")
    for r in ("bulk", "oxide"):
        if r not in regions:
            problems.append(f"영역 누락: {r}")
    if "bulk" in regions and "NetDoping" not in devsim.get_node_model_list(device=device, region="bulk"):
        problems.append("bulk 에 노드 모델 NetDoping 이 없음 (도핑 전에 저장했거나 도핑 변수명이 다름)")
    hk_regions = [r for r in regions if region_role(r) == "hk"]
    solved = [r for r in regions if region_role(r) in ("core", "hk")]     # 조교 물리가 Potential 을 푸는 영역

    # ---- 접점 변수명 ----
    for c in contacts:
        if contact_role(c) is None:
            cand = _similar(c, EXAMPLE_CONTACTS - set(contacts))
            hint = f" → 변수명을 '{cand[0]}' 로 바꾸세요" if cand else \
                " (규칙: gate/source/drain/body 고정, 저장 노드 면은 storage*, 플레이트 면은 plate*)"
            problems.append(f"접점 '{c}' 는 규칙에 없는 변수명입니다{hint}")
    missing = sorted(REQUIRED_CONTACTS - set(contacts))
    if missing:
        problems.append(f"접점 누락: {missing}")
    storage = [c for c in contacts if contact_role(c) == "storage"]
    plate = [c for c in contacts if contact_role(c) == "plate"]
    # 접점이 붙은 영역
    for c in contacts:
        got = list(devsim.get_region_list(device=device, contact=c))
        role = contact_role(c)
        if role == "core":
            want = CORE_CONTACT_REGION[c]
            if want in regions and got and want not in got:
                problems.append(f"접점 '{c}' 가 영역 {got} 에 붙어 있음 (규칙: {want})")
        elif role in ("storage", "plate") and got and region_role(got[0]) != "hk":
            problems.append(f"접점 '{c}' 는 hk 영역 위에 있어야 하는데 영역 {got} 에 있음")
    # 캡 짝 맞음
    if hk_regions and (not storage or not plate):
        problems.append(f"캡 유전막 {hk_regions} 이 있으면 storage* 접점과 plate* 접점이 각각 1개 이상 필요 (현재 storage {storage}, plate {plate})")
    if (storage or plate) and not hk_regions:
        problems.append(f"storage*/plate* 접점이 있는데 캡 유전막(hk*) 영역이 없음: {storage + plate}")

    # ---- 계면 ----
    if "bulk_oxide" not in interfaces:
        # bulk 와 oxide 사이에 있는 계면이 다른 이름으로 있으면 그 이름을 지목
        cand = [i for i in interfaces if set(devsim.get_region_list(device=device, interface=i)) == {"bulk", "oxide"}]
        problems.append("계면 누락: bulk_oxide" + (f" (bulk-oxide 사이 계면 '{cand[0]}' 의 이름을 bulk_oxide 로 바꾸세요)" if cand else " (bulk 와 oxide 사이에 계면을 만들고 이름을 bulk_oxide 로)"))
    for i in interfaces:
        pair = list(devsim.get_region_list(device=device, interface=i))
        if i == "bulk_oxide" and pair and set(pair) != {"bulk", "oxide"}:
            problems.append(f"계면 'bulk_oxide' 가 영역 {pair} 사이에 있음 (규칙: bulk 와 oxide)")
        bad = [r for r in pair if r not in solved]
        if bad:
            problems.append(f"계면 '{i}' 가 방정식 없는 영역 {bad} 에 걸려 있음 (계면은 bulk/oxide/hk* 사이에만)")

    # ---- 좌표 규약: 실리콘 표면 y = 0, 실리콘은 y > 0(아래), 산화막·캡은 y < 0(위). 조교 역추출(치수·캡 높이)이 이 규약을 전제 ----
    if "bulk" in regions and "oxide" in regions:
        yb = devsim.get_node_model_values(device=device, region="bulk", name="y")
        yo = devsim.get_node_model_values(device=device, region="oxide", name="y")
        if min(yb) < -1e-9 or max(yo) > 1e-9:
            problems.append(f"좌표 규약 위반: 실리콘 표면이 y=0 이어야 함 (bulk y {min(yb):.2e}~{max(yb):.2e}, oxide y {min(yo):.2e}~{max(yo):.2e} cm). 배포 코드처럼 y 가 아래로 증가")

    # ---- 재료 이름 검사 (+ 가이드) ----
    # bulk 는 Silicon, oxide 는 SiO2(Oxide) 로 고정: 다른 이름을 써도 조교 물리는 Si/SiO2 로 계산하므로 미리 거부
    if "bulk" in regions and _norm(devsim.get_material(device=device, region="bulk")) not in ("silicon", "si"):
        problems.append(f"bulk material 은 Silicon 이어야 합니다 (현재 '{devsim.get_material(device=device, region='bulk')}')")
    if "oxide" in regions and _norm(devsim.get_material(device=device, region="oxide")) not in ("oxide", "sio2"):
        problems.append(f"oxide(게이트 산화막) material 은 Oxide 또는 SiO2 여야 합니다 (현재 '{devsim.get_material(device=device, region='oxide')}'). 게이트 산화막은 SiO2 고정")
    for region in hk_regions:
        mat = devsim.get_material(device=device, region=region)
        if _norm(mat) in ("oxide", "sio2"):
            continue                      # SiO2 로 계산
        key, msg = resolve_material(mat, CAP_TABLE, f"유전막({region})")
        if msg: problems.append(msg)
    if "gate_metal" in regions:
        mat = devsim.get_material(device=device, region="gate_metal")
        key, msg = resolve_material(mat, GATE_TABLE, "게이트(gate_metal)")
        if msg: problems.append(msg)
    else:
        problems.append("gate_metal 영역이 없어 게이트 재료를 알 수 없습니다: 산화막 위에 방정식 없는 영역 gate_metal 을 만들고 material 에 과제 PPT 재료 표의 게이트 이름을 쓰세요")
    return problems


def main():
    try:   # 학생 이름에 든 특수문자(아래첨자 등)가 콘솔 인코딩(cp949)에 없어도 죽지 않게
        sys.stdout.reconfigure(errors="replace"); sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print("usage: python check_structure_file.py cell.devsim"); return 1
    path = sys.argv[1]
    devsim.load_devices(file=path)
    dev = devsim.get_device_list()[0]
    problems = validate(dev)
    if problems:
        print("REJECT:", path)
        print("  아래 항목을 고친 뒤 다시 저장해 주세요:")
        for p in problems:
            print("  -", p)
        return 2
    mats = {r: devsim.get_material(device=dev, region=r) for r in devsim.get_region_list(device=dev)}
    print("OK:", path)
    print("  regions/materials:", mats)
    print("  contacts:", sorted(devsim.get_contact_list(device=dev)))
    print("  interfaces:", sorted(devsim.get_interface_list(device=dev)))
    return 0


if __name__ == "__main__":
    sys.exit(main())