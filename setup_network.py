"""Generate and validate real SUMO assets for SmartMARL."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GRID_DIR = Path("smartmarl/configs/grid5x5")
STANDARD_NET = GRID_DIR / "grid5x5.net.xml"
STANDARD_ROUTES = GRID_DIR / "grid5x5.rou.xml"
INDIAN_ROUTES = GRID_DIR / "grid5x5_indian.rou.xml"
STANDARD_CFG = GRID_DIR / "grid5x5.sumocfg"
INDIAN_CFG = GRID_DIR / "grid5x5_indian.sumocfg"


@dataclass(frozen=True)
class SumoAssets:
    net_file: Path
    standard_routes: Path
    indian_routes: Path
    standard_sumocfg: Path
    indian_sumocfg: Path

    def all_files(self) -> tuple[Path, ...]:
        return (
            self.net_file,
            self.standard_routes,
            self.indian_routes,
            self.standard_sumocfg,
            self.indian_sumocfg,
        )


def assets_bundle(grid_dir: Path = GRID_DIR) -> SumoAssets:
    return SumoAssets(
        net_file=grid_dir / "grid5x5.net.xml",
        standard_routes=grid_dir / "grid5x5.rou.xml",
        indian_routes=grid_dir / "grid5x5_indian.rou.xml",
        standard_sumocfg=grid_dir / "grid5x5.sumocfg",
        indian_sumocfg=grid_dir / "grid5x5_indian.sumocfg",
    )


def write_sumocfg(path: Path, route_files: str, net_file: str = "grid5x5.net.xml") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "<configuration>\n"
            "    <input>\n"
            f"        <net-file value=\"{net_file}\"/>\n"
            f"        <route-files value=\"{route_files}\"/>\n"
            "    </input>\n"
            "    <time>\n"
            "        <begin value=\"0\"/>\n"
            "        <end value=\"3600\"/>\n"
            "    </time>\n"
            "</configuration>\n"
        ),
        encoding="utf-8",
    )


def _resolve_binary(name: str) -> str | None:
    direct = shutil.which(name)
    if direct:
        return direct

    local = Path(".venv/bin") / name
    if local.exists():
        return str(local.resolve())
    return None


def find_random_trips_script() -> str | None:
    candidates = []
    sumo_home = os.environ.get("SUMO_HOME", "").strip()
    if sumo_home:
        candidates.append(Path(sumo_home) / "tools" / "randomTrips.py")

    sumo_bin = _resolve_binary("sumo")
    if sumo_bin:
        sumo_path = Path(sumo_bin).resolve()
        candidates.extend(
            [
                sumo_path.parent.parent / "share" / "sumo" / "tools" / "randomTrips.py",
                sumo_path.parent.parent.parent / "share" / "sumo" / "tools" / "randomTrips.py",
            ]
        )

    for pattern in (".venv/lib/python*/site-packages/sumo/tools/randomTrips.py",):
        for match in Path(".").glob(pattern):
            candidates.append(match)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _run(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess[str]:
    kwargs = {"text": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(cmd, check=True, **kwargs)


def extract_valid_edges(net_file: Path) -> list[str]:
    tree = ET.parse(net_file)
    root = tree.getroot()
    edges = []
    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id", "")
        function = edge.attrib.get("function", "")
        if edge_id and not edge_id.startswith(":") and function != "internal":
            edges.append(edge_id)
    return edges


def _junction_count(net_file: Path) -> int:
    tree = ET.parse(net_file)
    return len(tree.getroot().findall("junction"))


def validate_net_file(net_file: Path, min_edges: int = 20, min_junctions: int = 25) -> bool:
    if not net_file.exists():
        return False
    try:
        edges = extract_valid_edges(net_file)
        junctions = _junction_count(net_file)
    except Exception:
        return False
    return len(edges) >= min_edges and junctions >= min_junctions


def _route_like_nodes(root: ET.Element) -> list[ET.Element]:
    return [child for child in root if child.tag in {"trip", "vehicle", "flow"}]


def validate_route_file(route_file: Path, min_routes: int = 50) -> bool:
    if not route_file.exists():
        return False
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()
        route_nodes = _route_like_nodes(root)
        if len(route_nodes) < min_routes:
            return False
        return True
    except Exception:
        return False


def validate_sumocfg(sumocfg: Path, route_file_name: str) -> bool:
    if not sumocfg.exists():
        return False
    try:
        tree = ET.parse(sumocfg)
        root = tree.getroot()
        route_node = root.find("./input/route-files")
        net_node = root.find("./input/net-file")
        if route_node is None or net_node is None:
            return False
        return (
            route_node.attrib.get("value") == route_file_name
            and net_node.attrib.get("value") == "grid5x5.net.xml"
        )
    except Exception:
        return False


def validate_assets(bundle: SumoAssets) -> bool:
    return (
        validate_net_file(bundle.net_file)
        and validate_route_file(bundle.standard_routes)
        and validate_route_file(bundle.indian_routes)
        and validate_sumocfg(bundle.standard_sumocfg, "grid5x5.rou.xml")
        and validate_sumocfg(bundle.indian_sumocfg, "grid5x5_indian.rou.xml")
    )


def generate_grid_network(
    net_file: Path,
    grid_number: int = 5,
    grid_length: int = 400,
    force: bool = False,
) -> None:
    if net_file.exists() and not force and validate_net_file(net_file):
        return

    netgenerate = _resolve_binary("netgenerate")
    if not netgenerate:
        raise RuntimeError("netgenerate not found. Install SUMO tools or set SUMO_HOME.")

    cmd = [
        netgenerate,
        "--grid",
        f"--grid.number={grid_number}",
        f"--grid.length={grid_length}",
        f"--output-file={net_file}",
        "--default.speed=13.9",
        "--tls.guess=true",
        "--junctions.corner-detail=0",
        "--no-turnarounds=true",
    ]
    _run(cmd)
    if not validate_net_file(net_file):
        raise RuntimeError(f"Generated network is invalid: {net_file}")


def generate_standard_routes_with_random_trips(
    net_file: Path,
    route_file: Path,
    *,
    seed: int = 42,
    duration_seconds: int = 3600,
    period_seconds: float = 1.2,
    force: bool = False,
) -> None:
    if route_file.exists() and not force and validate_route_file(route_file):
        return

    random_trips = find_random_trips_script()
    if not random_trips:
        raise RuntimeError("randomTrips.py not found. Install SUMO tools or set SUMO_HOME.")

    python_bin = shutil.which("python3") or shutil.which("python") or "python3"
    cmd = [
        python_bin,
        random_trips,
        "-n",
        str(net_file),
        "-r",
        str(route_file),
        "--seed",
        str(seed),
        "--begin",
        "0",
        "--end",
        str(duration_seconds),
        "--period",
        str(period_seconds),
        "--min-distance",
        "600",
        "--fringe-factor",
        "5",
        "--validate",
        "--trip-attributes",
        'departLane="best" departSpeed="max" departPos="base"',
    ]
    _run(cmd)
    if not validate_route_file(route_file):
        raise RuntimeError(f"Generated standard routes are invalid: {route_file}")


def _remove_vehicle_types(root: ET.Element) -> None:
    for child in list(root):
        if child.tag == "vType":
            root.remove(child)


def _insert_vtypes(root: ET.Element, vtypes: Iterable[ET.Element]) -> None:
    insert_idx = 0
    for idx, child in enumerate(list(root)):
        if child.tag in {"trip", "vehicle", "flow"}:
            insert_idx = idx
            break

    for vtype in reversed(list(vtypes)):
        root.insert(insert_idx, vtype)


def make_indian_routes_from_standard(
    standard_routes: Path,
    indian_routes: Path,
    *,
    seed: int = 42,
    force: bool = False,
) -> None:
    if indian_routes.exists() and not force and validate_route_file(indian_routes):
        return

    tree = ET.parse(standard_routes)
    root = tree.getroot()
    _remove_vehicle_types(root)

    # Software-only but literature-grounded vehicle definitions for Indian mixed traffic.
    vtypes = [
        ET.Element(
            "vType",
            {
                "id": "motorcycle",
                "vClass": "motorcycle",
                "length": "2.0",
                "minGap": "0.8",
                "accel": "3.6",
                "decel": "5.0",
                "tau": "0.7",
                "sigma": "0.6",
                "maxSpeed": "16.0",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.10",
                "lcSpeedGain": "1.8",
            },
        ),
        ET.Element(
            "vType",
            {
                "id": "auto_rickshaw",
                "vClass": "taxi",
                "length": "3.2",
                "minGap": "1.0",
                "accel": "2.3",
                "decel": "4.2",
                "tau": "0.95",
                "sigma": "0.5",
                "maxSpeed": "12.5",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.20",
                "lcSpeedGain": "1.4",
            },
        ),
        ET.Element(
            "vType",
            {
                "id": "car",
                "vClass": "passenger",
                "length": "4.5",
                "minGap": "2.0",
                "accel": "2.6",
                "decel": "4.5",
                "tau": "1.1",
                "sigma": "0.5",
                "maxSpeed": "14.0",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.35",
                "lcSpeedGain": "1.3",
            },
        ),
        ET.Element(
            "vType",
            {
                "id": "heavy_vehicle",
                "vClass": "truck",
                "length": "9.0",
                "minGap": "2.5",
                "accel": "1.3",
                "decel": "3.2",
                "tau": "1.6",
                "sigma": "0.4",
                "maxSpeed": "10.0",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.65",
                "lcSpeedGain": "1.0",
            },
        ),
    ]
    _insert_vtypes(root, vtypes)

    rng = random.Random(seed)
    types = ["motorcycle", "auto_rickshaw", "car", "heavy_vehicle"]
    probs = [0.45, 0.15, 0.30, 0.10]

    for child in _route_like_nodes(root):
        child.set("type", rng.choices(types, weights=probs, k=1)[0])

    tree.write(indian_routes, encoding="utf-8", xml_declaration=False)
    if not validate_route_file(indian_routes):
        raise RuntimeError(f"Generated Indian heterogeneous routes are invalid: {indian_routes}")


def ensure_sumo_assets(
    *,
    grid_dir: Path = GRID_DIR,
    force_regenerate: bool = False,
    strict_tools: bool = False,
    seed: int = 42,
    duration_seconds: int = 3600,
) -> SumoAssets:
    bundle = assets_bundle(grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    if validate_assets(bundle) and not force_regenerate:
        return bundle

    have_tools = bool(_resolve_binary("netgenerate")) and bool(find_random_trips_script())
    if strict_tools and not have_tools:
        raise RuntimeError(
            "SUMO asset generation requires real SUMO tools (netgenerate + randomTrips.py), "
            "but they were not found."
        )

    if not have_tools and validate_assets(bundle):
        return bundle

    if not have_tools:
        raise RuntimeError(
            "SUMO tools are unavailable and existing assets are missing or invalid. "
            "Install SUMO tools before continuing."
        )

    generate_grid_network(bundle.net_file, force=force_regenerate)
    generate_standard_routes_with_random_trips(
        bundle.net_file,
        bundle.standard_routes,
        seed=seed,
        duration_seconds=duration_seconds,
        force=force_regenerate,
    )
    make_indian_routes_from_standard(
        bundle.standard_routes,
        bundle.indian_routes,
        seed=seed,
        force=force_regenerate,
    )
    write_sumocfg(bundle.standard_sumocfg, route_files="grid5x5.rou.xml")
    write_sumocfg(bundle.indian_sumocfg, route_files="grid5x5_indian.rou.xml")

    if not validate_assets(bundle):
        raise RuntimeError("SUMO assets failed validation after generation.")
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real SUMO assets for SmartMARL.")
    parser.add_argument("--grid-dir", default=str(GRID_DIR))
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Require real SUMO tools to be available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = ensure_sumo_assets(
        grid_dir=Path(args.grid_dir),
        force_regenerate=bool(args.force_regenerate),
        strict_tools=bool(args.strict),
        seed=int(args.seed),
        duration_seconds=int(args.duration_seconds),
    )
    print("SUMO assets ready:")
    for path in bundle.all_files():
        print(" -", path)


if __name__ == "__main__":
    main()
