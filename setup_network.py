"""Generate SUMO grid network files for SmartMARL."""

from __future__ import annotations

import glob
import os
import random
import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


GRID_DIR = Path("smartmarl/configs/grid5x5")


def write_sumocfg(path: Path, route_files: str, net_file: str = "grid5x5.net.xml") -> None:
    path.write_text(
        f"""<configuration>
    <input>
        <net-file value=\"{net_file}\"/>
        <route-files value=\"{route_files}\"/>
    </input>
    <time>
        <begin value=\"0\"/>
        <end value=\"3600\"/>
    </time>
</configuration>
""",
        encoding="utf-8",
    )


def write_fallback_net(path: Path) -> None:
    path.write_text(
        """<net version=\"1.16\">
    <location netOffset=\"0.00,0.00\" convBoundary=\"0.00,0.00,2000.00,2000.00\" origBoundary=\"0.00,0.00,2000.00,2000.00\"/>
</net>
""",
        encoding="utf-8",
    )


def extract_valid_edges(net_file: Path) -> list[str]:
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        edges = []
        for edge in root.findall("edge"):
            edge_id = edge.attrib.get("id", "")
            function = edge.attrib.get("function", "")
            if edge_id and not edge_id.startswith(":") and function != "internal":
                edges.append(edge_id)
        return edges
    except Exception:
        return []


def write_routes_standard_fallback(path: Path, edge_id: str) -> None:
    path.write_text(
        f"""<routes>
    <vType id=\"car\" accel=\"2.6\" decel=\"4.5\" sigma=\"0.5\" length=\"4.5\" maxSpeed=\"14\"/>
    <route id=\"r0\" edges=\"{edge_id}\"/>
    <flow id=\"f0\" type=\"car\" route=\"r0\" begin=\"0\" end=\"3600\" number=\"800\"/>
</routes>
""",
        encoding="utf-8",
    )


def find_random_trips_script() -> str | None:
    sumo_home = os.environ.get("SUMO_HOME", "").strip()
    candidates = []
    if sumo_home:
        candidates.append(str(Path(sumo_home) / "tools" / "randomTrips.py"))

    candidates += glob.glob(".venv/lib/python*/site-packages/sumo/tools/randomTrips.py")

    for c in candidates:
        if Path(c).exists():
            return c
    return None


def generate_standard_routes_with_random_trips(net_file: Path, route_file: Path) -> bool:
    random_trips = find_random_trips_script()
    if not random_trips:
        return False

    python_bin = str(Path(".venv/bin/python")) if Path(".venv/bin/python").exists() else "python3"
    env = os.environ.copy()
    env["PATH"] = f"{Path('.venv/bin').resolve()}:{env.get('PATH', '')}"

    cmd = [
        python_bin,
        random_trips,
        "-n",
        str(net_file),
        "-r",
        str(route_file),
        "--seed",
        "42",
        "--begin",
        "0",
        "--end",
        "3600",
        "--period",
        "1.2",
        "--min-distance",
        "600",
        "--fringe-factor",
        "5",
        "--validate",
    ]
    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def make_indian_routes_from_standard(standard_routes: Path, indian_routes: Path, seed: int = 42) -> bool:
    try:
        tree = ET.parse(standard_routes)
        root = tree.getroot()

        for child in list(root):
            if child.tag == "vType":
                root.remove(child)

        vtypes = [
            ET.Element("vType", {
                "id": "motorcycle",
                "length": "1.8",
                "maxSpeed": "15",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.2",
            }),
            ET.Element("vType", {
                "id": "auto_rickshaw",
                "length": "3.5",
                "maxSpeed": "12",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.2",
            }),
            ET.Element("vType", {
                "id": "car",
                "length": "4.5",
                "maxSpeed": "14",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.5",
            }),
            ET.Element("vType", {
                "id": "heavy_vehicle",
                "length": "9.0",
                "maxSpeed": "10",
                "laneChangeModel": "SL2015",
                "lcCooperative": "0.8",
            }),
        ]

        insert_idx = 0
        for idx, child in enumerate(list(root)):
            if child.tag in {"trip", "vehicle", "flow"}:
                insert_idx = idx
                break

        for vt in reversed(vtypes):
            root.insert(insert_idx, vt)

        rng = random.Random(seed)
        types = ["motorcycle", "auto_rickshaw", "car", "heavy_vehicle"]
        probs = [0.45, 0.15, 0.30, 0.10]

        for child in root:
            if child.tag in {"trip", "vehicle", "flow"}:
                t = rng.choices(types, weights=probs, k=1)[0]
                child.set("type", t)

        tree.write(indian_routes, encoding="utf-8", xml_declaration=False)
        return True
    except Exception:
        return False


def main() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)

    net_file = GRID_DIR / "grid5x5.net.xml"
    netgenerate = shutil.which("netgenerate")
    if not netgenerate and Path(".venv/bin/netgenerate").exists():
        netgenerate = str(Path(".venv/bin/netgenerate"))

    if netgenerate:
        cmd = [
            netgenerate,
            "--grid",
            "--grid.number=5",
            "--grid.length=400",
            f"--output-file={net_file}",
            "--default.speed=13.9",
            "--tls.guess=true",
        ]
        try:
            subprocess.run(cmd, check=True)
            print("Generated SUMO network using netgenerate")
        except subprocess.CalledProcessError:
            write_fallback_net(net_file)
            print("netgenerate failed; wrote fallback network file")
    else:
        write_fallback_net(net_file)
        print("netgenerate not found; wrote fallback network file")

    standard_routes = GRID_DIR / "grid5x5.rou.xml"
    indian_routes = GRID_DIR / "grid5x5_indian.rou.xml"

    random_ok = generate_standard_routes_with_random_trips(net_file, standard_routes)

    if not random_ok:
        edge_ids = extract_valid_edges(net_file)
        route_edge = edge_ids[0] if edge_ids else "edge_fallback"
        write_routes_standard_fallback(standard_routes, route_edge)
        print(f"randomTrips unavailable; wrote fallback routes on edge '{route_edge}'")

    indian_ok = make_indian_routes_from_standard(standard_routes, indian_routes)
    if not indian_ok:
        edge_ids = extract_valid_edges(net_file)
        route_edge = edge_ids[0] if edge_ids else "edge_fallback"
        write_routes_standard_fallback(indian_routes, route_edge)
        print("Failed to synthesize Indian route mix; wrote fallback Indian routes")

    write_sumocfg(GRID_DIR / "grid5x5.sumocfg", route_files="grid5x5.rou.xml")
    write_sumocfg(GRID_DIR / "grid5x5_indian.sumocfg", route_files="grid5x5_indian.rou.xml")

    print(f"Network artifacts written to {GRID_DIR}")


if __name__ == "__main__":
    main()
