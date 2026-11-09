"""Generate the deterministic two-intersection CityFlow corridor."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "corridor_2x1"
ROAD_LENGTH = 250.0
LANE_COUNT = 2
LANE_WIDTH = 3.2
MAX_SPEED = 13.89

NODES = {
    "J1": (0.0, 0.0),
    "J2": (ROAD_LENGTH, 0.0),
    "W": (-ROAD_LENGTH, 0.0),
    "N1": (0.0, ROAD_LENGTH),
    "S1": (0.0, -ROAD_LENGTH),
    "E": (2.0 * ROAD_LENGTH, 0.0),
    "N2": (ROAD_LENGTH, ROAD_LENGTH),
    "S2": (ROAD_LENGTH, -ROAD_LENGTH),
}
CONNECTIONS = (
    ("W", "J1"),
    ("J1", "J2"),
    ("J2", "E"),
    ("N1", "J1"),
    ("J1", "S1"),
    ("N2", "J2"),
    ("J2", "S2"),
)
SIGNALS = ("J1", "J2")


def road_id(start: str, end: str) -> str:
    return f"road_{start}_{end}"


def build_roadnet() -> dict:
    roads = []
    incident_roads: dict[str, list[str]] = {node: [] for node in NODES}
    for first, second in CONNECTIONS:
        for start, end in ((first, second), (second, first)):
            roads.append(
                {
                    "id": road_id(start, end),
                    "points": [
                        {"x": NODES[start][0], "y": NODES[start][1]},
                        {"x": NODES[end][0], "y": NODES[end][1]},
                    ],
                    "startIntersection": start,
                    "endIntersection": end,
                    "lanes": [
                        {"width": LANE_WIDTH, "maxSpeed": MAX_SPEED}
                        for _ in range(LANE_COUNT)
                    ],
                }
            )
            incident_roads[start].append(road_id(start, end))
            incident_roads[end].append(road_id(start, end))

    intersections = []
    for node_id, point in NODES.items():
        if node_id not in SIGNALS:
            intersections.append(
                {
                    "id": node_id,
                    "point": {"x": point[0], "y": point[1]},
                    "width": 0,
                    "virtual": True,
                    "roads": incident_roads[node_id],
                    "roadLinks": [],
                    "trafficLight": {"roadLinkIndices": [], "lightphases": []},
                }
            )
            continue

        incoming = [
            road
            for road in roads
            if road["endIntersection"] == node_id
        ]
        outgoing = [
            road
            for road in roads
            if road["startIntersection"] == node_id
        ]
        links = []
        for inbound in incoming:
            source = inbound["startIntersection"]
            sx, sy = NODES[source]
            ix, iy = point
            vertical_approach = sx == ix
            for outbound in outgoing:
                target = outbound["endIntersection"]
                if target == source:
                    continue
                tx, ty = NODES[target]
                incoming_vector = (ix - sx, iy - sy)
                outgoing_vector = (tx - ix, ty - iy)
                cross_product = (
                    incoming_vector[0] * outgoing_vector[1]
                    - incoming_vector[1] * outgoing_vector[0]
                )
                if incoming_vector == outgoing_vector:
                    movement_type = "go_straight"
                elif cross_product > 0:
                    movement_type = "turn_left"
                else:
                    movement_type = "turn_right"
                links.append(
                    {
                        "type": movement_type,
                        "startRoad": inbound["id"],
                        "endRoad": outbound["id"],
                        "laneLinks": [
                            {
                                "startLaneIndex": start_lane,
                                "endLaneIndex": end_lane,
                            }
                            for start_lane in range(LANE_COUNT)
                            for end_lane in range(LANE_COUNT)
                        ],
                        "_phase": 0 if vertical_approach else 1,
                    }
                )

        phases = [
            {
                "time": 30,
                "availableRoadLinks": [
                    index
                    for index, link in enumerate(links)
                    if link["_phase"] == phase
                ],
            }
            for phase in (0, 1)
        ]
        phases.append({"time": 3, "availableRoadLinks": []})
        for link in links:
            del link["_phase"]

        intersections.append(
            {
                "id": node_id,
                "point": {"x": point[0], "y": point[1]},
                "width": 12,
                "virtual": False,
                "roads": incident_roads[node_id],
                "roadLinks": links,
                "trafficLight": {
                    "roadLinkIndices": list(range(len(links))),
                    "lightphases": phases,
                },
            }
        )

    return {"intersections": intersections, "roads": roads}


def _routes_from(start: str) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {node: [] for node in NODES}
    for first, second in CONNECTIONS:
        graph[first].append(second)
        graph[second].append(first)

    routes = {start: []}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        for neighbor in graph[current]:
            if neighbor in routes:
                continue
            routes[neighbor] = [
                *routes[current],
                road_id(current, neighbor),
            ]
            pending.append(neighbor)
    return routes


def build_flows(end_time: int = 3600) -> list[dict]:
    vehicle = {
        "length": 5.0,
        "width": 1.8,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": MAX_SPEED,
        "headwayTime": 1.5,
    }
    od_pairs = (
        ("W", "E", 2.5),
        ("E", "W", 4.0),
        ("N1", "S1", 5.0),
        ("S1", "N1", 5.0),
        ("N2", "S2", 5.0),
        ("S2", "N2", 5.0),
        ("N1", "E", 7.0),
        ("S2", "W", 7.0),
        ("W", "N2", 7.0),
        ("E", "S1", 7.0),
    )
    route_cache = {start: _routes_from(start) for start, _, _ in od_pairs}
    return [
        {
            "vehicle": vehicle,
            "route": route_cache[start][end],
            "interval": interval,
            "startTime": 0,
            "endTime": end_time,
        }
        for start, end, interval in od_pairs
    ]


def generate(output_dir: Path, end_time: int = 3600) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "roadnet.json").write_text(
        json.dumps(build_roadnet(), indent=2), encoding="utf-8"
    )
    (output_dir / "flow.json").write_text(
        json.dumps(build_flows(end_time), indent=2), encoding="utf-8"
    )
    config = {
        "interval": 1.0,
        "seed": 0,
        "dir": ".",
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "roadnet_log.txt",
        "replayLogFile": "replay.txt",
    }
    (output_dir / "cityflow_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Generated 2-junction CityFlow corridor at {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--end-time", type=int, default=3600)
    args = parser.parse_args()
    if args.end_time <= 0:
        parser.error("--end-time must be positive.")
    generate(args.output, args.end_time)


if __name__ == "__main__":
    main()
