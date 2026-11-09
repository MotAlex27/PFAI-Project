import json
import tempfile
import unittest
from pathlib import Path

from generate_scenario import (
    LANE_COUNT,
    SIGNALS,
    build_flows,
    build_roadnet,
    generate,
)


class ScenarioTests(unittest.TestCase):
    def test_corridor_has_two_signalized_four_approach_intersections(self) -> None:
        roadnet = build_roadnet()
        signalized = [
            item
            for item in roadnet["intersections"]
            if not item["virtual"]
        ]
        self.assertEqual({item["id"] for item in signalized}, set(SIGNALS))
        for intersection in signalized:
            self.assertEqual(len(intersection["roads"]), 8)
            self.assertEqual(len(intersection["trafficLight"]["lightphases"]), 3)
            for road in roadnet["roads"]:
                if road["endIntersection"] == intersection["id"]:
                    self.assertEqual(len(road["lanes"]), LANE_COUNT)
            self.assertTrue(
                all(
                    intersection["trafficLight"]["lightphases"][phase][
                        "availableRoadLinks"
                    ]
                    for phase in (0, 1)
                )
            )

    def test_flow_routes_are_connected_and_have_vehicle_attributes(self) -> None:
        roadnet = build_roadnet()
        roads = {road["id"]: road for road in roadnet["roads"]}
        intersections = {
            item["id"]: item
            for item in roadnet["intersections"]
            if not item["virtual"]
        }
        flows = build_flows()
        self.assertGreaterEqual(len(flows), 8)
        for flow in flows:
            self.assertGreaterEqual(len(flow["route"]), 2)
            self.assertTrue(set(flow["route"]).issubset(roads))
            self.assertGreater(flow["interval"], 0)
            self.assertIn("maxSpeed", flow["vehicle"])
            for current, following in zip(flow["route"], flow["route"][1:]):
                self.assertEqual(
                    roads[current]["endIntersection"],
                    roads[following]["startIntersection"],
                )
                junction_id = roads[current]["endIntersection"]
                if junction_id in intersections:
                    movements = intersections[junction_id]["roadLinks"]
                    self.assertTrue(
                        any(
                            movement["startRoad"] == current
                            and movement["endRoad"] == following
                            for movement in movements
                        )
                    )

    def test_signal_movements_cover_turns_and_connect_to_existing_roads(self) -> None:
        roadnet = build_roadnet()
        road_ids = {road["id"] for road in roadnet["roads"]}
        for intersection in roadnet["intersections"]:
            if intersection["virtual"]:
                continue
            movements = intersection["roadLinks"]
            self.assertTrue(any(link["type"] != "go_straight" for link in movements))
            for link in movements:
                self.assertIn(link["startRoad"], road_ids)
                self.assertIn(link["endRoad"], road_ids)
                self.assertEqual(
                    len(link["laneLinks"]), LANE_COUNT * LANE_COUNT
                )
                self.assertEqual(
                    {
                        (lane_link["startLaneIndex"], lane_link["endLaneIndex"])
                        for lane_link in link["laneLinks"]
                    },
                    {
                        (start_lane, end_lane)
                        for start_lane in range(LANE_COUNT)
                        for end_lane in range(LANE_COUNT)
                    },
                )

    def test_generated_config_is_portable_and_references_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            generate(output)
            config = json.loads(
                (output / "cityflow_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["dir"], ".")
            self.assertTrue((output / config["roadnetFile"]).is_file())
            self.assertTrue((output / config["flowFile"]).is_file())


if __name__ == "__main__":
    unittest.main()
