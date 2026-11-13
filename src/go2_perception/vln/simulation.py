"""One synthetic MuJoCo camera proxy. No robot hardware/network dependencies.

Kinematic proposals have no Go2 locomotion interpretation. Geometry/contact tests
are evaluator-only; the policy receives RGB and instruction. Static step-and-wait.
"""

import math

import numpy as np


class CameraRoom:
    def __init__(self, scene):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.proxy = self.model.geom("proxy_collision").id
        self.floor = self.model.geom("floor").id
        self.clock = 0.0
        self.pose = np.array([4.3, 0.0, math.radians(25)])
        self.renderer = None
        self._place()

    def _place(self):
        x, y, yaw = self.pose
        self.data.mocap_pos[0] = [x, y, 0.25]
        self.data.mocap_quat[0] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.mj.mj_forward(self.model, self.data)

    def render(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=256, width=448)
        self.renderer.update_scene(self.data, camera="rgb")
        return self.renderer.render().copy()

    def contacts(self):
        # Static/mocap pairs are filtered from the dynamics contact list. Query
        # signed geom distances for this kinematic proxy instead of treating a
        # zero dynamics-contact count as clearance.
        collisions = []
        for geom in range(self.model.ngeom):
            if geom in (self.proxy, self.floor):
                continue
            distance = self.mj.mj_geomDistance(
                self.model, self.data, self.proxy, geom, 0.01, None
            )
            if distance < 0:
                collisions.append(
                    {
                        "geom1": self.proxy,
                        "geom2": geom,
                        "distance": float(distance),
                        "method": "mj_geomDistance",
                    }
                )
        return collisions

    def advance(self, first_waypoint):
        """Apply first absolute ego waypoint over 0.25 simulated seconds, with substeps.

        Collision-bound rejection is a simulator constraint, not learned avoidance.
        Lateral motion is permitted for this kinematic camera proxy, not a TurtleBot.
        """
        step = np.asarray(first_waypoint, dtype=float)
        if step.shape != (3,) or not np.isfinite(step).all():
            raise ValueError("Expected finite [forward_m,left_m,yaw_rad]")
        if np.linalg.norm(step[:2]) > 0.25 or abs(step[2]) > math.radians(32):
            return {
                "applied": False,
                "reason": "step_outside_proxy_bounds",
                "actual": [0.0, 0.0, 0.0],
            }
        start = self.pose.copy()
        yaw = start[2]
        world = np.array(
            [
                math.cos(yaw) * step[0] - math.sin(yaw) * step[1],
                math.sin(yaw) * step[0] + math.cos(yaw) * step[1],
                step[2],
            ]
        )
        for fraction in np.linspace(0.1, 1, 10):
            self.pose = start + fraction * world
            self._place()
            contacts = self.contacts()
            if contacts:
                self.pose = start
                self._place()
                return {
                    "applied": False,
                    "reason": "simulator_contact_constraint",
                    "contacts": contacts,
                    "actual": [0.0, 0.0, 0.0],
                }
        self.clock += 0.25
        self.data.time = self.clock
        return {
            "applied": True,
            "actual": step.tolist(),
            "simulated_dt_s": 0.25,
            "contacts": [],
        }

    def evaluation(self):
        x, y, _ = self.pose
        return {
            "pose_xy_yaw": self.pose.tolist(),
            "goal_region": {"x_range": [5.8, 6.5], "y_range": [-0.6, 0.6]},
            "inside_stopping_region": bool(5.8 <= x <= 6.5 and abs(y) <= 0.6),
            "distance_to_goal_center_m": float(
                np.linalg.norm(self.pose[:2] - [6.15, 0.0])
            ),
            "contacts": self.contacts(),
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
