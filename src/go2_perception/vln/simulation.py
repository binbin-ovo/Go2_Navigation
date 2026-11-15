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

    def execute_prefix(self, waypoints, *, native_stop=False, limits=None):
        """Opt-in Phase 6D follower; the Phase 6C advance() baseline is unchanged.

        Commit only collision-free substeps. On interruption retain the last safe
        pose and report the rejected candidate, never teleport to a later target.
        """
        from go2_perception.vln.waypoints import bounded_prefix, path_errors

        plan = bounded_prefix(
            waypoints, self.pose, native_stop=native_stop, limits=limits
        )
        started = self.clock
        actual = [self.pose.tolist()]
        times = [0.0]
        rejected = None
        contacts = self.contacts()
        if contacts:
            rejected = {
                "reason": "initial_contact",
                "pose": self.pose.tolist(),
                "contacts": contacts,
            }
        else:
            for t, candidate in zip(
                plan["planned_times_s"][1:], plan["planned_world_poses"][1:]
            ):
                before = self.pose.copy()
                self.pose = np.asarray(candidate, dtype=float)
                self._place()
                contacts = self.contacts()
                if contacts:
                    rejected = {
                        "reason": "simulator_contact_constraint",
                        "pose": candidate,
                        "contacts": contacts,
                    }
                    self.pose = before
                    self._place()
                    break
                self.clock = started + t
                self.data.time = self.clock
                actual.append(candidate)
                times.append(t)
        return plan | {
            "applied": rejected is None,
            "constraint_rejection": rejected,
            "actual_world_poses": actual,
            "actual_times_s": times,
            "simulated_dt_s": self.clock - started,
            "actual_endpoint": self.pose.tolist(),
            "path_error": path_errors(actual, plan["decision_pose"], waypoints),
        }

    def execute_waypoint_policy(
        self, waypoints, policy, *, native_stop=False, limits=None
    ):
        """Expose either existing first-row baseline or ONE opt-in prefix policy."""
        from go2_perception.vln.waypoints import path_errors, to_world, waypoint_array

        chunk = waypoint_array(waypoints)
        if policy == "bounded_prefix":
            return self.execute_prefix(chunk, native_stop=native_stop, limits=limits)
        if policy != "first_waypoint":
            raise ValueError("Unknown simulation waypoint policy")
        if native_stop and np.any(chunk != 0):
            raise ValueError("Native STOP contradicts nonzero chunk")
        origin = self.pose.copy()
        world = to_world(chunk, origin)
        # Preserve legacy baseline limits and atomic rejection/rollback behavior.
        legacy = (
            {"applied": True, "actual": [0, 0, 0]}
            if native_stop
            else self.advance(chunk[0])
        )
        moved = legacy["applied"] and not native_stop
        delta = self.pose - origin
        actual = (
            [origin.tolist()]
            if not moved
            else [
                (origin + fraction * delta).tolist()
                for fraction in np.linspace(0, 1, 11)
            ]
        )
        times = [0.0] if not moved else np.linspace(0, 0.25, 11).tolist()
        rejected = (
            None
            if legacy["applied"]
            else {
                "reason": legacy["reason"],
                "contacts": legacy.get("contacts", []),
                "baseline_atomic_rollback": True,
            }
        )
        # The baseline considers only row zero. Adherence is measured against it.
        return {
            "policy": policy,
            "native_stop": native_stop,
            "decision_pose": origin.tolist(),
            "native_chunk": chunk.tolist(),
            "world_chunk": world.tolist(),
            "selected_indices": [] if native_stop else [0],
            "requested_world_targets": [] if native_stop else [world[0].tolist()],
            "requested_native_first_waypoint": chunk[0].tolist(),
            "planned_world_poses": [origin.tolist()]
            if native_stop
            else [origin.tolist(), world[0].tolist()],
            "planned_endpoint": origin.tolist() if native_stop else world[0].tolist(),
            "actual_world_poses": actual,
            "actual_times_s": times,
            "actual_endpoint": self.pose.tolist(),
            "simulated_dt_s": times[-1],
            "applied": legacy["applied"],
            "constraint_rejection": rejected,
            "legacy_baseline_result": legacy,
            "path_error": path_errors(actual, origin, chunk[:1]),
            "physical_execution_authorized": False,
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
