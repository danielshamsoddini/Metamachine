"""Brachiation scene extensions, isolated from the simulator-independent task.

MuJoCo is used only to compile and validate the reference morphology at startup.
Rollout and selective resets operate exclusively on Newton/Warp device buffers.
"""

import numpy as np
import warp as wp

from .newton import NewtonBackend
from .newton_brachiation_kernels import contact_summary, tether_forces
from .newton_model import NewtonModelTranslation


class BrachiationNewtonBackend(NewtonBackend):
    def __init__(self, model, *, num_worlds=1, device="cuda:0", contact_distance=0.0):
        translation = NewtonModelTranslation(model, brachiation=True)
        b = translation.builder
        shape_names = [s.rsplit("/", 1)[-1] for s in b.shape_label]
        geom_ids = [model.geom(name).id for name in shape_names]
        # Preserve MuJoCo's parent/weld filtering after splitting shoulders.
        for a, ga in enumerate(geom_ids):
            for c in range(a):
                gc = geom_ids[c]
                ba, bc = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gc])
                wa, wc = int(model.body_weldid[ba]), int(model.body_weldid[bc])
                parent_a = int(model.body_weldid[model.body_parentid[wa]])
                parent_c = int(model.body_weldid[model.body_parentid[wc]])
                compatible = (model.geom_contype[ga] & model.geom_conaffinity[gc]) or (
                    model.geom_contype[gc] & model.geom_conaffinity[ga]
                )
                if (
                    not compatible
                    or wa == wc
                    or (wa != 0 and wc != 0 and (parent_a == wc or parent_c == wa))
                ):
                    b.shape_collision_filter_pairs.append((c, a))
        super().__init__(
            model,
            mode="native",
            device=device,
            num_worlds=num_worlds,
            translation=translation,
        )
        self.contact_distance = float(contact_distance)
        self.bar_count = model.nmocap
        self.bar_bodies = [
            self.body_map[model.body(f"course_bar_{i}").id]
            for i in range(self.bar_count)
        ]
        self.bar_joints = [list(b.joint_child).index(i) for i in self.bar_bodies]
        self.joint_stride = b.joint_count
        root = model.body("torso").id
        cart = model.body("cart").id

        def descendant(body, ancestor):
            while body > 0:
                if body == ancestor:
                    return True
                body = int(model.body_parentid[body])
            return False

        classes, hands, bars = [], [], []
        for geom, name in zip(geom_ids, shape_names, strict=False):
            body = int(model.geom_bodyid[geom])
            robot, in_cart = descendant(body, root), descendant(body, cart)
            hook = name.startswith(("left_hook_", "right_hook_"))
            bar = name.startswith("course_bar_geom_")
            classes.append(
                int(robot)
                + 2 * int(in_cart)
                + 4 * int(robot and body != root)
                + 8 * int(hook)
                + 16 * int(bar)
                + 32 * int(name == "floor")
                + 64 * int(name == "torso_geom")
            )
            hands.append((0 if name.startswith("left_") else 1) if hook else -1)
            bars.append(int(name.rsplit("_", 1)[1]) if bar else -1)
        self._world = wp.array(
            np.repeat(np.arange(num_worlds), len(shape_names)),
            dtype=wp.int32,
            device=self.device,
        )
        self._class, self._hands, self._bars = [
            wp.array(np.tile(x, num_worlds), dtype=wp.int32, device=self.device)
            for x in (classes, hands, bars)
        ]
        self.loads = wp.zeros(
            (num_worlds, 2, self.bar_count), dtype=wp.float32, device=self.device
        )
        self.touching = wp.zeros((num_worlds, 2), dtype=wp.int32, device=self.device)
        self.flags = wp.zeros((num_worlds, 4), dtype=wp.int32, device=self.device)
        self.tether_length = wp.zeros(num_worlds, device=self.device)
        self.tether_tension = wp.zeros(num_worlds, device=self.device)
        self.tether_args = None
        if model.ntendon:
            sites = [
                model.site(name).id
                for name in ("cart_tether_anchor", "torso_tether_anchor")
            ]
            time, ratio = model.tendon_solref_lim[0]
            if time <= 0 or ratio <= 0:
                raise ValueError(
                    "Native tether requires positive solreflimit time and damping ratio"
                )
            mass = 1.0 / float(model.tendon_invweight0[0])
            impedance = float(model.tendon_solimp_lim[0, 1])
            self.tether_args = [
                self.body_stride,
                *[self.body_map[int(model.site_bodyid[s])] for s in sites],
                *[wp.vec3(*model.site_pos[s]) for s in sites],
                float(model.tendon_lengthspring[0, 1]),
                float(model.tendon_range[0, 1]),
                float(model.tendon_stiffness[0]),
                float(model.tendon_damping[0]),
                mass / (time * ratio * impedance) ** 2,
                2 * mass / (time * impedance),
            ]
        self.scene_manifest = {
            "solver": "Featherstone",
            "mesh_wheels": "original convex meshes",
            "bars": "per-world fixed-root transforms changed at reset",
            "shoulders": "serial hinges with massless intermediate links",
            "tether": "exact deadband spring and viscous damping; unilateral penalty limit using compiled reference effective mass and solreflimit",
            "tether_limit_matches_mujoco_solver": False,
            "unused_xml_sensors": "observations computed directly from state",
        }

    def apply_tether(self, state):
        if self.tether_args is not None:
            wp.launch(
                tether_forces,
                dim=self.num_worlds,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    self.newton_model.body_com,
                    *self.tether_args,
                ],
                outputs=[state.body_f, self.tether_length, self.tether_tension],
                device=self.device,
            )

    def _integrate(
        self, state_in, state_out, control, contacts, solver=None, pipeline=None
    ):
        state_in.clear_forces()
        self.apply_tether(state_in)
        (pipeline or self.collision_pipeline).collide(state_in, contacts)
        (solver or self.solver).step(state_in, state_out, control, contacts, self.dt)
        # Contacts/loads represent the state at which forces were applied, just
        # as MJX.step's constraint diagnostics do. Do not recompute at state_out.
        self.summarize_contacts(state_in, contacts)

    def summarize_contacts(self, state=None, contacts=None):
        state = self.state if state is None else state
        c = self.contacts if contacts is None else contacts
        m = self.newton_model
        for value in (self.loads, self.touching, self.flags):
            value.zero_()
        wp.launch(
            contact_summary,
            dim=c.rigid_contact_max,
            inputs=[
                state.body_q,
                state.body_qd,
                m.body_com,
                m.shape_body,
                self._world,
                self._class,
                self._hands,
                self._bars,
                m.shape_material_ke,
                m.shape_material_kd,
                c.rigid_contact_count,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
                c.rigid_contact_stiffness,
                c.rigid_contact_damping,
                self.contact_distance,
            ],
            outputs=[self.loads, self.touching, self.flags],
            device=self.device,
        )

    def prepare_torch_scene(self, reference_qpos):
        """Bind stable device views and reset coordinates once, before rollout."""
        import torch

        n = self.num_worlds
        device = str(self.device)
        self._tensor_states = [
            (
                wp.to_torch(b.state.joint_q).view(n, -1),
                wp.to_torch(b.state.joint_qd).view(n, -1),
                wp.to_torch(b.state.body_q).view(n, -1, 7),
                wp.to_torch(b.state.body_qd).view(n, -1, 6),
            )
            for b in self._buffers
        ]
        self.local_com = wp.to_torch(self.newton_model.body_com).view(
            n, self.body_stride, 3
        )
        self.motor_torque = wp.to_torch(self.actuator_torque).view(n, self.num_motors)
        self._tensor_contact_count = wp.to_torch(self.contacts.rigid_contact_count)
        self._tensor_bar_joints = torch.as_tensor(
            np.asarray(self.bar_joints)[None]
            + np.arange(n)[:, None] * self.joint_stride,
            dtype=torch.long,
            device=device,
        )
        self._tensor_joint_transforms = wp.to_torch(self.newton_model.joint_X_p)
        self._tensor_initial_q = wp.to_torch(self.initial_q).view(n, -1)
        self._root_reset_q = int(
            self._qstarts[self.joint_map[self.model.joint("root").id]]
        )
        q, _ = self.translation._to_newton(reference_qpos, np.zeros(self.model.nv))
        self._tensor_reset_q = torch.as_tensor(q, device=device)
        self._tensor_reset_mask = torch.ones(n, dtype=torch.int32, device=device)
        self._scene_reset_mask = wp.from_torch(self._tensor_reset_mask)

    @property
    def tensor_state(self):
        """Joint coordinates/velocities and world body poses/COM velocities."""
        return self._tensor_states[self._state_index]

    def contact_capacity_reached(self):
        return self._tensor_contact_count[0] >= self.contacts.rigid_contact_max

    def reset_scene(self, mask, root_translation, bar_positions, bar_quaternions):
        """Reset selected worlds and bar transforms using only device inputs."""
        import torch

        selected = mask.bool()
        transforms = torch.cat((bar_positions, bar_quaternions), -1)
        self._tensor_joint_transforms[self._tensor_bar_joints] = transforms
        initial = self._tensor_reset_q[None].expand(self.num_worlds, -1).clone()
        initial[:, self._root_reset_q : self._root_reset_q + 3] += root_translation
        self._tensor_initial_q.copy_(
            torch.where(selected[:, None], initial, self._tensor_initial_q)
        )
        self._tensor_reset_mask.copy_(mask)
        self.reset_worlds(self._scene_reset_mask)
        # Match reset's forward contact diagnostics without advancing time.
        self.collision_pipeline.collide(self.state, self.contacts)
        self.summarize_contacts()
