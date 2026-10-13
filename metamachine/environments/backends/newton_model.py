"""Startup-only MJCF translation; no calls from the device rollout loop."""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .state import BackendCapabilityError


def _rotation(wxyz):
    return Rotation.from_quat(np.roll(wxyz, -1)).as_matrix()


class NewtonModelTranslation:
    def __init__(self, model, *, brachiation=False):
        import newton
        import warp as wp

        self.model = model
        self.brachiation = brachiation
        self._validate_model()
        # Export the already compiled morphology: defaults, density-derived
        # masses and principal inertias must not be independently reinterpreted.
        with tempfile.TemporaryDirectory(prefix="metamachine-newton-") as tmp:
            path = Path(tmp) / "compiled.xml"
            mujoco.mj_saveLastXML(str(path), model)
            if brachiation:
                # Mocap bars become fixed roots whose joint transforms are updated
                # only by the native reset path. The two-site tether is applied as
                # external body wrenches by BrachiationNewtonBackend.
                import xml.etree.ElementTree as ET

                tree = ET.parse(path)
                for body in tree.findall(".//body"):
                    body.attrib.pop("mocap", None)
                for tendon in tree.findall("tendon") + tree.findall("sensor"):
                    tree.getroot().remove(tendon)
                # MuJoCo applies multiple hinges in declaration order. Split
                # into massless intermediate links, leaving all geometry and
                # compiled inertia on the original body at the end of the chain.
                for parent in list(tree.iter()):
                    for body in list(parent.findall("body")):
                        joints = body.findall("joint")
                        if len(joints) > 1:
                            index = list(parent).index(body)
                            parent.remove(body)
                            outer = ET.Element(
                                "body",
                                name=body.get("name") + "_joint_frame",
                                pos=body.attrib.pop("pos", "0 0 0"),
                                quat=body.attrib.pop("quat", "1 0 0 0"),
                            )
                            parent.insert(index, outer)
                            cursor = outer
                            for j, joint in enumerate(joints[:-1]):
                                body.remove(joint)
                                cursor.append(joint)
                                if j < len(joints) - 2:
                                    child = ET.SubElement(
                                        cursor,
                                        "body",
                                        name=body.get("name") + f"_joint_frame_{j}",
                                    )
                                    cursor = child
                            cursor.append(body)
                tree.write(path)
            builder = newton.ModelBuilder()
            builder.add_mjcf(
                str(path),
                collapse_fixed_joints=False,
                ignore_inertial_definitions=False,
                parse_sites=False,
            )

        self.body_map = self._name_map(builder.body_label, model.nbody, "body", start=1)
        self.joint_map = self._name_map(builder.joint_label, model.njnt, "joint")
        # Copy compiled inertial properties at full precision (XML export is
        # rounded), and rotate principal inertia into each body's local frame.
        for mid, nid in self.body_map.items():
            mass = float(model.body_mass[mid])
            rot = _rotation(model.body_iquat[mid])
            inertia = rot @ np.diag(model.body_inertia[mid]) @ rot.T
            builder.body_mass[nid] = mass
            builder.body_inv_mass[nid] = 1.0 / mass if mass else 0.0
            builder.body_com[nid] = wp.vec3(*model.body_ipos[mid])
            builder.body_inertia[nid] = wp.mat33(inertia)
            builder.body_inv_inertia[nid] = wp.mat33(
                np.linalg.inv(inertia) if mass else np.zeros((3, 3))
            )

        self.builder = builder
        self._qstarts = np.asarray(builder.joint_q_start)
        self._vstarts = np.asarray(builder.joint_qd_start)

    def _name_map(self, labels, count, kind, start=0):
        short = [label.rsplit("/", 1)[-1] for label in labels]
        result = {}
        for i in range(start, count):
            name = getattr(self.model, kind)(i).name
            if not name or short.count(name) != 1:
                raise ValueError(
                    f"Newton requires unique named {kind}s; cannot map {name!r}"
                )
            result[i] = short.index(name)
        return result

    def _validate_model(self):
        m = self.model

        def require(ok, message):
            if not ok:
                raise BackendCapabilityError("Newton initial support: " + message)

        require(
            m.neq == m.nflex == m.nplugin == 0
            and (self.brachiation or m.ntendon == m.nmocap == 0),
            "equality constraints, tendons, mocap, flex bodies and plugins are unsupported",
        )
        if self.brachiation:
            require(
                all(
                    m.body(i).name.startswith("course_bar_")
                    for i in range(m.nbody)
                    if m.body_mocapid[i] >= 0
                ),
                "only course bars may be mocap bodies",
            )
            require(m.ntendon <= 1, "only the cart safety tether is supported")
            if m.ntendon:
                require(
                    m.tendon(0).name == "cart_safety_tether"
                    and m.tendon_num[0] == 2
                    and np.all(m.wrap_type == mujoco.mjtWrap.mjWRAP_SITE),
                    "tether must be a two-site spatial tendon",
                )
                require(
                    bool(m.tendon_limited[0])
                    and m.tendon_range[0, 0] == 0
                    and m.tendon_lengthspring[0, 0] == 0
                    and m.tendon_frictionloss[0] == 0,
                    "tether requires an upper length limit, zero lower spring length and no frictionloss",
                )
        require(np.all(m.body_gravcomp == 0), "gravity compensation is unsupported")
        require(
            self.brachiation or m.nsensor == 0,
            "sensors are unsupported; use state observations",
        )
        require(
            m.nhfield == 0 and (self.brachiation or m.nmesh == 0),
            "only primitive collision geometry is supported",
        )
        require(
            np.all(
                np.isin(
                    m.geom_type,
                    [0, 2, 3, 4, 5, 6, 7] if self.brachiation else [0, 2, 3, 4, 5, 6],
                )
            ),
            "unsupported collision geometry",
        )
        require(
            self.brachiation or np.all(m.body_jntnum <= 1),
            "use at most one joint per body",
        )
        require(
            np.all(np.isin(m.jnt_type, [0, 2, 3])),
            "only free, slide and hinge joints are supported",
        )
        require(
            np.all(m.jnt_stiffness == 0) and np.all(m.dof_frictionloss == 0),
            "joint springs and frictionloss are unsupported",
        )
        require(
            np.all(m.jnt_actfrclimited == 0),
            "joint actuator force limits are unsupported",
        )
        for j, kind in enumerate(m.jnt_type):
            if kind != mujoco.mjtJoint.mjJNT_FREE:
                require(
                    m.qpos0[m.jnt_qposadr[j]] == 0,
                    "nonzero joint reference positions are unsupported",
                )
            else:
                require(
                    m.body_parentid[m.jnt_bodyid[j]] == 0,
                    "free joints must connect to world",
                )
                v = m.jnt_dofadr[j]
                require(
                    np.all(m.dof_damping[v : v + 6] == 0)
                    and np.all(m.dof_armature[v : v + 6] == 0),
                    "free-joint damping and armature are unsupported",
                )
        require(
            np.all(m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            and np.all(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
            and np.all(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
            and np.all(m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE),
            "only stateless joint motors with fixed gain and no bias are supported",
        )
        require(
            np.all(m.actuator_gear[:, 1:] == 0),
            "only scalar motor gear ratios are supported",
        )
        require(
            all(m.jnt_type[j] in (2, 3) for j in m.actuator_trnid[:, 0]),
            "motors must actuate slide or hinge joints",
        )
        require(
            m.opt.density == 0 and m.opt.viscosity == 0 and np.all(m.opt.wind == 0),
            "fluid forces are unsupported",
        )
        require(
            m.opt.disableflags == 0 and m.opt.enableflags == 0,
            "custom MuJoCo option flags are unsupported",
        )

    def _to_newton(self, qpos, qvel):
        q = np.asarray(self.builder.joint_q, dtype=np.float32).copy()
        v = np.zeros(self.builder.joint_dof_count, dtype=np.float32)
        for mid, nid in self.joint_map.items():
            mq, mv = self.model.jnt_qposadr[mid], self.model.jnt_dofadr[mid]
            nq, nv = self._qstarts[nid], self._vstarts[nid]
            if self.model.jnt_type[mid] == mujoco.mjtJoint.mjJNT_FREE:
                q[nq : nq + 3] = qpos[mq : mq + 3]
                quat = np.asarray(qpos[mq + 3 : mq + 7], dtype=float)
                norm = np.linalg.norm(quat)
                if norm < 1e-12:
                    raise ValueError("Free-joint quaternion must be nonzero")
                q[nq + 3 : nq + 7] = np.roll(quat / norm, -1)
                rot = _rotation(qpos[mq + 3 : mq + 7])
                omega = rot @ qvel[mv + 3 : mv + 6]
                offset = rot @ self.model.body_ipos[self.model.jnt_bodyid[mid]]
                # Newton 1.6: free qd is world COM linear + world angular.
                # MuJoCo: world body-origin linear + body-local angular.
                v[nv : nv + 3] = qvel[mv : mv + 3] + np.cross(omega, offset)
                v[nv + 3 : nv + 6] = omega
            else:
                q[nq], v[nv] = qpos[mq], qvel[mv]
        return q, v
