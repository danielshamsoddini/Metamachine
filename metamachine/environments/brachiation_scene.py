"""Procedural training scenes derived from the user's unchanged robot XML."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def build_scene(cfg, source_path):
    bars = cfg.bars
    from .brachiation_courses import validate_course, is_spatial
    validate_course(cfg)
    if int(bars.count) < 2:
        raise ValueError("At least two bars are required")
    for key in ("spacing", "height", "height_offset", "lateral_offset"):
        lo, hi = bars[key]
        if lo > hi:
            raise ValueError(f"bars.{key}: minimum exceeds maximum")
    if bars.spacing[0] <= 2 * bars.radius:
        raise ValueError("Bar spacing must exceed the diameter")
    if cfg.task.start not in ("hanging", "hanging_in_cart"):
        raise ValueError("task.start must be hanging or hanging_in_cart")
    root = ET.parse(Path(source_path)).getroot()
    world = root.find("worldbody")
    cart = world.find("body[@name='cart']")
    if cfg.task.remove_cart and cart is not None:
        world.remove(cart)
    elif not cfg.task.remove_cart and cart is None:
        raise ValueError("task.remove_cart is false but the source XML has no cart")
    if cart is not None and not cfg.task.remove_cart:
        prepare_cart_wheels(root, cart, cfg)
    scale_robot(root, cfg)
    if cart is not None and not cfg.task.remove_cart:
        prepare_cart_cradle(root, cart, cfg)
    in_cart = cfg.task.start == "hanging_in_cart"
    if in_cart and cfg.task.remove_cart:
        raise ValueError("hanging_in_cart requires remove_cart: false")
    reference = str(cfg.bars.get("height_reference", "world"))
    if reference not in ("world", "cart_start") or (in_cart != (reference == "cart_start")):
        raise ValueError("hanging_in_cart requires bars.height_reference: cart_start; hanging requires world")
    first_height = cart_start_bar_height(root, cfg) if in_cart else 0.0
    for geom in list(world.findall("geom")):
        if (geom.get("name") or "").startswith("bar"):
            world.remove(geom)
    option = root.find("option")
    option.set("integrator", str(cfg.simulation.integrator))
    option.set("timestep", str(cfg.simulation.timestep))
    option.set("iterations", str(cfg.simulation.solver_iterations))
    option.set("ls_iterations", str(cfg.simulation.line_search_iterations))
    poses = None
    if is_spatial(cfg):
        import jax
        from .brachiation_courses import sample_spatial
        centers, quats, _ = sample_spatial(jax.random.PRNGKey(int(bars.spatial.preview_seed)), cfg, first_height)
        poses = np.asarray(centers), np.asarray(quats)
    for i in range(int(bars.count)):
        # Mocap poses are dynamic MJX data: different courses share one model.
        body = ET.SubElement(world, "body", name=f"course_bar_{i}", mocap="true",
                             pos=f"{i * np.mean(bars.spacing)} 0 {first_height if in_cart and i == 0 else first_height + np.mean(bars.height)}")
        geom = ET.SubElement(body, "geom", name=f"course_bar_geom_{i}",
                      type=str(bars.geometry), size=f"{bars.radius} {bars.half_length}",
                      quat="0.7071067811865476 0.7071067811865476 0 0",
                      material="bar_mat")
        if poses is not None:
            body.set("pos", " ".join(map(str, poses[0][i])))
            body.set("quat", " ".join(map(str, poses[1][i])))
            # Crossings are fixed rigid geometry; only robot/cart contacts are solved.
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
    # Make the generated XML itself open in the nominal hanging pose too.
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    qpos = hanging_qpos(model, cfg, data.mocap_pos[0])
    adr = model.joint("root").qposadr[0]
    world.find("body[@name='torso']").set("pos", " ".join(str(v) for v in qpos[adr:adr + 3]))
    return ET.tostring(root, encoding="unicode")


def prepare_cart_wheels(root, cart, cfg):
    """Retain cylinder visuals/inertia; use a polygonal tire for MJX collision."""
    settings = cfg.get("cart", {})
    if not {"wheel_collision", "wheel_segments"} <= set(settings) or set(settings) - {"wheel_collision", "wheel_segments", "cradle", "safety_tether"} or settings["wheel_collision"] != "convex_mesh":
        raise ValueError("Retaining the cart requires cart.wheel_collision: convex_mesh and wheel_segments")
    n = settings["wheel_segments"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 8 or n > 64 or n % 4:
        raise ValueError("cart.wheel_segments must be a multiple of 4 from 8 to 64")
    asset = root.find("asset")
    for wheel in cart.findall("body"):
        tire = wheel.find("geom")
        radius, half_width = np.fromstring(tire.get("size"), sep=" ")
        angles = np.arange(n) * (2 * np.pi / n)
        vertices = [(radius*np.cos(a), radius*np.sin(a), z) for z in (-half_width, half_width) for a in angles]
        name = tire.get("name") + "_collision"
        ET.SubElement(asset, "mesh", name=name, vertex=" ".join(f"{v:.12g}" for xyz in vertices for v in xyz))
        tire.set("contype", "0")
        tire.set("conaffinity", "0")
        ET.SubElement(wheel, "geom", name=name, type="mesh", mesh=name,
                      quat=tire.get("quat"), mass="0", rgba="0 0 0 0", group="3")


def cart_start_bar_height(root, cfg):
    """First bar axis giving simultaneous torso-on-deck and hook alignment."""
    cart = root.find("worldbody/body[@name='cart']")
    torso = root.find("worldbody/body[@name='torso']")
    deck = cart.find("geom[@name='cart_deck']")
    geom = torso.find("geom[@name='torso_geom']")
    deck_z = np.fromstring(cart.get("pos"), sep=" ")[2] + np.fromstring(deck.get("pos"), sep=" ")[2] + np.fromstring(deck.get("size"), sep=" ")[2]
    torso_z = deck_z + np.fromstring(geom.get("size"), sep=" ")[2]
    hand_offset = sum(np.fromstring(torso.find(f".//body[@name='{name}']").get("pos"), sep=" ")[2]
                      for name in ("left_upper_arm", "left_forearm", "left_hand"))
    return torso_z + hand_offset + float(cfg.task.initial_bar_to_hand_z)


def scale_robot(root, cfg):
    """Scale torso/arm geometry and masses while preserving entire hand subtrees.

    Body offsets to the hands scale; hook/palm geometry and masses do not.
    MuJoCo derives rigid-body inertias from the resized, explicitly massed geoms.
    Joint axes, ranges, damping and rotor armature are retained from the source.
    """
    settings = cfg.get("robot", None)
    if settings is None:
        return  # Archived configs retain the original source dimensions.
    expected = {"geometry_scale", "mass_scale", "preserve_hands"}
    if set(settings) != expected or settings.preserve_hands is not True:
        raise ValueError("robot requires geometry_scale, mass_scale, preserve_hands: true")
    scale, mass = float(settings.geometry_scale), float(settings.mass_scale)
    if not np.isfinite([scale, mass]).all() or scale <= 0 or mass <= 0:
        raise ValueError("robot scales must be positive and finite")
    torso = root.find("worldbody/body[@name='torso']")
    def multiply(element, attribute, factor):
        if element.get(attribute) is not None:
            values = np.fromstring(element.get(attribute), sep=" ") * factor
            element.set(attribute, " ".join(f"{v:.12g}" for v in values))
    def visit(body, is_root=False):
        if not is_root:
            multiply(body, "pos", scale)
        if body.get("name") in ("left_hand", "right_hand"):
            return
        if body.find("inertial") is not None:
            raise ValueError("robot scaling expects geom-derived inertias")
        for geom in body.findall("geom"):
            if geom.get("mass") is None:
                raise ValueError("robot scaling expects explicit geom masses")
            for attr in ("pos", "size", "fromto"):
                multiply(geom, attr, scale)
            multiply(geom, "mass", mass)
        for site in body.findall("site"):
            for attr in ("pos", "size"):
                multiply(site, attr, scale)
        for joint in body.findall("joint"):
            multiply(joint, "pos", scale)
        for child in body.findall("body"):
            visit(child)
    visit(torso, True)
    # Keep each forearm endpoint at the unchanged palm's center.
    for side in ("left", "right"):
        hand = torso.find(f".//body[@name='{side}_hand']")
        palm = hand.find(f"geom[@name='{side}_hand_palm']")
        geom = torso.find(f".//geom[@name='{side}_forearm_geom']")
        endpoint = np.fromstring(hand.get("pos"), sep=" ") + np.fromstring(palm.get("pos"), sep=" ")
        points = np.fromstring(geom.get("fromto"), sep=" ")
        points[3:] = endpoint
        geom.set("fromto", " ".join(f"{v:.12g}" for v in points))


def joint_indices(model):
    joints = model.actuator_trnid[:, 0]
    return model.jnt_qposadr[joints], model.jnt_dofadr[joints]


def hanging_qpos(model, cfg, bar_position):
    """Place both unrotated hook crowns just above the first bar, without welds."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hand = model.body("left_hand").id
    root = model.joint("root").qposadr[0]
    offset = data.xpos[hand] - data.qpos[root:root + 3]
    qpos = model.qpos0.copy()
    qpos[root:root + 3] = np.asarray(bar_position) - np.array(
        [offset[0], 0.0, offset[2] + float(cfg.task.initial_bar_to_hand_z)])
    return qpos


def prepare_cart_cradle(root, cart, cfg):
    """Four passive lower-torso guides, with clearance and an open top."""
    settings = cfg.cart.get("cradle")
    if settings is None:
        return
    if set(settings) != {"clearance", "height", "thickness", "mass_per_pad"}:
        raise ValueError("cart.cradle requires clearance, height, thickness, mass_per_pad")
    values = np.array([settings[k] for k in ("clearance", "height", "thickness", "mass_per_pad")], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("cart.cradle dimensions and mass must be positive and finite")
    clearance, height, thickness, mass = values
    torso = root.find("worldbody/body[@name='torso']/geom[@name='torso_geom']")
    half = np.fromstring(torso.get("size"), sep=" ")
    deck = cart.find("geom[@name='cart_deck']")
    deck_half = np.fromstring(deck.get("size"), sep=" ")
    deck_pos = np.fromstring(deck.get("pos"), sep=" ")
    inner = half[:2] + clearance
    if np.any(inner + thickness >= deck_half[:2]) or height >= 2 * half[2]:
        raise ValueError("Cradle must fit the deck and remain below the torso top")
    for axis, labels in ((0, ("back", "front")), (1, ("right", "left"))):
        for sign, label in zip((-1, 1), labels):
            pos = deck_pos.copy()
            pos[axis] += sign * (inner[axis] + thickness / 2)
            pos[2] += deck_half[2] + height / 2
            size = np.array([inner[0] + thickness, inner[1] + thickness, height / 2])
            size[axis] = thickness / 2
            ET.SubElement(cart, "geom", name="cart_cradle_" + label, type="box",
                          pos=" ".join(map(str, pos)), size=" ".join(map(str, size)),
                          mass=str(mass), rgba="0.3 0.5 0.5 1", friction="0.6 0.01 0.001",
                          solref="0.02 1")

    tether = cfg.cart.get("safety_tether")
    if tether is not None:
        if set(tether) != {"slack_length", "max_length", "stiffness", "damping"}:
            raise ValueError("cart.safety_tether requires slack_length, max_length, stiffness, damping")
        vals = np.array([tether[k] for k in ("slack_length", "max_length", "stiffness", "damping")], dtype=float)
        if not np.isfinite(vals).all() or np.any(vals <= 0) or vals[0] >= vals[1]:
            raise ValueError("Invalid safety tether lengths or compliance")
        # Lower-torso attachment moves freely inside the slack radius.
        ET.SubElement(cart, "site", name="cart_tether_anchor", pos=f"0 0 {deck_pos[2] + deck_half[2] + .01}", size=".005", rgba="0 0 0 0")
        robot = root.find("worldbody/body[@name='torso']")
        ET.SubElement(robot, "site", name="torso_tether_anchor", pos=f"0 0 {-half[2] + .04}", size=".005", rgba="0 0 0 0")
        tendons = root.find("tendon")
        if tendons is None:
            tendons = ET.SubElement(root, "tendon")
        spatial = ET.SubElement(tendons, "spatial", name="cart_safety_tether", limited="true",
                                range=f"0 {tether.max_length}", springlength=f"0 {tether.slack_length}",
                                stiffness=str(tether.stiffness), damping=str(tether.damping),
                                solreflimit="0.04 1", width=".004", rgba="0.8 0.6 0.2 1")
        ET.SubElement(spatial, "site", site="cart_tether_anchor")
        ET.SubElement(spatial, "site", site="torso_tether_anchor")
