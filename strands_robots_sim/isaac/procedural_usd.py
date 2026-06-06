"""Serialize a :class:`ProceduralRobot` dataclass into a USD articulation.

This is the "option 2" approach to wiring the procedural-builder
``add_robot("so100" | "panda" | "unitree_g1")`` path: instead of a
bespoke in-stage articulation-construction engine, emit a USD file
from the ``ProceduralRobot`` description (bodies + joints) and route
it through the already-validated ``IsaacSimulation._load_usd_robot``
loader (which references the USD onto the live stage, wraps it in an
``omni.isaac.core.articulations.Articulation``, and initialises it).

This reuses the USD-load code path rather than duplicating
articulation construction, so the procedural builders inherit the
same GPU-validated behaviour as USD-/URDF-loaded robots: joints
become observable via ``get_observation`` and actuatable via
``send_action``.

The serializer authors:

* A root ``Xform`` carrying ``UsdPhysics.ArticulationRootAPI`` (the
  prim the ``Articulation`` wrapper binds to).
* One rigid-body ``Xform`` per :class:`BodyDef` -- a shape geom
  (Cube / Sphere / Capsule / Cylinder) with ``RigidBodyAPI`` +
  ``CollisionAPI`` + ``MassAPI``, posed at the body's position /
  orientation.
* One physics joint per :class:`JointDef` -- ``RevoluteJoint`` /
  ``PrismaticJoint`` / ``FixedJoint`` connecting the parent / child
  bodies, with the axis mapped to the nearest principal token, joint
  limits, and a ``DriveAPI`` carrying the stiffness / damping.

``pxr`` is imported lazily inside the function so this module imports
cleanly without USD installed; the call site only runs after
``create_world`` has booted ``SimulationApp`` (which provides ``pxr``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands_robots_sim.isaac.procedural import BodyDef, ProceduralRobot


def _axis_to_token(axis: tuple[float, float, float]) -> str:
    """Map an arbitrary axis vector to the nearest USD principal token.

    UsdPhysics revolute / prismatic joints take a ``physics:axis``
    token of ``"X"`` / ``"Y"`` / ``"Z"`` rather than an arbitrary
    vector. The procedural robots use principal-aligned axes (e.g.
    ``(0, 0, 1)``, ``(0, -1, 0)``), so picking the dominant component
    is exact for them; off-axis vectors snap to the nearest principal
    direction (sign is carried by the joint frames / limits, so the
    token is unsigned).
    """
    ax, ay, az = abs(axis[0]), abs(axis[1]), abs(axis[2])
    if ax >= ay and ax >= az:
        return "X"
    if ay >= az:
        return "Y"
    return "Z"


def _author_shape(stage: Any, prim_path: str, body: "BodyDef") -> Any:
    """Define the geom prim for a body's shape; return the UsdGeom prim.

    Cube uses a unit ``size`` + ``xformOp:scale`` so non-uniform box
    dimensions (the common ``shape_size=(sx, sy, sz)`` case) author
    correctly. Sphere / Capsule / Cylinder use their native radius /
    height attributes.
    """
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    size = body.shape_size
    if body.shape == "box":
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)
        sx = size[0] if len(size) >= 1 else 0.05
        sy = size[1] if len(size) >= 2 else sx
        sz = size[2] if len(size) >= 3 else sx
        cube.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
        return cube
    if body.shape == "sphere":
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.CreateRadiusAttr(float(size[0]) if size else 0.05)
        return sphere
    if body.shape == "capsule":
        cap = UsdGeom.Capsule.Define(stage, prim_path)
        cap.CreateRadiusAttr(float(size[0]) if len(size) >= 1 else 0.05)
        cap.CreateHeightAttr(float(size[1]) if len(size) >= 2 else 0.10)
        cap.CreateAxisAttr("Z")
        return cap
    if body.shape == "cylinder":
        cyl = UsdGeom.Cylinder.Define(stage, prim_path)
        cyl.CreateRadiusAttr(float(size[0]) if len(size) >= 1 else 0.05)
        cyl.CreateHeightAttr(float(size[1]) if len(size) >= 2 else 0.10)
        cyl.CreateAxisAttr("Z")
        return cyl
    # Unknown shape: fall back to a small cube so the body still has
    # collision geometry rather than silently producing a massless,
    # collision-less prim that breaks articulation init.
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(0.05)
    return cube


def procedural_robot_to_usd(robot: "ProceduralRobot", usd_path: str) -> str:
    """Author a USD articulation file from a :class:`ProceduralRobot`.

    Parameters
    ----------
    robot : ProceduralRobot
        The dataclass description (bodies + joints) produced by
        ``strands_robots_sim.isaac.procedural.get_procedural_robot``.
    usd_path : str
        Destination ``.usd`` file path on disk. Overwritten if it
        already exists.

    Returns
    -------
    str
        The prim path of the articulation root inside the authored USD
        (``/<robot.name>``). The caller references this onto the live
        stage via ``add_reference_to_stage`` + wraps it in an
        ``Articulation``.
    """
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

    # CreateNew clobbers an existing file; callers key usd_path on the
    # robot name so re-adding the same procedural robot reuses the path.
    stage = Usd.Stage.CreateNew(usd_path) if not _exists(usd_path) else Usd.Stage.Open(usd_path)
    if stage is None:
        stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = f"/{robot.name}"
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root_xform.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    stage.SetDefaultPrim(root_prim)

    # --- bodies -------------------------------------------------------
    body_paths: list[str] = []
    for body in robot.bodies:
        bpath = f"{root_path}/{body.name}"
        geom = _author_shape(stage, bpath, body)
        prim = geom.GetPrim()

        # Pose: translate + orient (quaternion w,x,y,z). Author on the
        # geom's Xformable. The scale op (box) was already added in
        # _author_shape; add translate/orient as additional ops.
        xformable = UsdGeom.Xformable(prim)
        pos = body.position
        xformable.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        quat = body.orientation  # (w, x, y, z)
        xformable.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))

        # Physics: rigid body + collision + mass.
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(body.mass))

        body_paths.append(bpath)

    # --- joints -------------------------------------------------------
    for joint in robot.joints:
        jpath = f"{root_path}/{joint.name}"
        parent_path = body_paths[joint.parent_body]
        child_path = body_paths[joint.child_body]
        parent_body = robot.bodies[joint.parent_body]
        child_body = robot.bodies[joint.child_body]

        jtype = joint.joint_type
        if jtype == "fixed":
            j = UsdPhysics.FixedJoint.Define(stage, jpath)
        elif jtype == "prismatic":
            j = UsdPhysics.PrismaticJoint.Define(stage, jpath)
        else:  # revolute (default)
            j = UsdPhysics.RevoluteJoint.Define(stage, jpath)

        j.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])

        # Local anchor frames. localPos0 is the joint origin in the
        # parent's frame (child_pos - parent_pos); localPos1 is the
        # origin in the child's frame (0). This keeps the assembled
        # tree consistent with the bodies' authored world positions.
        rel = (
            float(child_body.position[0]) - float(parent_body.position[0]),
            float(child_body.position[1]) - float(parent_body.position[1]),
            float(child_body.position[2]) - float(parent_body.position[2]),
        )
        j.CreateLocalPos0Attr(Gf.Vec3f(*rel))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

        if jtype in ("revolute", "prismatic"):
            token = _axis_to_token(joint.axis)
            j.CreateAxisAttr(token)
            # Revolute limits are DEGREES in UsdPhysics; JointDef stores
            # radians. Prismatic limits are in stage units (meters) and
            # pass through unchanged.
            if jtype == "revolute":
                j.CreateLowerLimitAttr(math.degrees(joint.limit_lower))
                j.CreateUpperLimitAttr(math.degrees(joint.limit_upper))
            else:
                j.CreateLowerLimitAttr(float(joint.limit_lower))
                j.CreateUpperLimitAttr(float(joint.limit_upper))

            # Drive: carries the stiffness / damping so the joint can
            # hold position targets (what send_action sets). Drive type
            # is "angular" for revolute, "linear" for prismatic.
            drive_token = "angular" if jtype == "revolute" else "linear"
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), drive_token)
            drive.CreateStiffnessAttr(float(joint.stiffness))
            drive.CreateDampingAttr(float(joint.damping))
            drive.CreateTargetPositionAttr(0.0)

            # Armature (rotor inertia) improves articulation solver
            # stability; author via the PhysxSchema joint API when
            # available, ignore if the schema isn't present.
            try:
                physx_joint = PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
                physx_joint.CreateArmatureAttr(float(joint.armature))
            except Exception:  # noqa: BLE001 - armature is best-effort
                pass

    stage.GetRootLayer().Save()
    return root_path


def _exists(path: str) -> bool:
    import os

    return os.path.isfile(path)
