import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
p = root / "vendor/PX4-Autopilot/Tools/simulation/gz/models/x500_base/model.sdf"
m = ET.parse(p).getroot().find("model")
links = []
total = np.zeros((3, 3))
mass = 0
for link in m.findall("link"):
    i = link.find("inertial")
    if i is None:
        continue
    mass_i = float(i.findtext("mass"))
    pos = np.array([float(v) for v in link.findtext("pose", "0 0 0 0 0 0").split()[:3]])
    v = i.find("inertia")
    mat = np.diag([float(v.findtext(k)) for k in ["ixx", "iyy", "izz"]])
    for a, b, k in [(0, 1, "ixy"), (0, 2, "ixz"), (1, 2, "iyz")]:
        mat[a, b] = mat[b, a] = float(v.findtext(k, "0"))
    translated = mat + mass_i * (np.eye(3) * np.dot(pos, pos) - np.outer(pos, pos))
    total += translated
    mass += mass_i
    links.append(
        dict(
            name=link.attrib["name"],
            mass_kg=mass_i,
            position_flu_m=pos.tolist(),
            inertia_local_kgm2=mat.tolist(),
        )
    )
report = dict(
    source=str(p),
    origin="pinned_upstream_geometry",
    total_mass_kg=mass,
    locked_rotor_inertia_about_base_origin_kgm2=total.tolist(),
    rotor_spin_inertia_core_kgm2=2e-5,
    limitation="upstream revolute rotor links are retained; locked-rotor sum is a bookkeeping reference, not a measured effective tensor. Core axial spin inertia is an independent engineering approximation.",
    body_drag_application="base_link center; CdA .08m2 each axis",
    links=links,
)
(root / "artifacts/mass-inertia-ledger.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(dict(total_mass_kg=mass, locked_inertia=total.tolist())))
