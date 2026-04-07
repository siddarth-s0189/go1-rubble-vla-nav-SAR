# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Stage5 SAR terrain generation utilities."""

import pathlib

import numpy as np


def generate_stage5_xml(seed: int = 42, n_rubble: int = 120) -> str:
  """Generate and write Stage 5 XML with a realistic debris field."""
  rng = np.random.RandomState(seed)

  n_pebbles = int(n_rubble * 0.60)
  n_debris = int(n_rubble * 0.30)
  n_obstacles = n_rubble - n_pebbles - n_debris

  specs: list[tuple[float, str]] = []
  for _ in range(n_pebbles):
    sz = float(rng.uniform(0.03, 0.06))
    geom_type = "sphere" if rng.random() < 0.30 else "box"
    specs.append((sz, geom_type))
  for _ in range(n_debris):
    sz = float(rng.uniform(0.07, 0.12))
    geom_type = "sphere" if rng.random() < 0.30 else "box"
    specs.append((sz, geom_type))
  for _ in range(n_obstacles):
    sz = float(rng.uniform(0.13, 0.15))
    geom_type = "sphere" if rng.random() < 0.30 else "box"
    specs.append((sz, geom_type))

  specs.sort(key=lambda s: -s[0])
  fov_x_min, fov_x_max = 0.1, 4.9
  fov_y_min, fov_y_max = -1.45, 1.45
  safety_radius = 0.6
  min_gap = 0.05
  max_place_attempts = 200

  placed: list[tuple[float, float, float, str]] = []
  for size, geom_type in specs:
    for _ in range(max_place_attempts):
      x = float(rng.uniform(fov_x_min, fov_x_max))
      y = float(rng.uniform(fov_y_min, fov_y_max))
      if x * x + y * y < safety_radius * safety_radius:
        continue
      overlap = False
      for (ex, ey, es, _) in placed:
        dist_sq = (x - ex) ** 2 + (y - ey) ** 2
        min_dist = size + es + min_gap
        if dist_sq < min_dist * min_dist:
          overlap = True
          break
      if overlap:
        continue
      placed.append((x, y, size, geom_type))
      break

  rubble_lines: list[str] = []
  for idx, (x, y, size, geom_type) in enumerate(placed, start=1):
    z = size
    rubble_lines.append(
        f'    <body name="rubble{idx}" pos="{x:.3f} {y:.3f} {z:.3f}">'
        f'<geom name="rubble{idx}_geom" type="{geom_type}" '
        f'size="{size:.3f} {size:.3f} {size:.3f}" '
        f'material="debris_matte" '
        f'condim="3" margin="0.02" gap="0.01" '
        f'contype="1" conaffinity="1" priority="1" group="0"/></body>'
    )
  rubble_xml = "\n".join(rubble_lines)

  xml_content = f"""<mujoco model="go1 SAR stage 5 - Debris Field">
  <include file="go1_mjx_feetonly_sar.xml"/>

  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" file="assets/rocky_texture.png"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance=".8"/>
    <material name="floor_dark" rgba="0.1 0.1 0.1 1"/>
    <material name="concrete" rgba="0.3 0.3 0.3 1"/>
    <material name="debris_matte" rgba="0.25 0.25 0.25 1" specular="0" shininess="0"/>
    <material name="rust_metal" rgba="0.5 0.1 0.05 1" specular="0.8"/>
    <material name="charcoal_wall" rgba="0.15 0.15 0.15 1"/>
    <hfield name="hfield" file="assets/hfield.png" size="10 10 .05 1.0"/>
  </asset>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" type="directional" castshadow="true"/>
    <geom name="floor" type="hfield" hfield="hfield" material="groundplane" contype="1" conaffinity="0" priority="1"
      friction="1.0"/>
    <site name="goal" pos="5 0 0" rgba="1 0 0 1" size="0.1"/>
    <camera name="birds_eye" pos="2.5 0 3.5" xyaxes="1 0 0 0 1 0" fovy="60"/>
{rubble_xml}
  </worldbody>

  <include file="sensor_feet.xml"/>

  <keyframe>
    <key name="home" qpos="0 0 0.35 1 0 0 0 0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"
      ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
  </keyframe>
</mujoco>
"""
  xml_path = pathlib.Path(__file__).parent / "xmls" / "scene_mjx_feetonly_sar_stage5.xml"
  xml_path.write_text(xml_content)
  print(
      f"[Stage5] Generated {len(placed)} rubble pieces "
      f"(seed={seed}) -> {xml_path}"
  )
  return str(xml_path)
