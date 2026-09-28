#!/bin/bash
# Fetch the third-party assets that warehouse_v2.world.sdf renders with. They are
# downloaded into the git-ignored src/sim/models_external/ instead of being tracked.
#
#   warehouse_pbr  tileable material sets from ambientCG, CC0 1.0 (public
#                  domain). Concrete034 -> concrete_*, Metal032 -> metal_*.
#                  They surface the floor, wall lining, racking steel and dock
#                  office of warehouse_v2.
#   Large_Crate    OpenRobotics, CC-BY 4.0, Gazebo Fuel. The plastic totes.
#
#   bash src/sim/fetch_external_models.sh
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/models_external"
mkdir -p "$DEST"

TEX="$DEST/warehouse_pbr/materials/textures"
if [ ! -f "$TEX/concrete_albedo.png" ]; then
  mkdir -p "$TEX"
  # Only Color, NormalGL, Roughness and Metalness are kept; the Displacement and
  # NormalDX variants are unused and are 10-17 MB each.
  fetch() {  # $1 ambientCG asset, $2 local stem, $3 "metal" to keep metalness
    echo "fetching $1 (CC0, ambientCG)"
    curl -sL -o /tmp/acg.zip "https://ambientcg.com/get?file=$1_2K-PNG.zip"
    rm -rf /tmp/acg && mkdir -p /tmp/acg
    unzip -qo /tmp/acg.zip "*Color.png" "*NormalGL.png" "*Roughness.png" "*Metalness.png" -d /tmp/acg || true
    cp "/tmp/acg/$1_2K-PNG_Color.png" "$TEX/$2_albedo.png"
    cp "/tmp/acg/$1_2K-PNG_Roughness.png" "$TEX/$2_roughness.png"
    [ "$3" = "metal" ] && cp "/tmp/acg/$1_2K-PNG_Metalness.png" "$TEX/$2_metalness.png"
    # 16-bit normal maps are 10-17 MB; ogre2 wants 8-bit RGB anyway.
    python3 -c "from PIL import Image; Image.open('/tmp/acg/$1_2K-PNG_NormalGL.png').convert('RGB').save('$TEX/$2_normal.png', optimize=True)"
    rm -rf /tmp/acg /tmp/acg.zip
  }
  fetch Concrete034 concrete plain
  fetch Metal032 metal metal
  cat > "$DEST/warehouse_pbr/model.config" <<'CFG'
<?xml version="1.0"?>
<model>
  <name>warehouse_pbr</name>
  <version>1.0</version>
  <description>
    Tileable PBR material sets for warehouse_v2's generated geometry. Source:
    ambientCG (https://ambientcg.com), CC0 1.0 Universal. Not a mesh model: this
    exists only so the maps resolve through a model:// URI.
  </description>
</model>
CFG
fi
# Large Crate: the plastic totes in section C. NOTE its model.sdf ships a
# collision box with a NEGATIVE z extent (Box019) which makes DART abort, so the
# world draws the MESH directly and never includes the model. Do not "fix" that
# by including it.
if [ ! -d "$DEST/Large_Crate" ]; then
  echo "fetching Large Crate (CC-BY 4.0, OpenRobotics)"
  curl -sL -o /tmp/lc.zip \
    "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Large%20Crate/1/Large%20Crate.zip"
  mkdir -p "$DEST/Large_Crate" && unzip -qo /tmp/lc.zip -d "$DEST/Large_Crate" && rm /tmp/lc.zip
fi

# Derivative tote albedos. The crate's own map has mean 20/255, so a stack of
# them is a solid black mass; SDF clamps colour to [0,1] so a >1 diffuse cannot
# lift it. These are multiplicative lifts of the CC-BY source (attribution:
# OpenRobotics), which keeps the moulding detail a flat fill would erase.
if [ ! -f "$TEX/tote_grey_albedo.png" ]; then
  echo "generating tote albedo variants"
  python3 - "$DEST" <<'PY'
from PIL import Image
import numpy as np, sys
d = sys.argv[1]
a = np.asarray(Image.open(f"{d}/Large_Crate/materials/textures/Crate_Albedo.jpg").convert("RGB"), dtype=float)
for name, gain in (("tote_grey", (2.9, 2.9, 2.9)), ("tote_blue", (1.5, 2.2, 3.4))):
    b = np.clip(a * np.array(gain) + 18, 0, 255).astype(np.uint8)
    Image.fromarray(b).save(f"{d}/warehouse_pbr/materials/textures/{name}_albedo.png", optimize=True)
PY
fi

echo "external models ready in $DEST"
du -sh "$DEST"
