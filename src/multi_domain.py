
"""
multi_domain.py
Place multiple STL/CAD buildings into a single shared voxel domain and
export one combined headerless RAW file.
 
Concept:
    domain (explicit size, e.g. 200m x 200m x 100m, pitch=2m)
      building 1 -> placed at (x, y) ground position, resting on z=0
      building 2 -> placed at another (x, y) position
      ...
    -> each building is voxelized against the SAME global domain grid
    -> results are OR-combined into one occupancy array
    -> written out as a single RAW file (same byte format as main.py)
 
Config file (JSON) format:
{
  "domain": {
    "size_m": [200, 200, 100],   // domain extents in meters (Lx, Ly, Lz)
    "pitch": 2.0                  // voxel size in meters
  },
  "buildings": [
    {
      "stl": "building1.stl",
      "position_m": [10, 10, 0],  // where the building's footprint corner
                                    // (min x, min y) and base (min z) go
      "target_height": 40,         // optional: rescale to this height (m)
      "scale_axis": "z",           // optional, default "z"
      "facing": "posx"              // optional: posx/negx/posy/negy
    },
    ...
  ]
}
 
Usage:
    python multi_domain.py config.json output.raw [--visualize] [--max-faces N]
"""
 
import argparse
import json
 
import numpy as np
import trimesh
 
from main import (
    load_mesh, simplify_if_needed, orient_mesh, scale_mesh,
    write_raw, read_raw, visualize as visualize_occupancy,
)
 
 
# ---------------------------------------------------------------------------
# Stage 1: domain grid (independent of any single mesh's bounds)
# ---------------------------------------------------------------------------
 
def define_domain_grid(size_m, pitch):
    """
    Define the shared domain grid. Domain origin is always (0, 0, 0);
    all building positions are given relative to this origin.
    """
    size_m = np.array(size_m, dtype=float)
    n_cells = np.ceil(size_m / pitch).astype(int)
    n_cells = np.maximum(n_cells, 1)
    nx, ny, nz = n_cells
 
    x_centers = (np.arange(nx) + 0.5) * pitch
    y_centers = (np.arange(ny) + 0.5) * pitch
    z_centers = (np.arange(nz) + 0.5) * pitch
 
    grid_info = {
        "pitch": pitch,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "z_centers": z_centers,
    }
    return nx, ny, nz, grid_info
 
 
def inspect_domain(nx, ny, nz, grid_info):
    print("=" * 50)
    print("DOMAIN GRID")
    print("=" * 50)
    print(f"pitch: {grid_info['pitch']}")
    print(f"grid shape (nx, ny, nz): ({nx}, {ny}, {nz})")
    print(f"total voxels: {nx * ny * nz:,}")
    print(f"expected RAW size: {nx * ny * nz:,} bytes")
 
 
# ---------------------------------------------------------------------------
# Stage 2: place one building into the domain
# ---------------------------------------------------------------------------
 
def place_building(mesh, position_m):
    """
    Translate the mesh so that its bounding-box minimum corner
    (footprint corner, resting on the ground) lands at position_m.
    This is the "place on the city grid" convention: buildings sit on
    z=0 (plus whatever z offset position_m specifies) at the given
    (x, y) ground coordinate.
    """
    current_min = mesh.bounds[0]
    offset = np.array(position_m, dtype=float) - current_min
    mesh.apply_translation(offset)
    return mesh
 
 
def voxelize_building_into_domain(mesh, nx, ny, nz, domain_grid_info,
                                   pitch, domain_occupancy, name):
    """
    Voxelize a single building only within the domain index range its
    bounding box overlaps (not the whole domain -- much cheaper), then
    OR the result into the shared domain_occupancy array in place.
    """
    x_centers = domain_grid_info["x_centers"]
    y_centers = domain_grid_info["y_centers"]
    z_centers = domain_grid_info["z_centers"]
 
    bmin, bmax = mesh.bounds
 
    def index_range(centers, lo, hi):
        idx = np.where((centers >= lo - pitch) & (centers <= hi + pitch))[0]
        if len(idx) == 0:
            return None
        return idx[0], idx[-1] + 1
     
    # Domain index range this building's bounding box overlaps.
    xr = index_range(x_centers, bmin[0], bmax[0])
    yr = index_range(y_centers, bmin[1], bmax[1])
    zr = index_range(z_centers, bmin[2], bmax[2])
 
    if xr is None or yr is None or zr is None:
        print(f"  WARNING: '{name}' falls entirely outside the domain, skipping")
        return domain_occupancy
 
    ix0, ix1 = xr
    iy0, iy1 = yr
    iz0, iz1 = zr
 
    sub_x = x_centers[ix0:ix1]
    sub_y = y_centers[iy0:iy1]
    sub_z = z_centers[iz0:iz1]
 
    n_points = len(sub_x) * len(sub_y) * len(sub_z)
    print(f"  '{name}': testing {n_points:,} points "
          f"(domain slice x[{ix0}:{ix1}] y[{iy0}:{iy1}] z[{iz0}:{iz1}])")
 
    xx, yy, zz = np.meshgrid(sub_x, sub_y, sub_z, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    inside = mesh.contains(points)
    sub_occupancy = inside.reshape(len(sub_x), len(sub_y), len(sub_z))
    slice_ = domain_occupancy[ix0:ix1, iy0:iy1, iz0:iz1]
    # OR-combine into the shared domain array
    domain_occupancy[ix0:ix1, iy0:iy1, iz0:iz1] = np.maximum(
        slice_, sub_occupancy.astype(np.uint8)
    )
 
    print(f"  '{name}': {int(sub_occupancy.sum()):,} solid voxels added")
    return domain_occupancy
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Place multiple STL buildings into one shared RAW domain."
    )
    parser.add_argument("config", help="path to JSON config file")
    parser.add_argument("output_raw", help="path to output .raw file")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-faces", type=int, default=10000)
    args = parser.parse_args()
 
    with open(args.config) as f:
        config = json.load(f)
 
    domain_cfg = config["domain"]
    nx, ny, nz, domain_grid_info = define_domain_grid(
        domain_cfg["size_m"], domain_cfg["pitch"]
    )
    inspect_domain(nx, ny, nz, domain_grid_info)
 
    domain_occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
 
    print("=" * 50)
    print(f"PLACING {len(config['buildings'])} BUILDINGS")
    print("=" * 50)
 
    for b in config["buildings"]:
        name = b["stl"]
        print(f"\n-- {name} --")
        mesh = load_mesh(name)
        mesh = simplify_if_needed(mesh, args.max_faces)
        mesh = orient_mesh(mesh, b.get("facing"), b.get("rotate_x", 0),
                            b.get("rotate_y", 0), b.get("rotate_z", 0))
        mesh = scale_mesh(mesh, b.get("target_height"), b.get("scale_axis", "z"))
        mesh = place_building(mesh, b["position_m"])
        print(f"  placed at bounds: {mesh.bounds[0]} -> {mesh.bounds[1]}")
 
        domain_occupancy = voxelize_building_into_domain(
            mesh, nx, ny, nz, domain_grid_info, domain_cfg["pitch"],
            domain_occupancy, name
        )
 
    print("=" * 50)
    print("DOMAIN VOXELIZATION RESULT")
    print("=" * 50)
    n_solid = int(domain_occupancy.sum())
    n_total = domain_occupancy.size
    print(f"total solid voxels: {n_solid:,} / {n_total:,} "
          f"({100 * n_solid / n_total:.2f}%)")
 
    write_raw(domain_occupancy, args.output_raw)
 
    # round-trip verify
    check = read_raw(args.output_raw, nx, ny, nz)
    assert np.array_equal(check, domain_occupancy), "round-trip mismatch!"
    print("round-trip verify: OK")
 
    if args.visualize:
        from pathlib import Path
        png_name = f"{Path(args.output_raw).stem}_verify.png"
        visualize_occupancy(domain_occupancy, png_name)
 
    print("=" * 50)
    print(f"DONE. domain grid=({nx},{ny},{nz})  "
          f"buildings={len(config['buildings'])}  output={args.output_raw}")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()
 
