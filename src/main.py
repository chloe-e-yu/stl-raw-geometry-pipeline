"""
main.py
STL/CAD -> voxel grid -> headerless RAW file.
__________________________________________
1. Background
   - The GPU-based urban wind-field LBM solver divides the computational
     domain into a 3D grid, and each grid cell must be classified as
     fluid (air) or solid (building).
   - Until now, building data has come from OSM (map data), but precise
     custom shapes (custom-designed buildings, experimental obstacles,
     etc.) are hard to represent that way.
   - So a separate program is needed to convert STL/CAD shapes into the
     same input format.

2. What is a RAW file
   - Not a specific standard format — just a binary file storing data in
     order, with no header.
   - Each grid cell (voxel) is stored as uint8: fluid=0, solid=1.
   - Since there's no header, the grid size / units / axis directions
     can't be recovered from the file itself (the writer and reader must
     agree on them separately).

3. This task
   STL/CAD shape
     -> check the shape's size and coordinates
     -> define the 3D grid to use for computation
     -> determine, for each grid cell, whether it's inside or outside
        the shape
     -> mark fluid as 0, solid as 1
     -> save as a RAW file in the defined order
     -> read the generated file back to verify the shape
   - No need to follow the existing OSM generator's code structure —
     write this as a completely new program.
   - Language / libraries / input format / grid-setup approach are all
     free to decide.

4. Attachments
   - city_citymask.raw: example of the actual RAW size/format
     (400x400x250, uint8, 40MB)
   - Chloe_OSM_Generator_Reference: reference code for OSM->RAW
     conversion (reference only, not meant to be modified)

5. RAW format required to interface with the LBM solver
   - dtype: unsigned 8-bit integer (uint8)
   - fluid region: 0, solid region: 1
   - logical grid size: (nx, ny, nz)
   - fastest-varying axis: Z direction
   - no header
   - exact file size: nx * ny * nz bytes
   - index = z + nz * (y + ny * x)

6. Why use the RAW format
   - STL/CAD is useful for representing shapes (triangles, curved
     surfaces, etc.), but the solver only needs a fast fluid/solid
     lookup on every computation step.
   - Converting to RAW ahead of time in a preprocessing stage means the
     solver doesn't need to know the original format, and can read
     contiguous bytes via fread and copy them straight into GPU memory.

7. Criteria for choosing test shapes
   - Should look like a clearly defined building/structure.
   - Should be a closed (watertight) 3D shape where inside/outside can
     be distinguished.
   - X, Y, Z sizes should differ so axis ordering is easy to verify.
   - Units / real-world size should be verifiable.
   - Complexity should be reasonable (doesn't need to be complex from
     the start).
   - Should be a public model with verifiable source and usage terms.
_________________________
The original STL -> RAW flow stays the same.

1. Grid size configuration (physical pitch)
    - --pitch: directly specify the real-world physical size (m) of one
      voxel
    - --target-cells: fit the longest axis to a target number of grid
      cells and back-derive the pitch
2. Real-world object size configuration (scaling)
    - --target-height: scale the whole shape uniformly so the object's
      size along a given axis matches a target physical size (m)
    - --scale-axis: which axis to use as the basis for sizing (default z)
3. Object orientation configuration
    - --facing: assuming the STL's front originally faces +X, rotate
      around the Z axis so it faces the desired direction (+x/-x/+y/-y)
    - --rotate-x/--rotate-y/--rotate-z: directly specify the rotation
      angle (degrees) around each axis

Application order: rotate -> scale -> define grid
(the bounding box accurately reflects the state after each step)
__________________________________________________

Pipeline: read STL -> define grid -> inside/outside test -> write
          fluid=0/solid=1 -> read back to verify
(the flow described in "3. This task" above)

Usage:
    python main.py <input.stl> <pitch> <output.raw>
    [--pitch P | --target-cells N]
    [--target-height H [--scale-axis {x,y,z}]]
    [--facing {posx, negx, posy, negy} | --rotate-x D --rotate-y D --rotate-z D]
    [--visualize] [--max-faces N]
"""

import argparse

import numpy as np
import trimesh


def load_mesh(stl_path):
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    return mesh


def simplify_if_needed(mesh, max_faces):
    n_faces = len(mesh.faces)
    if n_faces <= max_faces:
        print(f"face count OK: {n_faces:,} <= {max_faces:,}, no simplification needed")
        return mesh

    print(f"face count {n_faces:,} exceeds {max_faces:,}, simplifying...")
    simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
    print(f"  faces after simplification: {len(simplified.faces):,}")
    print(f"  watertight after simplification: {simplified.is_watertight}")
    return simplified


def inspect_mesh(mesh, label="MESH INSPECTION"):
    print("=" * 50)
    print(f"STAGE 1: {label}")
    print("=" * 50)

    # criterion 7: "should be a closed 3D shape where inside/outside
    # can be distinguished"
    print(f"watertight: {mesh.is_watertight}")
    if not mesh.is_watertight:
        print("  WARNING: mesh is not watertight. Voxelization results")
        print("  may be inaccurate. Consider trimesh.repair.")

    bounds = mesh.bounds
    print(f"bounds (min): {bounds[0]}")
    print(f"bounds (max): {bounds[1]}")

    # criterion 7: "X, Y, Z sizes should differ so it's easy to check
    # whether the axis order got swapped"
    extents = mesh.extents
    print(f"extents (x, y, z size): {extents}")
    if len(set(extents.round(6))) < 3:
        print("  NOTE: two or more axes have the same size.")

    print(f"volume: {mesh.volume:.4f}")
    if mesh.volume < 0:
        print("  WARNING: negative volume. Face normals may be flipped.")

FACING_TO_Z_ANGLE_DEG = {
    "posx": 0,
    "posy": 90,
    "negx": 180,
    "negy": 270,
}

def orient_mesh(mesh, facing, rotate_x, rotate_y, rotate_z):
    if facing is None and rotate_x == 0 and rotate_y == 0 and rotate_z == 0:
        print("orientation: no rotation applied")
        return mesh

    center = mesh.centroid

    if facing is not None:
        angle_deg = FACING_TO_Z_ANGLE_DEG[facing]
        print(f"oreintation: facing={facing} -> rotating {angle_deg} deg around Z")
        matrix = trimesh.transformations.rotation_matrix(
            np.radians(angle_deg), [0,0,1], point=center
            )
        mesh.apply_transform(matrix)
        return mesh

    print(f"orientation: explicit rotation (x={rotate_x}, y={rotate_y}, z={rotate_z})")
    for axis, angle_deg in zip(
        [[1,0,0],[0,1,0],[0,0,1]],
        [rotate_x, rotate_y, rotate_z],
    ):
        if angle_deg == 0:
            continue
        matrix = trimesh.transformations.rotation_matrix(
            np.radians(angle_deg), axis, point=center
        )
        mesh.apply_transform(matrix)

    return mesh

AXIS_INDEX = {"x":0, "y":1, "z":2}

def scale_mesh(mesh, target_height, scale_axis):
    """scale the mesh uniformly so that its extent along scale_axis equals
    targtet_height(physical units, e.g. meters)
    """
    if target_height is None:
        print("scaling: no scaling applied (using original STL, coordinate size)")
        return mesh

    if target_height <=0:
        raise ValueError(
             f"--target-height must be positive (got: {target_height})"
        )

    axis_idx = AXIS_INDEX[scale_axis]
    current_size = mesh.extents[axis_idx]

    if current_size <= 0:
        raise ValueError(f"'{scale_axis}' axis size is <= 0, cannot scale")

    factor = target_height / current_size
    print(f"scaling: current size along '{scale_axis}'={current_size:.6f} "
          f"-> target={target_height} (factor={factor:.6f})")
    
    mesh.apply_scale(factor)
    return mesh

def resolve_pitch(mesh, pitch, target_cells):
    """ use pitch directly if given. 
        Otherwise derive pitch from target_cells: 
            the longest axis extent by divided by target_cells. 
    """
    if pitch is not None:
        if pitch <= 0:
            raise ValueError(f"--pitch must be positive (got: {pitch})")
        print(f"pitch: using directly specified value ({pitch})")
        return pitch

    if target_cells is not None:
        if target_cells <= 0:
            raise ValueError(
                f"--target-cells must be positive (got: {target_cells})"
            )
        longest_extent = mesh.extents.max()
        pitch = longest_extent / target_cells
        print (f"pitch: dervied from target_cells={target_cells}"
               f"(longest axis={longest_extent:.6f}/{target_cells} = {pitch:.6f})")
        return pitch
    raise ValueError("either --pitch or --target-cells must be specified")

def define_grid(mesh, pitch):
    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    extents = maxs - mins

    n_cells = np.ceil(extents / pitch).astype(int)
    n_cells = np.maximum(n_cells, 1)
    nx, ny, nz = n_cells

    x_centers = mins[0] + (np.arange(nx) + 0.5) * pitch
    y_centers = mins[1] + (np.arange(ny) + 0.5) * pitch
    z_centers = mins[2] + (np.arange(nz) + 0.5) * pitch

    grid_info = {
        "bounds": bounds,
        "pitch": pitch,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "z_centers": z_centers,
    }
    return nx, ny, nz, grid_info


def inspect_grid(nx, ny, nz, grid_info):
    print("=" * 50)
    print("STAGE 2: GRID DEFINITION")
    print("=" * 50)
    print(f"pitch: {grid_info['pitch']}")
    print(f"grid shape (nx, ny, nz): ({nx}, {ny}, {nz})")
    print(f"total voxels: {nx * ny * nz:,}")
    # criterion 5: "exact file size nx x ny x nz bytes"
    print(f"expected RAW size: {nx * ny * nz:,} bytes")


def voxelize_mesh(mesh, nx, ny, nz, grid_info):
    x_centers = grid_info["x_centers"]
    y_centers = grid_info["y_centers"]
    z_centers = grid_info["z_centers"]

    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    print(f"testing {points.shape[0]:,} points against the mesh...")
    inside = mesh.contains(points)

    # criterion 3: "mark fluid as 0, solid as 1"
    occupancy = inside.reshape(nx, ny, nz).astype(np.uint8)
    return occupancy


def inspect_occupancy(occupancy):
    print("=" * 50)
    print("STAGE 3: VOXELIZATION RESULT")
    print("=" * 50)
    n_solid = int(occupancy.sum())
    n_total = occupancy.size
    print(f"shape: {occupancy.shape}, dtype: {occupancy.dtype}")
    print(f"solid voxels: {n_solid:,} / {n_total:,} "
          f"({100 * n_solid / n_total:.2f}%)")
    if n_solid == 0:
        print("  WARNING: no solid voxels found. Check units / bounds.")


def write_raw(occupancy, output_path):
    # criterion 5:
    #   dtype: uint8, fluid=0, solid=1, no header
    #   fastest-varying axis: Z (index = z + nz*(y + ny*x))
    # If the numpy array has shape (nx, ny, nz), .tofile()'s default
    # storage order (C-order) matches this convention exactly.
    assert occupancy.dtype == np.uint8, "occupancy must be uint8"
    occupancy.tofile(output_path)

    print("=" * 50)
    print("STAGE 4: WRITE RAW")
    print("=" * 50)
    print(f"wrote {output_path} ({occupancy.size:,} bytes expected)")


def read_raw(raw_path, nx, ny, nz):
    data = np.fromfile(raw_path, dtype=np.uint8)

    # criterion 5: "exact file size nx x ny x nz bytes"
    expected_size = nx * ny * nz
    assert data.size == expected_size, (
        f"file size mismatch: expected {expected_size}, got {data.size}"
    )
    return data.reshape(nx, ny, nz)


def verify_raw(raw_path, nx, ny, nz):
    # criterion 3: "read the generated file back to verify the shape"
    print("=" * 50)
    print("STAGE 5: VERIFY RAW")
    print("=" * 50)

    occupancy = read_raw(raw_path, nx, ny, nz)
    print(f"read back shape: {occupancy.shape}")

    unique_vals = np.unique(occupancy)
    print(f"unique values: {unique_vals}")
    assert set(unique_vals.tolist()).issubset({0, 1}), "found values other than 0/1!"
    print("OK: file size and values are valid")

    return occupancy


def visualize(occupancy, png_path=None):
    import datetime
    import matplotlib.pyplot as plt

    if png_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = f"verify_output_{timestamp}.png"

    solid_coords = np.argwhere(occupancy == 1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        solid_coords[:, 0], solid_coords[:, 1], solid_coords[:, 2],
        c=solid_coords[:, 2], cmap="viridis", marker="s", s=10,
    )
    ax.set_xlabel("x (grid index)")
    ax.set_ylabel("y (grid index)")
    ax.set_zlabel("z (grid index)")
    ax.set_title("Voxelized shape (solid voxels)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"saved plot to {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an STL/CAD file into a headerless RAW voxel file."
    )
    parser.add_argument("stl_path", help="path to input STL/CAD file")
    parser.add_argument("output_raw", help="path to output .raw file")
    grid_group = parser.add_mutually_exclusive_group(required=True)
    grid_group.add_argument("--pitch", type=float, 
                            help="voxel size in physical units (e.g. meters)")
    grid_group.add_argument("--target-cells", type=int, 
                                help="derive pitch so the longest axis has this many cells")
    parser.add_argument("--target-height", type=float, default=None, help="scale the mesh so its size along --scale-axis"
                        "equals this value (physical units, e.g. meters)")
    parser.add_argument("--scale-axis", choices = ["x", "y", "z"], default="z",
                        help="which axis --target-height applies to(defalut: z)")
    parser.add_argument("--facing", choices=["posx", "negx", "posy", "negy"], default=None, 
                        help="rotate so the mesh's front (assumed +x) "
                        "points this direction")
    parser.add_argument("--rotate-x", type=float, default=0,
                        help="rotation angle (degrees) around X axis")
    parser.add_argument("--rotate-y", type=float, default=0,
                        help="rotation angle (degrees) around Y axis")
    parser.add_argument("--rotate-z", type=float, default=0,
                        help="rotation angle (degrees) around Z axis")
    parser.add_argument("--visualize", action="store_true",
                         help="save a 3D scatter plot of the result")
    parser.add_argument("--max-faces", type=int, default=10000,
                         help="simplify the mesh if it has more faces than this "
                              "(default: 10000)")
    args = parser.parse_args()

    has_explicit_rotation = args.rotate_x or args.rotate_y or args.rotate_z
    if args.facing is not None and has_explicit_rotation:
        parser.error(
            "--facing and --rotate-x/--rotate-y/--rotate-z cannot be used "
            "together, USe only one of them"
        )
    # task-3 flow: read shape -> check size/coordinates
    mesh = load_mesh(args.stl_path)
    mesh = simplify_if_needed(mesh, args.max_faces)
    inspect_mesh(mesh,label="MESH INSPECTION (original)")
    print("=" * 50)
    print("ORIENTATION")
    print("=" * 50)
    mesh = orient_mesh(mesh, args.facing, args.rotate_x, args.rotate_y, args.rotate_z)

    print("=" * 50)
    print("SCALING")
    print("=" * 50)
    mesh = scale_mesh(mesh, args.target_height, args.scale_axis)

    if args.facing is not None or has_explicit_rotation or args.target_height is not None:
        inspect_mesh(mesh, label="MESH INSPECTION (after transform)")

    # task-3 flow: define the 3D grid
    pitch = resolve_pitch(mesh, args.pitch, args.target_cells)
    nx, ny, nz, grid_info = define_grid(mesh, pitch)
    inspect_grid(nx, ny, nz, grid_info)

    # task-3 flow: inside/outside test, fluid=0/solid=1
    occupancy = voxelize_mesh(mesh, nx, ny, nz, grid_info)
    inspect_occupancy(occupancy)

    # task-3 flow: write the RAW file
    write_raw(occupancy, args.output_raw)

    # task-3 flow: read the generated file back to verify the shape
    verify_raw(args.output_raw, nx, ny, nz)

    if args.visualize:
        from pathlib import Path
        png_name = f"{Path(args.output_raw).stem}_verify.png"
        visualize(occupancy, png_name)

    print("=" * 50)
    print(f"DONE. grid=({nx},{ny},{nz})  pitch={pitch:.6f}  "
          f"output={args.output_raw}")
    print("=" * 50)


if __name__ == "__main__":
    main()