"""
main.py
STL/CAD -> voxel grid -> headerless RAW file.
__________________________________________
1. 배경
   - GPU 기반 도심 바람장 LBM solver는 계산 영역을 3D 격자로 나누고,
     각 격자가 fluid(공기)인지 solid(건물)인지 구분해야 함
   - 지금까지는 OSM(지도 데이터)에서 건물 정보를 가져왔지만, 정밀한 custom
     형상(직접 설계한 건물, 실험용 장애물 등)은 표현이 어려움
   - 그래서 STL/CAD 형상을 같은 입력 형식으로 변환하는 별도 프로그램이 필요
 
2. RAW 파일이란
   - 특정 표준 형식이 아니라, header 없이 데이터만 순서대로 저장한 binary 파일
   - 각 격자(voxel)를 uint8로 저장: fluid=0, solid=1
   - Header가 없어 격자 크기/단위/좌표축 방향은 파일 자체로 알 수 없음
     (만드는 쪽과 읽는 쪽이 별도로 알고 있어야 함)
 
3. 이번 작업
   STL/CAD 형상
     -> 형상의 크기와 좌표 확인
     -> 계산에 사용할 3차원 격자 정의
     -> 각 격자가 형상 내부인지 외부인지 판정
     -> 유체는 0, 고체는 1로 표시
     -> 정해진 순서로 RAW 파일 저장
     -> 생성한 파일을 다시 읽어 형상 확인
   - 기존 OSM generator 코드 구조를 따를 필요 없음, 완전히 새 프로그램으로 작성
   - 언어/라이브러리/입력 형식/격자 설정 방식은 자유롭게 결정
 
4. 첨부파일
   - city_citymask.raw: 실제 RAW 크기/형식 예시 (400x400x250, uint8, 40MB)
   - Chloe_OSM_Generator_Reference: OSM->RAW 변환 참고 코드 (참고용, 수정 대상 아님)
 
5. LBM solver와 연결하기 위해 맞춰야 하는 RAW 형식
   - 자료형: 부호 없는 8비트 정수 uint8
   - 유체 영역: 0, 고체 영역: 1
   - 논리 격자 크기: (nx, ny, nz)
   - 가장 빠르게 변하는 축: Z 방향
   - Header 없음
   - 정확한 파일 크기: nx * ny * nz bytes
   - index = z + nz * (y + ny * x)
 
6. RAW 형식을 쓰는 이유
   - STL/CAD는 형상(삼각형, 곡면 등) 표현에 유용하지만, solver는 매 계산마다
     fluid/solid만 빠르게 확인하면 됨
   - 전처리 단계에서 미리 RAW로 변환해두면 solver가 원본 형식을 몰라도 되고,
     fread로 연속 byte를 읽어 GPU 메모리로 바로 복사 가능
 
7. 테스트 형상 선정 기준
   - 건물/구조물처럼 형상이 명확할 것
   - 내부/외부 구분 가능한 닫힌(watertight) 3차원 형상
   - X, Y, Z 크기가 서로 달라 축 순서 확인이 쉬울 것
   - 단위/실제 크기를 확인할 수 있을 것
   - 복잡도가 적절할 것 (처음부터 복잡한 형상일 필요 없음)
   - 출처와 사용 조건을 확인할 수 있는 공개 모델일 것
_________________________
기존 STL -> RAW 흐름은 그대로 유지

1. 격자 크기 설정 (physical pitch)
    - --pitch: voxel 한 칸의 실제 물리 크기(m)를 직접 지정
    - --target-cells: 가장 긴 축을 목표를 격자 개수로 맞추고 pitch를 역산
2. 물체의 실제 크기 설정 (scaling)
    - --target-height: 물체의 특정 축 방향 크기를 목표 물리 크기(m)로 맞추고, 전체 형상을 같은 비율로 확대/축소
    - --scale-axis: 어느 축을 기준으로 크기를 맞출지 (기본값 z)
3. 물체의 방향 설정 (orientation)
    - --facing: STL의 정면이 원래 +X를 향한다고 가정하고 원하는 방향 (+x/-x/+y/-y)을 보도록 Z축 기준 회전 
    - --rotate-x/--rotate-y/--rotate-z: 각 축 기준 회전각(도)를 직접 지정

적용 순서: 회전 -> 크기 조정 -> 격자 정의 
(bounding box 각 단계 이후 상태를 정확히 반영함
__________________________________________________

Pipeline: STL 읽기 -> 격자 정의 -> 내부/외부 판정 -> fluid=0/solid=1 저장
          -> 다시 읽어 확인
(no.3 "이에 해볼 작업"의 흐름)

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

    # no. 7: "내부와 외부를 구분할 수 있는 닫힌 3차원 형상일 것"
    print(f"watertight: {mesh.is_watertight}")
    if not mesh.is_watertight:
        print("  WARNING: mesh is not watertight. Voxelization results")
        print("  may be inaccurate. Consider trimesh.repair.")

    bounds = mesh.bounds
    print(f"bounds (min): {bounds[0]}")
    print(f"bounds (max): {bounds[1]}")

    # no. 7: "X, Y, Z 방향의 크기가 서로 달라 축 순서가 바뀌었는지
    # 확인하기 쉬울 것"
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
        print (f"pitch: derviedd from target_cells={target_cells}"
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
    # no. 5: "정확한 파일 크기 nx x ny x nz bytes"
    print(f"expected RAW size: {nx * ny * nz:,} bytes")


def voxelize_mesh(mesh, nx, ny, nz, grid_info):
    x_centers = grid_info["x_centers"]
    y_centers = grid_info["y_centers"]
    z_centers = grid_info["z_centers"]

    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    print(f"testing {points.shape[0]:,} points against the mesh...")
    inside = mesh.contains(points)

    # no.3: "유체는 0, 고체는 1로 표시"
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
    # no. 5:
    #   자료형: uint8, 유체=0, 고체=1, header 없음
    #   가장 빠르게 변하는 축: Z (index = z + nz*(y + ny*x))
    # numpy 배열이 (nx, ny, nz) shape이면 .tofile()의 기본 저장 순서
    # (C-order)가 이 규칙과 그대로 일치함.
    assert occupancy.dtype == np.uint8, "occupancy must be uint8"
    occupancy.tofile(output_path)

    print("=" * 50)
    print("STAGE 4: WRITE RAW")
    print("=" * 50)
    print(f"wrote {output_path} ({occupancy.size:,} bytes expected)")


def read_raw(raw_path, nx, ny, nz):
    data = np.fromfile(raw_path, dtype=np.uint8)

    # no. 5: "정확한 파일 크기 nx x ny x nz bytes"
    expected_size = nx * ny * nz
    assert data.size == expected_size, (
        f"file size mismatch: expected {expected_size}, got {data.size}"
    )
    return data.reshape(nx, ny, nz)


def verify_raw(raw_path, nx, ny, nz):
    # no.3: "생성한 파일을 다시 읽어 형상 확인"
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


def visualize(occupancy, png_path="verify_output.png"):
    import matplotlib.pyplot as plt

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
    # no.3 흐름: 형상 읽기 -> 크기/좌표 확인
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

    # no.3 흐름: 3차원 격자 정의
    pitch = resolve_pitch(mesh, args.pitch, args.target_cells)
    nx, ny, nz, grid_info = define_grid(mesh, pitch)
    inspect_grid(nx, ny, nz, grid_info)

    # no.3 흐름: 내부/외부 판정, 유체=0/고체=1
    occupancy = voxelize_mesh(mesh, nx, ny, nz, grid_info)
    inspect_occupancy(occupancy)

    # no.3 흐름: RAW 파일 저장
    write_raw(occupancy, args.output_raw)

    # no.3 흐름: 생성한 파일을 다시 읽어 형상 확인
    verify_raw(args.output_raw, nx, ny, nz)

    if args.visualize:
        visualize(occupancy)

    print("=" * 50)
    print(f"DONE. grid=({nx},{ny},{nz})  pitch={pitch:.6f}  "
          f"output={args.output_raw}")
    print("=" * 50)


if __name__ == "__main__":
    main()