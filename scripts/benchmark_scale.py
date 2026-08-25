"""
benchmark_scale.py
Generate N test buildings, place them into a shared domain via
multi_domain.py's logic, and measure wall-clock time as N grows.
"""

import json
import subprocess
import sys
import time
import resource

import numpy as np
import trimesh


def generate_buildings(n, out_dir="."):
    """Generate n simple box-shaped test buildings with varied sizes."""
    rng = np.random.default_rng(42)
    names = []
    for i in range(n):
        w = rng.uniform(4, 10)
        d = rng.uniform(4, 10)
        h = rng.uniform(8, 40)
        box = trimesh.creation.box(extents=[w, d, h])
        name = f"{out_dir}/bench_building_{i:04d}.stl"
        box.export(name)
        names.append(name)
    return names


def generate_config(n, domain_size, pitch, out_path="bench_config.json"):
    """Place n buildings on a grid layout within the domain, non-overlapping."""
    rng = np.random.default_rng(7)
    names = generate_buildings(n)

    # simple grid layout so buildings (mostly) don't overlap
    cols = int(np.ceil(np.sqrt(n)))
    spacing_x = domain_size[0] / cols
    spacing_y = domain_size[1] / cols

    buildings = []
    for i, name in enumerate(names):
        row = i // cols
        col = i % cols
        x = col * spacing_x + rng.uniform(1, max(2, spacing_x * 0.2))
        y = row * spacing_y + rng.uniform(1, max(2, spacing_y * 0.2))
        target_height = float(rng.uniform(10, 50))
        buildings.append({
            "stl": name,
            "position_m": [float(x), float(y), 0],
            "target_height": target_height,
            "scale_axis": "z",
        })

    config = {
        "domain": {"size_m": domain_size, "pitch": pitch},
        "buildings": buildings,
    }
    with open(out_path, "w") as f:
        json.dump(config, f)
    return out_path


def run_benchmark(n_values, domain_size, pitch):
    results = []
    for n in n_values:
        config_path = generate_config(n, domain_size, pitch)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "multi_domain.py", config_path, f"bench_{n}.raw"],
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"n={n}: FAILED (returncode={proc.returncode})")
            print(proc.stderr[-2000:])
            results.append({"n": n, "elapsed": elapsed, "status": "FAILED"})
            continue

        # pull solid voxel count and grid shape from output for sanity
        solid_line = [l for l in proc.stdout.splitlines() if "total solid voxels" in l]
        grid_line = [l for l in proc.stdout.splitlines() if "grid shape" in l]
        results.append({
            "n": n,
            "elapsed": elapsed,
            "status": "OK",
            "detail": solid_line[0] if solid_line else "",
            "grid": grid_line[0] if grid_line else "",
        })
        print(f"n={n:4d} buildings -> {elapsed:6.2f}s  status=OK  "
              f"({solid_line[0] if solid_line else ''})")

    return results


if __name__ == "__main__":
    # domain large enough to hold up to a few hundred small buildings
    domain_size = [400, 400, 60]
    pitch = 2.0

    n_values = [10, 50, 100, 300, 1000, 2000]
    results = run_benchmark(n_values, domain_size, pitch)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in results:
        print(r)
        
# ---- 결과 저장 + 그래프 생성 ----
import json as _json
with open("bench_results.json", "w") as f:
    _json.dump(results, f, indent=2)

import matplotlib.pyplot as plt
ok_results = [r for r in results if r["status"] == "OK"]
ns = [r["n"] for r in ok_results]
ts = [r["elapsed"] for r in ok_results]

plt.figure(figsize=(8, 5))
plt.plot(ns, ts, marker='o', linewidth=2)
plt.xlabel('Number of Buildings')
plt.ylabel('Execution Time (s)')
plt.title('Scale Benchmark: Execution Time vs. Number of Buildings')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scale_benchmark.png', dpi=200)
print("Saved: scale_benchmark.png")
for r in ok_results:
    print(f"n={r['n']:5d} -> {r['elapsed']:.2f}s")
