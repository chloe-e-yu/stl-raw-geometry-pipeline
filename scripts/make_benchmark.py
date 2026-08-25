import matplotlib.pyplot as plt

buildings = [10, 50, 100, 300, 1000, 2000]
times = [0.28, 0.34, 0.42, 0.72, 1.90, 3.59]

plt.figure(figsize=(8, 5))
plt.plot(buildings, times, marker='o', linewidth=2)
plt.xlabel('Number of Buildings')
plt.ylabel('Execution Time (s)')
plt.title('Scale Benchmark: Execution Time vs. Number of Buildings')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scale_benchmark.png', dpi=200)
print("Saved: scale_benchmark.png")
