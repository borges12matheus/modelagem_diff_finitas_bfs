import numpy as np
from pathlib import Path

# -----------------------------
# Caso Driver & Seegmiller BFS
# -----------------------------
h = 0.0127  # altura do degrau [m]

xmin, xmax = -130*h, 50*h
ymin, ymax = 0.0, 9*h
z = 0.0

# -----------------------------
# Resolução da grade
# -----------------------------
nx, ny = 350, 160

xs = np.linspace(xmin, xmax, nx)
ys = np.linspace(ymin + 1e-6, ymax - 1e-6, ny)

pts = []

for y in ys:
    for x in xs:

        # Remove região sólida antes do degrau:
        # para x < 0, o fluido só existe acima de y = h
        if x < 0.0 and y < h:
            continue

        pts.append((x, y, z))

pts = np.array(pts, dtype=float)

out_path = Path("gridPoints_driver_seegmiller.xyz")

with out_path.open("w") as f:
    f.write(f"{len(pts)}\n")
    f.write("(\n")

    for x, y, z in pts:
        f.write(f"({x:.8f} {y:.8f} {z:.8f})\n")

    f.write(")\n")

print(f"Gerado {out_path} com {len(pts)} pontos válidos.")
print(f"Grade original: {nx} x {ny} = {nx * ny} pontos.")
print(f"Pontos removidos: {nx * ny - len(pts)}.")
print(f"Domínio físico: x=[{xmin:.4f}, {xmax:.4f}], y=[{ymin:.4f}, {ymax:.4f}]")
print(f"Domínio adimensional: x/h=[{xmin/h:.1f}, {xmax/h:.1f}], y/h=[{ymin/h:.1f}, {ymax/h:.1f}]")
