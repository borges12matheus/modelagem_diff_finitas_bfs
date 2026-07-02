import numpy as np
from pathlib import Path

# ======================================================
# Configuração do domínio
# ======================================================

x_min = -0.0206
x_max = 0.2900

y_min = -0.0254
y_max = 0.0254

Nx = 500
Ny = 160

# Caminho de saída
output_file = Path("gridPoints.xyz")

# ======================================================
# Geração da malha
# ======================================================

x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)

X, Y = np.meshgrid(x, y, indexing="ij")

# ======================================================
# Escrita no formato OpenFOAM
# ======================================================

with open(output_file, "w") as f:
    f.write("(\n")

    for i in range(Nx):
        for j in range(Ny):
            f.write(
                f"({X[i,j]:.8f} {Y[i,j]:.8f} 0.0)\n"
            )

    f.write(")\n")

print(f"Arquivo salvo em: {output_file.resolve()}")
print(f"Total de pontos: {Nx*Ny:,}")