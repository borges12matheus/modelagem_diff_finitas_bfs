from pathlib import Path
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "equations"
OUT.mkdir(parents=True, exist_ok=True)

formulas = [
    r"\frac{d^{2}u}{dy^{2}} = C\left(1+\alpha\sin(\pi y)\right)",
    r"0 \leq y \leq H",
    r"u(0)=0,\qquad u(H)=0",
    r"\rho\left(\frac{\partial \mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)=-\nabla p+\mu\nabla^{2}\mathbf{u}",
    r"u=(u,v), u = u(x,y), v = v(x,y), p = p(x,y)",
    r"\nabla = \left(\frac{\partial}{\partial x},\frac{\partial}{\partial y}\right)",
    r"u\cdot\nabla = u\frac{\partial}{\partial x}+v\frac{\partial}{\partial y}",
    r"u\cdot\nabla u = u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}",
    r"-\nabla p = -\left(\frac{\partial p}{\partial x}\right)",
    r"\frac{\partial u}{\partial t} = 0",
    r"\nu\nabla^{2}u = \nu\left(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}\right)",
    r"\rho\left(u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}\right)=-\frac{\partial p}{\partial x}+\mu\left(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}\right)",
    r"\frac{\partial u}{\partial x} \approx 0 \rightarrow v \approx 0",
    r"\frac{\partial p}{\partial x} \approx -C",
    r"\frac{d^{2}u}{dy^{2}} =\frac{1}{\mu}\frac{dp}{dx}",
    r"C = \frac{1}{\mu}\frac{dp}{dx}",
    r"\frac{d^{2}u}{dy^{2}} = C",
    r"\frac{d^{2}u}{dy^{2}} = C(1+\alpha\sin(\pi y))",
    r"\frac{d^2u}{dy^2}= S(y)",
    r"f(y)=1+\alpha\sin(\pi y)",
    r"\frac{d^2u}{dy^2}= C\cdot f(y)",
    r"\Delta y=\frac{H}{N-1}",
    r"\frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta y^2} = S(y_i)",
    r"u_{i-1} - 2u_i + u_{i+1} = S(y_i)\,\Delta y^2",
    r"A_{w_i}u_{i-1} + Ap_iu_i + Ae_iu_{i+1} = b_i",
    r"\frac{\partial u^{2}{\partial y^{2}}=\frac{1}{\mu}\frac{\partial p}{\partial x}"


]

for i, formula in enumerate(formulas, start=1):

    tex = textwrap.dedent(f"""
    \\documentclass[preview,border=2pt]{{standalone}}
    \\usepackage{{amsmath}}
    \\usepackage{{amssymb}}

    \\begin{{document}}
    \\[
    {formula}
    \\]
    \\end{{document}}
    """)

    tex_file = OUT / f"equacao{i}.tex"
    tex_file.write_text(tex, encoding="utf-8")

    result = subprocess.run(
        ["latex", "-interaction=nonstopmode", tex_file.name],
        cwd=OUT,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Erro ao compilar {tex_file.name}")

    subprocess.run(
        ["dvisvgm", f"equacao{i}.dvi", "-n", "-o", f"equacao{i}.svg"],
        cwd=OUT,
        check=True
    )

print(f"SVGs gerados em: {OUT}")