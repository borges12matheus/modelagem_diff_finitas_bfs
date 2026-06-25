# Modelo simplificado de escoamento laminar em canal inspirado no Backward Facing Step, resolvido por diferenças finitas

### Formulação do problema

A proposta consiste em partir de um problema físico real, o escoamento em um canal com degrau (*Backward Facing Step - BFS*), e construir uma versão simplificada adequada aos requisitos da disciplina.

A formulação pode ser estruturada da seguinte forma:

- O problema real considerado é o escoamento após um degrau em um canal.
- O problema completo do *Backward Facing Step* é governado pelas equações de Navier-Stokes bidimensionais.
- Como a disciplina exige a resolução de uma **Equação Diferencial Ordinária (EDO)** via método de diferenças finitas, considera-se uma simplificação física do problema.
- Assume-se uma região do escoamento distante da zona de recirculação, onde o comportamento do fluido pode ser aproximado de forma mais simples.
- São adotadas as seguintes hipóteses:
  - escoamento laminar;
  - regime permanente;
  - escoamento totalmente desenvolvido;
  - fluido incompressível.

### Equação governante simplificada

A partir dessas hipóteses, a equação governante reduz-se para:

$$\mu \frac{d^2u}{dy^2} = \frac{dp}{dx}$$


Ou, de forma simplificada:

$$
\frac{d^2u}{dy^2} = C
$$

onde:

- $u(y)$ representa a velocidade do fluido;
- $\mu$ é a viscosidade dinâmica;
- $\frac{dp}{dx}$ representa o gradiente de pressão;
-  $C$ é uma constante.

### Condições de contorno

As condições de contorno adotadas são:

$u(0)=0$ e $u(H)=0$


Essas condições representam a condição de não deslizamento (*no-slip*) nas paredes superior e inferior do canal.

### Discretização por diferenças finitas

Aplicando o método de diferenças finitas para a derivada de segunda ordem:


$$\frac{d^2u}{dy^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta y^2}$$

Substituindo na equação governante:


$$\frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta y^2} = C$$

### Solução numérica

A discretização resulta em um sistema linear algébrico, que pode ser resolvido numericamente para obtenção do perfil de velocidade no canal.