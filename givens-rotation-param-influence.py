import numpy as np
a = np.array([
[2,2,0,0],
[2,2,0,0],
[0,0,1,0],
[0,0,0,1],
])
b = np.array([
[1,0,0,0],
[0,3,3,0],
[0,3,3,0],
[0,0,0,1],
])
c = [
[1,0,0,0],
[0,1,0,0],
[0,0,4,4],
[0,0,4,4],
]
d = [
[5,0,0,5],
[0,1,0,0],
[0,0,1,0],
[5,0,0,5],
]
print(a@b@c@d)

import sympy as sp
num_params = 4
planes = [(i, (i+1) % num_params) for i in range(num_params)]
angles = sp.symbols('θ0 θ1 θ2 θ3', real=True)
rotations = sp.eye(num_params)
# sequentially apply each Givens rotation
for (i, j), angle in zip(planes, angles):
    c = sp.cos(angle)
    s = sp.sin(angle)
    givens_rotation = sp.eye(num_params)
    # 2×2 block in the (i,j) plane
    givens_rotation[i, i] = c;  givens_rotation[j, i] = s
    givens_rotation[i, j] = -s; givens_rotation[j, j] = c
    rotations = rotations @ givens_rotation

# for each entry, collect which θ’s actually appear
influence = {}
for i in range(4):
    for j in range(4):
        influence[(i,j)] = {θ for θ in angles if θ in sp.simplify(rotations[i,j]).free_symbols}

for i in range(4):
    row = []
    for j in range(4):
        deps = sorted(str(t) for t in influence[(i,j)])
        row.append("{" + ",".join(deps) + "}")
    print("row", i, ":", row)
