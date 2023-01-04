import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import meshzoo
import numpy as np

import gstools as gs

# generate a triangulated hexagon with meshzoo
points, cells = meshzoo.rectangle_tri(
    np.linspace(0.0, 1.0, 11),
    np.linspace(0.0, 1.0, 11),
    variant="zigzag",  # or "up", "down", "center"
)
mesh = meshio.Mesh(points, {"triangle": cells})

# number of fields
fields_no = 1
# model setup
# model = gs.Gaussian(dim=2, len_scale=0.5)
model = gs.Exponential(dim=2, var=1, len_scale=[0.0001, 0.5], angles=np.pi/4.)
srf = gs.SRF(model, mean=1)

for i in range(fields_no):
    srf.mesh(mesh, points="centroids", name="c-field-{}".format(i), seed=i)

for i in range(fields_no):
    srf.mesh(mesh, points="points", name="p-field-{}".format(i), seed=i)

triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
# figure setup
cols = 1
rows = int(np.ceil(fields_no / cols))

fig = plt.figure(figsize=[2 * cols, 2 * rows])
for i, field in enumerate(mesh.point_data, 1):
    ax = fig.add_subplot(rows, cols, i)
    ax.tricontourf(triangulation, mesh.point_data[field])
    ax.triplot(triangulation, linewidth=0.5, color="k")
    ax.set_aspect("equal")
fig.tight_layout()
plt.savefig('random_field.png')
plt.show()

mesh.write("mesh_ensemble.vtk")
