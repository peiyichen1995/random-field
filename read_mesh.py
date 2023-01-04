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

import pdb
