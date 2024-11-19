from triangle import triangulate
from scipy.spatial import Delaunay
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

inpnts = Path(r'C:\Users\CNB968\OneDrive - MT\Yellowstone\Canyon Cr\2024-03-12_BATHYMETRY\RTK\Canyon_Cr_Bathy_All.csv')

pnts = pd.read_csv(inpnts)

points = pnts[['Easting', 'Northing']].values

trng = Delaunay(points)
ttrng = triangulate(dict(vertices=points))

mtrng = tri.Triangulation(points[:,0], points[:,1], trng.simplices)

plt.triplot(mtrng)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
