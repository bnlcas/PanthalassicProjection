# The Panthalassic Projection

![Demo animation](./assets/demo.webp)

An oceancentric world map that shows all the world's oceans in a single continous expanse while minimizing the relative distortion of every major current. This builds upon the [Spilhaus Projection](https://en.wikipedia.org/wiki/Athelstan_Spilhaus#Spilhaus_World_Ocean_Map_Projection) but with the goal to be a faithful representational of the scale of the major currents of the ocean gyres.

## Summary

The Panthalassic projection draws the whole world ocean as one connected, crease-free surface, with every major current close to its true size and shape. It has no closed formula. It is computed: a triangle mesh of the globe is cut through the continents, then flattened by numerical optimization that weights the currents oceanographers compare most.

Across all water, 80% of the ocean is drawn within 8% of its true size, and the median shape error is 5°. The Gulf Stream, Kuroshio, Benguela, the equatorial currents and the Antarctic Circumpolar Current all land between 0.92 and 1.08 of true size, with at most 9° of shape error.

The price is paid on land. Eurasia, Africa and the Americas are cut away and appear only as coastlines, and Antarctica, Australia and the large islands are free to shrink or swell.

&#91;image: The Panthalassic projection with a 10-degree graticule and the equator in gold\]

The whole world ocean on the Panthalassic projection, with a 10° graticule. Antarctica sits near the center inside the Circumpolar ring. The Atlantic and Arctic run up as one lobe, the Pacific fills the lower left and the Indian Ocean the lower right. Parallels bend into smooth arcs, and nowhere at sea do they kink.

## Why another ocean map

Conformal ocean maps get the shape of the world ocean right but draw its currents at very different sizes. Spilhaus built his World Ocean Map on Adams' World in a Square II, which is conformal except at the four corners of the square, and whose areas are hugely exaggerated toward the poles and along the antimeridian ([Esri, Adams square II](https://desktop.arcgis.com/en/arcmap/latest/map/projections/adams-square-ii.htm)). A boundary current drawn on such a map cannot be compared at a glance with one in another basin.

We hit the same wall with the Peirce quincuncial projection. Even with its four singular points moved onto land, the fourth must sit in the central Pacific. The equatorial Pacific is then drawn about 1.6 times too large, rising without limit at the singular point, and the equator bends 90° there.

The Panthalassic projection drops conformality and equal area as hard rules and optimizes directly for what an oceanographer compares:

- one connected world ocean, cut only through land (plus the narrow Bering Strait)
- no creases or singular points anywhere at sea, so flow reads continuously
- comparable size and shape for the western and eastern boundary currents, the equatorial current systems and the Antarctic Circumpolar Current
- land allowed to absorb whatever distortion is left over

## Method

The projection is a piecewise-linear map from a triangle mesh of the globe to the plane, found by minimizing a weighted distortion energy over the water. Each step below is one function in the accompanying script.

1. **Land mask.** Rasterize land on a 0.25° equirectangular grid as a land fraction. Fill enclosed water bodies (Caspian, Great Lakes), keeping the Black Sea.
2. **Mesh.** Subdivide an icosahedron six times: 81,920 triangles, about 1° across. Class each triangle as land or water from four mask samples.
3. **Straits and cuts.** Carve open the straits a 1° mesh would close: Gibraltar, Bab-el-Mandeb, Hormuz, the Danish straits, Hudson Strait. Force land along Central America and Panama, across Suez, and across the Bering Strait. Eurasia, Africa and the Americas then form one landmass, which becomes the map's outer edge.
4. **Coastal buffer.** Keep two rings of continent triangles along every coast, so coastlines are drawn from the fine mask rather than from mesh edges.
5. **Topology check.** The map domain must be one piece with Euler characteristic 1 (a topological disk). Antarctica, Australia, Greenland and the islands stay inside it as land triangles. At level 6 the domain holds 64,964 triangles and 33,154 vertices.
6. **Weights.** Open ocean 1. Key zones 8: twelve boundary-current boxes, the band within 12° of the equator, and the Circumpolar band from 66°S to 40°S. Arctic north of 70°N 0.3, marginal seas 0.4, land 0.02. Final tuning raises the Agulhas box to 24, Drake Passage to 16 and Madagascar's land to 0.3.
7. **Optimize.** Start from an azimuthal equidistant map centered on 65°S, 80°W, the antipode of a point in Siberia, so the start is already one-to-one. Minimize the energy below with local/global iterations (Rabinovich et al., *Scalable Locally Injective Mappings*, 2017). Each step's length is capped so no triangle flips. Optimize first with the base weights, then refine with the local adjustments, each stage up to 400 iterations; the whole build takes about 16 minutes on one CPU core.
8. **Orient and sample.** Put the South Pole at the origin with the North Pole straight above it. Units are Earth radii at true scale. The forward map is tabulated on a 0.5° grid; the inverse is a triangle lookup in the plane.

The energy sums, over triangles f with weight w and spherical area A, the symmetric Dirichlet distortion of the triangle's singular values σ1 and σ2. It is smallest (4 per unit area) when a triangle keeps its true size and shape, and grows without limit as a triangle collapses, which keeps the map free of folds.

```latex
E(u) = \sum_f w_f A_f \left( \sigma_{1}^{2} + \sigma_{1}^{-2} + \sigma_{2}^{2} + \sigma_{2}^{-2} \right)
```

## Results

On the Panthalassic map, ten reference currents fall between 0.92 and 1.08 of true size, with at most 9° of shape error. The Peirce quincuncial keeps shapes perfect but spreads sizes from 0.88 to 1.59. An ocean-lobed Mollweide keeps area exact but bends shapes by up to 35° and must split the Circumpolar Current.

&#91;embedded content: Computed with the projection script and a Tissot-indicatrix check, averaged over a 2° patch at each current · Peirce with its singular points on land\]

Across all water between 80°S and 80°N, 80% of the ocean is drawn within 0.92 to 1.08 of true linear scale, and the median shape error is 5°. The optimized energy sits within 2% of its theoretical lower bound. Tuning cut the Agulhas from 1.16 to 1.06 of true size; the East Australian Current, at 0.92, is now the smallest.

## Limitations and design choices

- **Land is sacrificed.** Eurasia, Africa and the Americas are cut away and survive only as coastlines on two sides of each ocean. Antarctica shrinks, and Madagascar and other islands may swell, because land carries almost no weight.
- **Latitude is curved.** Parallels are smooth arcs around Antarctica, not straight lines. A seasonal shift such as the migration of the Intertropical Convergence Zone reads as movement toward or away from Antarctica. It is consistent from basin to basin, with no kinks.
- **Two water cuts remain.** The Bering Strait is cut, so Arctic–Pacific exchange is not continuous. Where the Panama isthmus is narrower than one triangle, the cut line also covers a sliver of coastal water.
- **The Arctic is down-weighted.** It sits at the tip of the Atlantic lobe and absorbs more distortion than the open basins.
- **Resolution.** The mesh is about 1° (110 km). Distortion is controlled at that scale. Finer detail, such as narrow straits, is placed smoothly but not optimized individually.
- **It is one solution, not the solution.** The result depends on the weights, the cut lines and the mesh. Different priorities, such as weighting the Arctic or shelf seas more, give a different but equally valid Panthalassic map. The script exposes every choice in one parameter table.

## Reproducing it

One Python file, `panthalassic.py`, rebuilds the projection from a land mask and reproduces the numbers above. It needs numpy and scipy, plus Pillow for PNG masks and matplotlib for GeoJSON input and previews. Every choice in the Method section lives in its `PARAMS` table.

```bash
python panthalassic.py build   --land land_mask_0p25deg.png --out panthalassic.npz   # about 16 minutes
python panthalassic.py report  panthalassic.npz                                       # distortion at named currents
python panthalassic.py preview panthalassic.npz --land land_mask_0p25deg.png          # PNG with graticule
python panthalassic.py web     panthalassic.npz                                       # forward grid for the live map
```

The build runs in two stages of up to 400 iterations each: first the base weights, then the Agulhas, Drake Passage and Madagascar adjustments. In Python, the saved projection maps both ways:

```python
from panthalassic import Panthalassic
P = Panthalassic.load('panthalassic.npz')
x, y = P.forward(lon, lat)      # degrees -> map units (1 unit = 1 Earth radius)
lon, lat = P.inverse(x, y)      # NaN outside the map
```

Files that accompany this note:

| File | What it is |
| --- | --- |
| panthalassic.py | The full method: mask, mesh, cuts, weights, optimizer, forward and inverse maps |
| panthalassic.npz | The computed projection: 33,154 vertices on the sphere and in the plane (1.1 MB) |
| land\_mask\_0p25deg.png | The 0.25° land mask used here, traced from a Wikimedia Commons Peirce map |
| panthalassic\_preview.png | The map with a 10° graticule |
| panthalassic\_year.mp4 | The 30-second one-year loop |

For publication, rebuild from Natural Earth's public-domain land polygons by passing `ne_50m_land.geojson` to `--land`. That route was written for this purpose but could not be tested here, because this workspace has no network access. The PNG route produced every result in this note.

## The visualization

The projection is shown in action on a live map, [All Seas Lead South](https://claude.ai/artifact/35HJCXJHAaivLGRunttX6H), and in a 30-second, one-year loop video (panthalassic\_year.mp4, 1080 × 1350, 30 frames per second). Particles trace the surface currents. Each particle is colored by sea surface temperature and drawn brighter where the current runs faster. The live map adds the Peirce quincuncial and a globe for comparison, and a 10° graticule that shows how each layout bends the Earth.

The ocean in these animations is modeled, not observed. It was built without access to satellite products such as NASA OSCAR currents or NOAA OISST temperatures:

- **Mean currents:** a wind-driven (Stommel) circulation model solved on real coastlines, with added Antarctic Circumpolar Current fronts, Gulf Stream and Kuroshio jets, equatorial currents and a seasonal monsoon.
- **Eddies:** synthetic, divergence-free noise whose strength follows the energetic regions. They drift a few kilometers a day and repeat every year, so the video loops without a seam.
- **Temperature:** an idealized climatology with fronts, upwelling, warm pools and an annual cycle.

The projection itself does not depend on any of this. Real current and temperature fields can be drawn on it unchanged, using the script's forward and inverse maps.

## Name and acknowledgements

The authors proposed the name, prompted by their love of Melville's *Moby-Dick*. It is also literal: Panthalassa, from the Greek for "all sea", was the single world ocean that surrounded Pangaea, and today's Pacific is its direct descendant ([Wikipedia, Panthalassa](https://en.wikipedia.org/wiki/Panthalassa)). A map that shows today's oceans as one body of water borrows the word across 225 million years, and across the line between paleoceanography and physical oceanography.

The idea owes most to Athelstan Spilhaus's world ocean maps, which first put the oceans, not the continents, at the center. It also builds on Charles Sanders Peirce's quincuncial projection, used here as the starting point, and on Jarke van Wijk's myriahedral projections, which showed that maps cut through land can keep oceans whole ([van Wijk, 2008](https://www.win.tue.nl/~vanwijk/myriahedral)). The optimizer follows Rabinovich, Poranne, Panozzo and Sorkine-Hornung, *Scalable Locally Injective Mappings* (ACM Transactions on Graphics, 2017).
