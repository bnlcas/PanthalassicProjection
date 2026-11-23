#!/usr/bin/env python3
"""
The Panthalassic projection: an ocean-first world map computed by optimization.

The world ocean is drawn as one connected, crease-free surface on which the major
currents keep close to their true size and shape. There is no closed formula: a
triangle mesh of the globe is cut through the continents and flattened by
minimizing a weighted symmetric-Dirichlet distortion energy, with the boundary
currents, the equatorial band and the Antarctic Circumpolar Current weighted most.

Commands
  python panthalassic.py build   --land LAND [--level 6] [--iters 400] [--out panthalassic.npz]
  python panthalassic.py report  panthalassic.npz
  python panthalassic.py preview panthalassic.npz --land LAND [--png preview.png]
  python panthalassic.py web     panthalassic.npz [--json ocean_proj.json]

LAND is an equirectangular land mask (PNG or .npy; land = 1/white, water = 0/black,
row 0 at 90N, column 0 at 180W), or a GeoJSON of land polygons such as Natural
Earth's ne_50m_land.geojson (public domain), which is rasterized for you.

In Python
  from panthalassic import Panthalassic
  P = Panthalassic.load('panthalassic.npz')
  x, y = P.forward(lon, lat)      # degrees -> map units (1 unit = 1 Earth radius at true scale)
  lon, lat = P.inverse(x, y)      # NaN outside the map

Requires numpy and scipy; Pillow to read PNG masks; matplotlib for GeoJSON input and previews.
"""
import argparse, json, sys, time
import numpy as np
from scipy import sparse, ndimage as ndi
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------- parameters
PARAMS = {
    'mesh_level': 6,                    # icosphere subdivisions: level 6 ~ 1 degree triangles (81,920 faces)
    'buffer_rings': 2,                  # rings of continent faces kept along every coast
    'mask_res_deg': 0.25,
    # narrow straits carved open so marginal seas join the world ocean: (lon, lat) segment ends
    'carve': [((-7.5, 35.9), (-3.5, 36.2)),      # Gibraltar
              ((42.6, 13.9), (44.2, 12.1)),      # Bab-el-Mandeb
              ((55.4, 26.4), (57.4, 26.0)),      # Hormuz
              ((9.5, 57.8), (12.6, 55.4)),       # Kattegat / Danish straits
              ((-66.0, 61.0), (-71.0, 62.6))],   # Hudson Strait
    # cut lines: forced to stay continent so the continents form one outer edge
    'cut': [((-97.5, 19.0), (-92.0, 15.5)), ((-92.0, 15.5), (-88.0, 13.5)), ((-88.0, 13.5), (-83.5, 9.8)),   # Central America
            ((-83.5, 9.8), (-79.5, 9.2)), ((-79.5, 9.2), (-77.2, 8.0)), ((-77.2, 8.0), (-76.5, 7.0)),       # Panama
            ((32.2, 29.8), (32.5, 31.4)), ((32.5, 31.4), (34.3, 31.3)),                                     # Suez / Sinai
            ((170.0, 67.6), (-177.0, 66.8)), ((-177.0, 66.8), (-169.0, 66.0)), ((-169.0, 66.0), (-161.5, 65.2))],  # Bering
    'segment_width_deg': 0.75,
    # points that must lie inside the continents that are cut away
    'continents': {'Eurasia': (90, 55), 'Africa': (20, 5), 'North America': (-100, 45), 'South America': (-60, -10)},
    'keep_seas': [(34, 43)],             # enclosed water bodies kept despite having no mesh-scale strait (Black Sea)
    # energy weights per face (relative to open ocean = 1)
    'w_key': 8.0, 'w_arctic': 0.3, 'w_marginal': 0.4, 'w_land': 0.02,
    'key_zones': {   # (lon0, lon1, lat0, lat1)
        'Gulf Stream': (-82, -40, 24, 46), 'Kuroshio': (118, 172, 20, 42), 'Agulhas': (15, 45, -44, -24),
        'Brazil-Malvinas': (-62, -35, -50, -20), 'East Australian': (148, 166, -42, -18), 'Somali': (40, 62, -6, 16),
        'Benguela': (4, 20, -36, -14), 'California': (-132, -108, 20, 46), 'Humboldt': (-88, -70, -42, -4),
        'Canary': (-26, -8, 12, 36), 'Labrador': (-66, -44, 42, 62), 'Oyashio': (140, 160, 38, 52)},
    'equatorial_band_deg': 12, 'acc_band': (-66, -40), 'arctic_from_lat': 70,
    'marginal': [(-6, 42, 30, 46), (10, 31, 53, 66), (-96, -76, 51, 66), (32, 44, 12, 30), (47, 57, 23, 31)],
    'overrides': [                       # (lon0, lon1, lat0, lat1, weight, 'ocean'|'land'), applied last
        (15, 45, -44, -24, 24.0, 'ocean'),      # Agulhas: the Cape corner otherwise swells to ~1.16x
        (-75, -50, -66, -52, 16.0, 'ocean'),    # Drake Passage: keeps the Agulhas fix from pushing distortion here
        (42, 51.5, -26.5, -11.5, 0.3, 'land')], # Madagascar: stop it ballooning into the Mozambique Channel
    'init_center': (-80.0, -65.0),       # azimuthal start, centred opposite a point inside Siberia
    'iters': 400, 'tol': 1e-6,
}

REPORT_POINTS = [('Gulf Stream', -68, 38), ('Kuroshio', 145, 35), ('Benguela', 12, -25), ('Agulhas', 25, -37),
                 ('Brazil', -45, -30), ('East Australian', 155, -30), ('California', -125, 35), ('Humboldt', -78, -20),
                 ('Equatorial Pacific W', 170, 0), ('Equatorial Pacific E', -120, 0), ('Equatorial Atlantic', -20, 0),
                 ('Equatorial Indian', 75, -2), ('ACC Drake Passage', -63, -58), ('ACC south of Africa', 20, -50),
                 ('ACC south of Australia', 130, -50), ('ACC South Pacific', -130, -58), ('Somali', 50, 5)]


# ----------------------------------------------------------------------------- sphere helpers
def sph(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def lonlat(P):
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(P[..., 1], P[..., 0])), np.degrees(np.arcsin(np.clip(P[..., 2], -1, 1)))


def in_box(lon, lat, lo0, lo1, la0, la1):
    return (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)


# ----------------------------------------------------------------------------- land mask
def load_land(path, res=0.25):
    """Equirectangular land fraction, row 0 at 90N, column 0 at 180W."""
    W, H = int(round(360 / res)), int(round(180 / res))
    if path.lower().endswith(('.geojson', '.json')):
        land = rasterize_geojson(path, W, H)
    elif path.lower().endswith('.npy'):
        land = np.load(path).astype(float)
    else:
        from PIL import Image
        im = Image.open(path).convert('L')
        if im.size != (W, H):
            im = im.resize((W, H), Image.BILINEAR)
        land = np.asarray(im, float) / 255.0
    if land.shape != (H, W):
        land = ndi.zoom(land, (H / land.shape[0], W / land.shape[1]), order=1)
    return fill_enclosed_seas(np.clip(land, 0, 1), res)


def rasterize_geojson(path, W, H, ss=2):
    """Land fraction from polygons (exterior rings minus holes), ss x ss samples per cell."""
    from matplotlib.path import Path
    gj = json.load(open(path))
    feats = gj['features'] if 'features' in gj else [gj]
    polys = []
    for f in feats:
        g = f.get('geometry', f)
        if g['type'] == 'Polygon':
            polys.append(g['coordinates'])
        elif g['type'] == 'MultiPolygon':
            polys.extend(g['coordinates'])
    acc = np.zeros((H, W))
    for sy in range(ss):
        for sx in range(ss):
            lon = -180 + (np.arange(W) + (sx + 0.5) / ss) * 360 / W
            lat = 90 - (np.arange(H) + (sy + 0.5) / ss) * 180 / H
            LO, LA = np.meshgrid(lon, lat)
            pts = np.c_[LO.ravel(), LA.ravel()]
            hit = np.zeros(len(pts), bool)
            for rings in polys:
                ext = np.asarray(rings[0], float)
                lo0, la0 = ext.min(0); lo1, la1 = ext.max(0)
                cand = np.nonzero((pts[:, 0] >= lo0) & (pts[:, 0] <= lo1) & (pts[:, 1] >= la0) & (pts[:, 1] <= la1))[0]
                if not len(cand):
                    continue
                inside = Path(ext).contains_points(pts[cand])
                for hole in rings[1:]:
                    inside &= ~Path(np.asarray(hole, float)).contains_points(pts[cand])
                hit[cand[inside]] = True
            acc += hit.reshape(H, W)
    return acc / (ss * ss)


def fill_enclosed_seas(land, res, keep=None):
    keep = PARAMS['keep_seas'] if keep is None else keep
    water = land < 0.5
    lab, n = ndi.label(water, structure=np.ones((3, 3)))
    sizes = ndi.sum(water, lab, range(1, n + 1))
    ok = np.zeros(n + 1, bool); ok[np.argmax(sizes) + 1] = True
    for lo, la in keep:
        ok[lab[int((90 - la) / res), int((lo + 180) / res)]] = True
    ok[0] = False
    return np.where(water & ~ok[lab], 1.0, land)


def sample_land(lq, lon, lat):
    H, W = lq.shape
    fx = (lon + 180) / 360 * W - 0.5; fy = (90 - lat) / 180 * H - 0.5
    x0 = np.floor(fx).astype(int); y0 = np.clip(np.floor(fy).astype(int), 0, H - 2)
    ax = fx - x0; ay = np.clip(fy - y0, 0, 1)
    X = lambda i: i % W
    return (lq[y0, X(x0)] * (1 - ax) + lq[y0, X(x0 + 1)] * ax) * (1 - ay) + (lq[y0 + 1, X(x0)] * (1 - ax) + lq[y0 + 1, X(x0 + 1)] * ax) * ay


# ----------------------------------------------------------------------------- mesh and cut topology
def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
                  [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
                  [8, 6, 7], [9, 8, 1]])
    V /= np.linalg.norm(V, axis=1)[:, None]
    for _ in range(level):
        edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        inv = inv.ravel()
        mids = V[uniq[:, 0]] + V[uniq[:, 1]]
        mids /= np.linalg.norm(mids, axis=1)[:, None]
        base = len(V); V = np.vstack([V, mids]); nF = len(F)
        a, b, c = base + inv[:nF], base + inv[nF:2 * nF], base + inv[2 * nF:]
        F = np.vstack([np.c_[F[:, 0], a, c], np.c_[F[:, 1], b, a], np.c_[F[:, 2], c, b], np.c_[a, b, c]])
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (n * V[F].mean(1)).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, F


def face_adjacency(F):
    nF = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(nF), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    k, f = key[order], fid[order]
    same = np.all(k[1:] == k[:-1], axis=1)
    a, b = f[:-1][same], f[1:][same]
    A = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(nF, nF))
    return (A + A.T).tocsr()


def masked(adj, m):
    D = sparse.diags(m.astype(float))
    A = (D @ adj @ D).tocsr(); A.eliminate_zeros()
    return A


def near_segment(C, a, b, width_deg):
    pa, pb = sph(*a), sph(*b)
    n = np.cross(pa, pb); n /= np.linalg.norm(n)
    d = np.abs(C @ n)
    inside = (np.cross(pa, C) @ n >= -1e-9) & (np.cross(C, pb) @ n >= -1e-9)
    return (np.degrees(np.arcsin(np.clip(d, 0, 1))) < width_deg) & inside


def build_domain(lq, prm, log=print):
    """Faces of an icosphere split into the map domain (water + interior land + coastal buffer) and the cut-away continents."""
    V, F = icosphere(prm['mesh_level'])
    C = V[F].mean(1); C /= np.linalg.norm(C, axis=1)[:, None]
    samples = [C] + [(2 * V[F[:, k]] + C) / 3 for k in range(3)]
    land = np.mean([sample_land(lq, *lonlat(q)) for q in samples], axis=0) > 0.5
    for a, b in prm['carve']:
        land[near_segment(C, a, b, 0.7)] = False
    forced = np.zeros(len(F), bool)
    for a, b in prm['cut']:
        forced |= near_segment(C, a, b, prm['segment_width_deg'])
    land |= forced
    adj = face_adjacency(F)
    nc, lab = connected_components(masked(adj, land), directed=False)
    probe = lambda lo, la: np.argmax(C @ sph(lo, la))
    conts = {lab[probe(*p)] for p in prm['continents'].values() if land[probe(*p)]}
    exterior = land & np.isin(lab, list(conts))
    water = ~land
    nw, wl = connected_components(masked(adj, water), directed=False)
    main = np.argmax(np.bincount(wl[water], minlength=nw))
    stray = water & (wl != main)
    for c in np.unique(wl[stray]):
        m = stray & (wl == c)
        exterior[m] = exterior[adj[m].indices].any(); land[m] = True
    ocean = ~land
    reach = ocean.copy()
    for _ in range(prm['buffer_rings']):
        reach |= np.asarray(adj[reach].sum(0)).ravel() > 0
    ext = exterior & ~(reach & ~forced)
    # topology: one cut-away piece, one map piece, no pinched vertices -> the map domain is a disk
    vfe = sparse.coo_matrix((np.ones(3 * len(F)), (F.ravel(), np.repeat(np.arange(len(F)), 3))), shape=(len(V), len(F))).tocsr()
    for _ in range(20):
        changed = False
        for piece, flip_to in ((ext, False), (~ext, True)):
            n_, l_ = connected_components(masked(adj, piece), directed=False)
            keep = np.argmax(np.bincount(l_[piece], minlength=n_))
            frag = piece & (l_ != keep)
            if frag.any():
                ext[frag] = flip_to; changed = True
        dom = ~ext
        idx = np.nonzero(dom)[0]
        vf = sparse.coo_matrix((np.ones(3 * len(idx)), (F[idx].ravel(), np.repeat(idx, 3))), shape=(len(V), len(F))).tocsr()
        deg_d, deg_a = np.diff(vf.indptr), np.diff(vfe.indptr)
        for v in np.nonzero((deg_d > 0) & (deg_d < deg_a))[0]:
            fs = vf.indices[vf.indptr[v]:vf.indptr[v + 1]]
            if connected_components(adj[fs][:, fs], directed=False)[0] > 1:
                ext[vfe.indices[vfe.indptr[v]:vfe.indptr[v + 1]]] = False; changed = True
        if not changed:
            break
    for name, p in prm['continents'].items():
        if not ext[probe(*p)]:
            raise RuntimeError(f'{name} is no longer part of the cut-away continents; widen the cut lines')
    dom = ~ext
    Fd = F[dom]
    E = np.unique(np.sort(np.concatenate([Fd[:, [0, 1]], Fd[:, [1, 2]], Fd[:, [2, 0]]]), axis=1), axis=0)
    chi = len(np.unique(Fd)) - len(E) + len(Fd)
    if chi != 1:
        raise RuntimeError(f'map domain is not a disk (Euler characteristic {chi})')
    log(f'mesh: {len(F)} faces, map domain {dom.sum()} faces ({ocean.sum()} water), disk topology ok')
    return V, F, C, dom, land


# ----------------------------------------------------------------------------- weights
def face_weights(clon, clat, land, prm):
    w = np.ones(len(clon))
    key = np.abs(clat) < prm['equatorial_band_deg']
    key |= (clat > prm['acc_band'][0]) & (clat < prm['acc_band'][1])
    for lo0, lo1, la0, la1 in prm['key_zones'].values():
        key |= in_box(clon, clat, lo0, lo1, la0, la1)
    w[key] = prm['w_key']
    w[clat > prm['arctic_from_lat']] = prm['w_arctic']
    marg = np.zeros(len(clon), bool)
    for b in prm['marginal']:
        marg |= in_box(clon, clat, *b)
    w[marg & ~key] = prm['w_marginal']
    w[land] = prm['w_land']
    for lo0, lo1, la0, la1, wt, kind in prm['overrides']:
        m = in_box(clon, clat, lo0, lo1, la0, la1) & (land if kind == 'land' else ~land)
        w[m] = wt
    return w


# ----------------------------------------------------------------------------- optimizer (SLIM, symmetric Dirichlet)
class Flattener:
    """Minimizes sum_f w_f A_f (s1^2 + s1^-2 + s2^2 + s2^-2) over plane positions, keeping every triangle unflipped.

    Local/global iterations after Rabinovich et al., 'Scalable Locally Injective Mappings' (2017)."""

    def __init__(self, X, F, w):
        self.F, self.w, self.nV = F, w, len(X)
        P0, P1, P2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
        e1 = P1 - P0; x1 = np.linalg.norm(e1, axis=1); e1 /= x1[:, None]
        n = np.cross(P1 - P0, P2 - P0); A2 = np.linalg.norm(n, axis=1); n /= A2[:, None]
        e2 = np.cross(n, e1)
        x2 = ((P2 - P0) * e1).sum(1); y2 = ((P2 - P0) * e2).sum(1)
        self.area = A2 / 2
        rows = np.repeat(np.arange(len(F)), 3)
        dx = np.c_[-1 / x1, 1 / x1, np.zeros(len(F))]
        dy = np.c_[(x2 - x1) / (x1 * y2), -x2 / (x1 * y2), 1 / y2]
        self.Dx = sparse.csr_matrix((dx.ravel(), (rows, F.ravel())), shape=(len(F), self.nV))
        self.Dy = sparse.csr_matrix((dy.ravel(), (rows, F.ravel())), shape=(len(F), self.nV))

    def jac(self, u):
        J = np.empty((len(self.F), 2, 2))
        J[:, 0, 0] = self.Dx @ u[:, 0]; J[:, 0, 1] = self.Dy @ u[:, 0]
        J[:, 1, 0] = self.Dx @ u[:, 1]; J[:, 1, 1] = self.Dy @ u[:, 1]
        return J

    def energy(self, u):
        J = self.jac(u)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if (det <= 0).any():
            return np.inf
        fro = (J ** 2).sum((1, 2))
        return (self.w * self.area * (fro + fro / det ** 2)).sum()

    def lower_bound(self):
        return 4 * (self.w * self.area).sum()

    def _proxy(self, u, lam=1e-4):
        U, S, Vt = np.linalg.svd(self.jac(u))
        R = U @ Vt
        with np.errstate(divide='ignore', invalid='ignore'):
            sw = np.sqrt((S - S ** -3) / (S - 1))
        sw = np.where(np.abs(S - 1) < 1e-6, 2.0, sw)
        Wm = U @ (sw[:, :, None] * np.transpose(U, (0, 2, 1)))
        c = np.sqrt(self.w * self.area)
        w11, w12, w21, w22 = (Wm[:, i, j] * c for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)])
        r11, r12, r21, r22 = R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1]
        dg, Dx, Dy = sparse.diags, self.Dx, self.Dy
        A = sparse.vstack([sparse.hstack([dg(w11) @ Dx, dg(w12) @ Dx]), sparse.hstack([dg(w11) @ Dy, dg(w12) @ Dy]),
                           sparse.hstack([dg(w21) @ Dx, dg(w22) @ Dx]), sparse.hstack([dg(w21) @ Dy, dg(w22) @ Dy])]).tocsc()
        b = np.concatenate([w11 * r11 + w12 * r21, w11 * r12 + w12 * r22, w21 * r11 + w22 * r21, w21 * r12 + w22 * r22])
        M = (A.T @ A).tocsc()
        reg = lam * M.diagonal().mean()
        sol = splu(M + reg * sparse.eye(M.shape[0])).solve(A.T @ b + reg * np.concatenate([u[:, 0], u[:, 1]]))
        return np.c_[sol[:self.nV], sol[self.nV:]]

    def _max_step(self, u, d):
        J, dJ = self.jac(u), self.jac(d)
        a, b, c, e = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
        da, db, dc, de = dJ[:, 0, 0], dJ[:, 0, 1], dJ[:, 1, 0], dJ[:, 1, 1]
        qa, qb, qc = da * de - db * dc, a * de + e * da - b * dc - c * db, a * e - b * c
        t = np.full(len(a), np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            lin = np.abs(qa) < 1e-14
            tl = -qc / qb
            t = np.where(lin & (tl > 0), tl, t)
            disc = qb * qb - 4 * qa * qc
            sq = np.sqrt(np.maximum(disc, 0))
            for r in ((-qb - sq) / (2 * qa), (-qb + sq) / (2 * qa)):
                t = np.where(~lin & (disc >= 0) & (r > 0), np.minimum(t, r), t)
        return t.min()

    def run(self, u, iters=400, tol=1e-6, log=print, patience=8):
        E, t0, recent = self.energy(u), time.time(), []
        for it in range(iters):
            d = self._proxy(u) - u
            t = min(1.0, 0.9 * self._max_step(u, d))
            En = self.energy(u + t * d)
            while En > E and t > 1e-10:
                t *= 0.5; En = self.energy(u + t * d)
            if En >= E:
                break
            rel = (E - En) / E
            u, E = u + t * d, En
            recent = (recent + [rel])[-patience:]
            done = len(recent) == patience and max(recent) < tol      # a sustained stall, not one short step
            if it % 10 == 0 or done:
                log(f'  iteration {it:3d}  energy {E:.5g}  (bound {self.lower_bound():.5g})  {time.time() - t0:.0f}s')
            if done:
                break
        return u


def azimuthal_start(X, lon0, lat0):
    c = sph(lon0, lat0)
    cos_t = np.clip(X @ c, -1, 1); th = np.arccos(cos_t)
    e = np.cross([0, 0, 1.0], c); e /= np.linalg.norm(e); n = np.cross(c, e)
    v = X - cos_t[:, None] * c[None, :]
    vn = np.linalg.norm(v, axis=1); vn[vn == 0] = 1
    return np.c_[th * (v @ e) / vn, th * (v @ n) / vn]


# ----------------------------------------------------------------------------- the projection object
class Panthalassic:
    """Piecewise-linear map from the sphere to the plane, stored on the optimized mesh.

    Map units: 1 = one Earth radius at true scale (6371 km). y points north-up on the page:
    the North Pole is placed straight above the South Pole."""

    def __init__(self, X, F, uv, w=None, params=None):
        self.X, self.F, self.uv = X, F, uv
        self.w, self.params = w, params or {}
        C = X[F].mean(1); self.C = C / np.linalg.norm(C, axis=1)[:, None]
        self._tree = cKDTree(self.C)
        pc = uv[F].mean(1)
        self._ptree = cKDTree(pc)
        self._grid = None

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        prm = json.loads(str(d['params'])) if 'params' in d else {}
        return cls(d['X'], d['F'], d['uv'], d['w'] if 'w' in d else None, prm)

    def save(self, path):
        np.savez_compressed(path, X=self.X, F=self.F, uv=self.uv, w=self.w, params=json.dumps(self.params))

    def forward(self, lon, lat, k=10):
        """lon/lat in degrees -> x, y in map units; NaN where the map is cut (inside Eurasia-Africa or the Americas)."""
        lon, lat = np.broadcast_arrays(np.asarray(lon, float), np.asarray(lat, float))
        P = sph(lon.ravel(), lat.ravel())
        _, cand = self._tree.query(P, k=k)
        out = np.full((len(P), 2), np.nan); best = np.full(len(P), -np.inf)
        for j in range(k):
            f = cand[:, j]
            T = np.stack([self.X[self.F[f, 0]], self.X[self.F[f, 1]], self.X[self.F[f, 2]]], 2)
            lam = np.linalg.solve(T, P[..., None])[..., 0]
            lam /= lam.sum(1, keepdims=True)
            mn = lam.min(1)
            upd = (mn > best) & (mn > -1e-6)
            out[upd] = (lam[:, :, None] * self.uv[self.F[f]]).sum(1)[upd]
            best = np.where(mn > best, mn, best)
        return out[:, 0].reshape(lon.shape), out[:, 1].reshape(lon.shape)

    def inverse(self, x, y, k=12):
        """x, y in map units -> lon/lat in degrees; NaN outside the map."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        Q = np.c_[x.ravel(), y.ravel()]
        _, cand = self._ptree.query(Q, k=k)
        out = np.full((len(Q), 3), np.nan); best = np.full(len(Q), -np.inf)
        for j in range(k):
            f = cand[:, j]
            p0, p1, p2 = self.uv[self.F[f, 0]], self.uv[self.F[f, 1]], self.uv[self.F[f, 2]]
            d1, d2, dq = p1 - p0, p2 - p0, Q - p0
            den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
            b1 = (dq[:, 0] * d2[:, 1] - dq[:, 1] * d2[:, 0]) / den
            b2 = (d1[:, 0] * dq[:, 1] - d1[:, 1] * dq[:, 0]) / den
            lam = np.c_[1 - b1 - b2, b1, b2]
            mn = lam.min(1)
            upd = (mn > best) & (mn > -1e-9)
            out[upd] = (lam[:, :, None] * self.X[self.F[f]]).sum(1)[upd]
            best = np.where(mn > best, mn, best)
        lo, la = lonlat(out)
        return lo.reshape(x.shape), la.reshape(x.shape)

    def distortion(self):
        """Per-face linear scale sqrt(s1 s2) and maximum angular distortion (degrees)."""
        fl = Flattener(self.X, self.F, np.ones(len(self.F)))
        S = np.linalg.svd(fl.jac(self.uv), compute_uv=False)
        return np.sqrt(S[:, 0] * S[:, 1]), np.degrees(2 * np.arcsin((S[:, 0] - S[:, 1]) / (S[:, 0] + S[:, 1]))), fl.area

    def report(self, land=None, out=print):
        scale, shape, area = self.distortion()
        water = np.ones(len(self.F), bool) if land is None else ~land
        clon, clat = lonlat(self.C)
        water &= np.abs(clat) < 80
        out(f'water area drawn at {np.percentile(scale[water], 10):.2f}-{np.percentile(scale[water], 90):.2f} of true linear scale '
            f'(10th-90th percentile); shape error median {np.median(shape[water]):.0f} deg, 90th percentile {np.percentile(shape[water], 90):.0f} deg')
        out(f'{"":24s} scale   shape')
        rows = []
        for name, lo, la in REPORT_POINTS:
            m = ((self.C @ sph(lo, la)) > np.cos(np.radians(2.5))) & water
            s = np.sqrt(np.sum(scale[m] ** 2 * area[m]) / np.sum(area[m])); a = np.sum(shape[m] * area[m]) / np.sum(area[m])
            rows.append((name, s, a)); out(f'{name:24s} {s:5.2f}x  {a:4.0f} deg')
        return rows

    def node_grid(self):
        """Forward map sampled on a 0.5-degree node grid (721 x 361, poles and the dateline included)."""
        if self._grid is None:
            lon = -180 + 0.5 * np.arange(721); lat = 90 - 0.5 * np.arange(361)
            LO, LA = np.meshgrid(lon, lat)
            x, y = self.forward(LO, LA)
            self._grid = (x, y, np.isfinite(x))
        return self._grid


def orient(X, F, uv):
    """Rotate and centre the plane map: South Pole at the origin, North Pole straight above it."""
    P = Panthalassic(X, F, uv)
    sx, sy = P.forward(0.0, -89.99); nx, ny = P.forward(0.0, 89.99)
    ang = np.arctan2(float(nx - sx), float(ny - sy))
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (uv - [float(sx), float(sy)]) @ R.T


# ----------------------------------------------------------------------------- commands
def cmd_build(a):
    prm = dict(PARAMS)
    if a.params:
        prm.update(json.load(open(a.params)))
    prm['mesh_level'] = a.level; prm['iters'] = a.iters
    lq = load_land(a.land, prm['mask_res_deg'])
    V, F, C, dom, land = build_domain(lq, prm)
    Fd = F[dom]; used = np.unique(Fd)
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    Fl, X = remap[Fd], V[used]
    clon, clat = lonlat(C[dom])
    # stage 1: base weights; stage 2: the local overrides, warm-started from stage 1
    base = dict(prm, overrides=[])
    fl = Flattener(X, Fl, face_weights(clon, clat, land[dom], base))
    u = azimuthal_start(X, *prm['init_center'])
    if np.isinf(fl.energy(u)):
        u[:, 0] *= -1                                     # keep the map orientation-preserving (no mirror image)
    print(f'stage 1: optimizing {len(Fl)} triangles, {len(X)} vertices')
    u = fl.run(u, prm['iters'], prm['tol'])
    w = face_weights(clon, clat, land[dom], prm)
    if prm['overrides']:
        print('stage 2: local weight overrides')
        fl = Flattener(X, Fl, w)
        u = fl.run(u, prm['iters'], prm['tol'])
    u = orient(X, Fl, u)
    P = Panthalassic(X, Fl, u, w, {k: v for k, v in prm.items()})
    P.save(a.out)
    np.save(a.out.replace('.npz', '_land.npy'), land[dom])
    print(f'saved {a.out}')
    P.report(land[dom])


def cmd_report(a):
    P = Panthalassic.load(a.file)
    lp = a.file.replace('.npz', '_land.npy')
    land = np.load(lp) if __import__('os').path.exists(lp) else None
    P.report(land)


def cmd_preview(a):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P = Panthalassic.load(a.file)
    lq = load_land(a.land)
    gx, gy, ok = P.node_grid()
    fig, ax = plt.subplots(figsize=(9, 10), dpi=130)
    fig.patch.set_facecolor('#15171b'); ax.set_facecolor('#15171b')
    # land fraction on the node grid, drawn as filled contours in map space
    lon = -180 + 0.5 * np.arange(721); lat = 90 - 0.5 * np.arange(361)
    LO, LA = np.meshgrid(lon, lat)
    lf = sample_land(lq, LO, LA)
    Z = np.where(ok, lf, np.nan)
    gxf, gyf = np.where(ok, gx, np.nan), np.where(ok, gy, np.nan)
    ax.contourf(np.nan_to_num(gxf), np.nan_to_num(gyf), np.ma.masked_invalid(1 - Z), levels=[0.5, 1.01], colors=['#1f4e79'])
    ax.contourf(np.nan_to_num(gxf), np.nan_to_num(gyf), np.ma.masked_invalid(Z), levels=[0.5, 1.01], colors=['#3a3f47'])
    ax.contour(np.nan_to_num(gxf), np.nan_to_num(gyf), np.ma.masked_invalid(Z), levels=[0.5], colors=['#d9dde3'], linewidths=0.5)
    def draw(los, las, **kw):
        x, y = P.forward(los, las)
        ax.plot(x, y, **kw)
    for la in range(-80, 90, 10):
        lo = np.linspace(-180, 180, 1441)
        draw(lo, np.full_like(lo, la), color='#f0c45a' if la == 0 else '#9fb3c8', lw=1.0 if la == 0 else (0.5 if la % 30 == 0 else 0.25))
    for lo in range(-180, 180, 10):
        la = np.linspace(-89.5, 89.5, 720)
        draw(np.full_like(la, lo), la, color='#9fb3c8', lw=0.5 if lo % 30 == 0 else 0.25)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('The Panthalassic projection  \u00b7  10\u00b0 graticule, equator in gold', color='#e9eaec', fontsize=12)
    fig.savefig(a.png, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'saved {a.png}')


def cmd_web(a):
    """Pack the forward grid as used by the live map: 16-bit x/y (hi/lo bytes), y pointing down."""
    import zlib, base64
    P = Panthalassic.load(a.file)
    x, y, ok = P.node_grid()
    _, (iy, ix) = ndi.distance_transform_edt(~ok, return_indices=True)
    xy = np.stack([x[iy, ix], -y[iy, ix]], -1)
    mn = xy[ok].min(0); span = (xy[ok].max(0) - mn).max() * 1.0001
    q = np.round((xy - mn) / span * 65535).astype(np.uint16)
    tex = np.stack([q[..., 0] >> 8, q[..., 0] & 255, q[..., 1] >> 8, q[..., 1] & 255], -1).astype(np.uint8)
    def pack(arr):
        flt = arr.copy(); flt[:, 1:] = (arr[:, 1:].astype(int) - arr[:, :-1].astype(int)) % 256
        return base64.b64encode(zlib.compress(flt.astype(np.uint8).tobytes(), 9)).decode()
    json.dump({'w': 721, 'h': 361, 'x0': float(mn[0]), 'y0': float(mn[1]), 'span': float(span), 'fwd': pack(tex),
               'ok': pack((ok * 255).astype(np.uint8)[..., None])}, open(a.json, 'w'))
    print(f'saved {a.json}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='compute the projection')
    b.add_argument('--land', required=True); b.add_argument('--out', default='panthalassic.npz')
    b.add_argument('--level', type=int, default=PARAMS['mesh_level']); b.add_argument('--iters', type=int, default=PARAMS['iters'])
    b.add_argument('--params', help='JSON file overriding entries of PARAMS')
    r = sub.add_parser('report', help='distortion at named currents'); r.add_argument('file')
    p = sub.add_parser('preview', help='PNG with coastlines and graticule'); p.add_argument('file'); p.add_argument('--land', required=True)
    p.add_argument('--png', default='panthalassic_preview.png')
    w = sub.add_parser('web', help='forward grid for the live map'); w.add_argument('file'); w.add_argument('--json', default='ocean_proj.json')
    a = ap.parse_args(argv)
    {'build': cmd_build, 'report': cmd_report, 'preview': cmd_preview, 'web': cmd_web}[a.cmd](a)


if __name__ == '__main__':
    main()
