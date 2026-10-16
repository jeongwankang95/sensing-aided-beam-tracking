"""gps_geometry.py: GPS geometry against best-beam motion on the test passes of each scenario, medians over passes: pass span (s),
the best beam's net change and path (indices) and their ratio, the bearing sweep seen from the base station over the pass
(deg), the bearing rate (slope over 11-frame windows, deg/s), the minimum and median range (m) and the GPS path speed (m/s).
Vehicle positions are the calibrated ones (unit2_loc_cal) where the scenario provides them (Scenario 9) and the raw GPS
(unit2_loc) elsewhere; a last line repeats Scenario 9 with the raw GPS.
Paper: supplementary Sec. S4: median GPS speeds on the test passes of 3.9, 4.9, 2.5 and 3.4 m/s on Scenarios 9, 31, 32 and 33;
on Scenario 9 raw GPS shrinks the median bearing sweep of a test pass from 77 deg to 8 deg.
usage: python analysis/gps_geometry.py"""
import numpy as np, pandas as pd, os, sys, glob
def bearing(bs, G):
    lat0, lon0 = np.radians(bs); lat, lon = np.radians(G[:, 0]), np.radians(G[:, 1])
    y = np.sin(lon-lon0)*np.cos(lat); x = np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0)
    dist = 6371000*np.arccos(np.clip(np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0), -1, 1))
    return np.degrees(np.arctan2(y, x)), dist
for scn, raw in (('scenario9', False), ('scenario31', False), ('scenario32', False), ('scenario33', False), ('scenario9', True)):
    root = f'dataset/{scn}'
    f = f'{root}/{scn}.csv' if os.path.exists(f'{root}/{scn}.csv') else glob.glob(f'{root}/*_dev.csv')[0]
    df = pd.read_csv(f); cols = list(df.columns)
    pcol = [c for c in cols if 'pwr' in c][0]
    gcal = [c for c in cols if 'unit2' in c and 'cal' in c]; graw = [c for c in cols if c.startswith('unit2_loc') and 'cal' not in c]
    gcol = gcal[0] if gcal and not raw else graw[0]
    tcol = [c for c in cols if 'time' in c.lower()][0]
    bsf = [c for c in cols if c.startswith('unit1_loc')]
    bs = np.loadtxt(os.path.join(root, str(df[bsf[0]].iloc[0]).lstrip('./')))[:2] if bsf else None
    tp = set(pd.read_csv(f'{root}/test_seqs_RA.csv')['seq_index'].unique())
    ts = df[tcol].astype(str).str.extract(r"(\d+)[-:](\d+)[-:](\d+)[-.](\d+)").astype(float)
    df['t'] = ts[0]*3600 + ts[1]*60 + ts[2] + ts[3]/10**ts[3].astype(int).astype(str).str.len()
    ordc = 'index' if 'index' in cols else cols[0]
    R = []
    for p, g in df.groupby('seq_index'):
        if p not in tp: continue
        g = g.sort_values(ordc)
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0) for r in g[pcol]])
        G = np.stack([np.loadtxt(os.path.join(root, str(r).lstrip('./')))[:2] for r in g[gcol]])
        b = P.argmax(1); th, r = bearing(bs, G); t = g['t'].values; thu = np.degrees(np.unwrap(np.radians(th)))
        rate = [abs(np.polyfit(t[i-5:i+6]-t[i], thu[i-5:i+6], 1)[0]) for i in range(5, len(t)-5)]
        step = np.hypot(np.diff(G[:, 0])*111320, np.diff(G[:, 1])*111320*np.cos(np.radians(G[0, 0])))
        R.append(dict(n=len(b), span=t[-1]-t[0], dbeam=abs(int(b[-1])-int(b[0])), path=int(np.abs(np.diff(b)).sum()),
                      dth=abs(thu[-1]-thu[0]), rate=np.median(rate) if rate else np.nan, rmin=r.min(), rmed=np.median(r),
                      spd=step.sum()/max(t[-1]-t[0], 1e-9)))
    R = pd.DataFrame(R); m = R.median()
    print(f'{scn:11s} gps={gcol:14s} passes={len(R):2d} span={m.span:5.2f}s dbeam={m.dbeam:5.1f} path={m.path:5.1f} '
          f'straight={np.median(R.dbeam/R.path.clip(lower=1)):.2f} dtheta={m.dth:6.1f}deg rate_med={m.rate:6.2f}deg/s '
          f'rmin={m.rmin:5.1f}m rmed={m.rmed:5.1f}m speed={m.spd:4.2f}m/s')
