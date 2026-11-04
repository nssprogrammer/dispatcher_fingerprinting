exec(open('load_evasion.py').read())
exec(open(str(ROOT / 'lib/persist_test.py')).read())
rng = np.random.default_rng(1); N=len(keys)
report(fplib, d.a.values.astype(np.int64), d.t_arrival.values, d.t_end.values, N, t0, t1, grp, rng, "NB8-B")
