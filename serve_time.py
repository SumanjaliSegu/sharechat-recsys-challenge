import lightgbm as lgb, numpy as np, time, os
M = {"LightGBM fast (lr 0.03)": "D:/sharechat/out/multi_k/lgb_full_v2emb_nopost_lr0.03_ist_fullho_dropH_stable_nocatid_dropE_reg_ff0.5.txt",
     "LightGBM slow (lr 0.01)": "D:/sharechat/out/multi_k/lgb_full_v2emb_nopost_lr0.01_ist_fullho_dropH_stable_nocatid_dropE_neg1.0_reg_ff0.5.txt"}
rng = np.random.default_rng(0)
for name, f in M.items():
    b = lgb.Booster(model_file=f); nf = b.num_feature(); nt = b.num_trees()
    X = rng.standard_normal((20000, nf)).astype(np.float32)
    b.predict(X[:1000], num_threads=1)
    for th in (1, 4):
        t = time.perf_counter(); b.predict(X, num_threads=th); dt_ = time.perf_counter() - t
        print(f"{name}: {nt} trees, {nf} inputs, {os.path.getsize(f)/1e6:.1f} MB | threads {th}: {dt_/20:.2f} ms per 1,000 rows")
    t = time.perf_counter()
    for i in range(200): b.predict(X[i:i+1], num_threads=1)
    print(f"   single row: {(time.perf_counter()-t)/200*1e3:.2f} ms")
