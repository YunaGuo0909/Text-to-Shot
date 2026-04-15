"""
Diagnose why smplh/ files aren't matching during preprocessing.

Usage:
    python scripts/diagnose_smplh.py
"""
import os
import pickle
import numpy as np

ET_ROOT = '/transfer/et-data'
TRAJ_DIR = os.path.join(ET_ROOT, 'traj')
SMPLH_DIR = os.path.join(ET_ROOT, 'smplh')


def main():
    # === 1. File-name matching between traj/ and smplh/ ===
    traj_ids = {f[:-4] for f in os.listdir(TRAJ_DIR) if f.endswith('.txt')}
    smplh_ids = {f[:-4] for f in os.listdir(SMPLH_DIR) if f.endswith('.pkl')}
    common = traj_ids & smplh_ids
    only_smplh = smplh_ids - traj_ids
    only_traj = traj_ids - smplh_ids

    print(f"traj/  has {len(traj_ids)} files")
    print(f"smplh/ has {len(smplh_ids)} files")
    print(f"Common IDs (should be high): {len(common)}")
    print(f"Only in smplh (orphan pkl):  {len(only_smplh)}")
    print(f"Only in traj  (no person):   {len(only_traj)}")

    # === 2. Test loading a sample that IS common ===
    if not common:
        print("\nNo common IDs. Check naming format.")
        return
    sample_id = sorted(common)[0]
    pkl_path = os.path.join(SMPLH_DIR, f'{sample_id}.pkl')
    print(f"\n--- Testing pkl load for: {sample_id} ---")
    try:
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        print(f"Type: {type(d).__name__}")
        if isinstance(d, dict):
            print(f"Keys: {list(d.keys())}")
            for k, v in d.items():
                shape = getattr(v, 'shape', None)
                dtype = getattr(v, 'dtype', None)
                print(f"  {k}: type={type(v).__name__}, shape={shape}, dtype={dtype}")

            if 'transl' in d:
                t = d['transl']
                if hasattr(t, 'cpu'):
                    t = t.cpu().numpy()
                t = np.array(t)
                print(f"\ntransl ndim={t.ndim}, shape={t.shape}")
                print(f"transl first 3 frames: {t[:3]}")
                print(f"transl range: min={t.min():.3f}, max={t.max():.3f}")
    except Exception as e:
        print(f"LOAD FAILED: {type(e).__name__}: {e}")

    # === 3. Scan a batch of common samples with actual load_person_joints logic ===
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.preprocess_et_data import load_person_joints

    n_test = min(100, len(common))
    test_ids = sorted(common)[:n_test]
    success, fail = 0, 0
    fail_reasons = {}
    for sid in test_ids:
        res = load_person_joints(SMPLH_DIR, sid)
        if res is not None:
            success += 1
        else:
            fail += 1
            # Re-run without exception swallowing to find real reason
            try:
                with open(os.path.join(SMPLH_DIR, f'{sid}.pkl'), 'rb') as f:
                    d = pickle.load(f)
                if not isinstance(d, dict):
                    fail_reasons['not_dict'] = fail_reasons.get('not_dict', 0) + 1
                elif 'transl' not in d:
                    fail_reasons['no_transl'] = fail_reasons.get('no_transl', 0) + 1
                else:
                    t = d['transl']
                    if hasattr(t, 'cpu'):
                        t = t.cpu().numpy()
                    t = np.array(t, dtype=np.float32)
                    fail_reasons[f'shape_{t.shape}'] = fail_reasons.get(f'shape_{t.shape}', 0) + 1
            except Exception as e:
                fail_reasons[f'err_{type(e).__name__}'] = fail_reasons.get(f'err_{type(e).__name__}', 0) + 1

    print(f"\n--- load_person_joints() on {n_test} common samples ---")
    print(f"Success: {success}  |  Fail: {fail}")
    if fail_reasons:
        print("Fail reasons:")
        for k, v in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
