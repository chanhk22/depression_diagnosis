import os, glob, pandas as pd, numpy as np

# default threshold for PHQ (PHQ-8): commonly 10 for moderate depression
DEFAULT_PHQ_THRESHOLD = 10.0

def _read_all_csvs(labels_dir):
    rows = []
    for p in glob.glob(os.path.join(labels_dir, "*.csv")):
        try:
            df = pd.read_csv(p)
            df['__source_file'] = os.path.basename(p)
            rows.append(df)
        except Exception as e:
            print(f"[label_mapping] failed to read {p}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)

def canonicalize_column_names(df):
    # lowercase keys
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    # find participant id column
    pid = None
    for cand in ['participant_id','participant','id','session']:
        if cand in cols:
            pid = cols[cand]; break
    # find PHQ score/binary
    phq_score = None
    phq_binary = None
    for cand in ['phq8_score','phq_score','phq8','phq_total','phq']:
        if cand in cols:
            phq_score = cols[cand]; break
    for cand in ['phq8_binary','phq_binary','phq_bin','phq8_b']:
        if cand in cols:
            phq_binary = cols[cand]; break
    gender = None
    for cand in ['gender','sex']:
        if cand in cols:
            gender = cols[cand]; break
    return {'pid':pid, 'phq_score':phq_score, 'phq_binary':phq_binary, 'gender':gender}

def load_labels(labels_dir, dataset_hint=None, phq_threshold=DEFAULT_PHQ_THRESHOLD):
    """
    Returns mapping: session_id -> {'PHQ_Score': float or None, 'PHQ_Binary': 0/1 or None, 'Gender': str or None, 'raw': raw_row}
    """
    df_all = _read_all_csvs(labels_dir)
    if df_all.empty:
        return {}

    mapping = {}
    cols = canonicalize_column_names(df_all)
    pid_col = cols['pid']
    for _, r in df_all.iterrows():
        pid = r.get(pid_col)
        if pd.isna(pid):
            continue
        sid = str(int(pid)) if isinstance(pid, (float,np.floating)) and float(pid).is_integer() else str(pid)
        entry = mapping.get(sid, {})
        # PHQ score
        if cols['phq_score'] and pd.notna(r.get(cols['phq_score'])):
            entry['PHQ_Score'] = float(r.get(cols['phq_score']))
        # PHQ binary
        if cols['phq_binary'] and pd.notna(r.get(cols['phq_binary'])):
            entry['PHQ_Binary'] = int(r.get(cols['phq_binary']))
        # Gender
        if cols['gender'] and pd.notna(r.get(cols['gender'])):
            entry['Gender'] = str(r.get(cols['gender']))
        # raw row (last wins)
        entry['raw_row'] = r.to_dict()
        mapping[sid] = entry

    # postprocess: ensure binary exists (derive from score if missing)
    for sid, e in mapping.items():
        if 'PHQ_Binary' not in e:
            if 'PHQ_Score' in e:
                e['PHQ_Binary'] = 1 if e['PHQ_Score'] >= phq_threshold else 0
            else:
                e['PHQ_Binary'] = None

    # consistency check: warn if mismatch
    inconsistencies = []
    for sid, e in mapping.items():
        if 'PHQ_Score' in e and 'PHQ_Binary' in e and e['PHQ_Binary'] is not None:
            derived = 1 if e['PHQ_Score'] >= phq_threshold else 0
            if derived != e['PHQ_Binary']:
                inconsistencies.append((sid, e['PHQ_Score'], e['PHQ_Binary'], derived))
    if inconsistencies:
        print("[label_mapping] Warning: found PHQ score/binary inconsistencies (sid, score, binary, derived_binary):")
        for t in inconsistencies[:10]:
            print(t)
    return mapping
