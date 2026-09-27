import os
import gc
import collections
import numpy as np
import pandas as pd
from preprocess import norm, get_tokens
from features import extract_features

MAX_CANDIDATES = 60
MAX_POSTINGS = 80
TRAIN_NEGATIVES_PER_SOURCE1 = 5

from tqdm.auto import tqdm

def load(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")

def prep(df):
    return [(str(r.entity_id), norm(r.business_name), norm(r.business_address), str(r.country).lower().strip()) for r in tqdm(df.itertuples(), total=len(df), desc="Normalizing Text")]

def build_index(S):
    index = collections.defaultdict(list)
    for j, (_, name, address, country) in enumerate(S):
        toks = get_tokens(name, address)
        for token in toks:
            key = (country, token)
            if len(index[key]) < MAX_POSTINGS:
                index[key].append(j)
    return index

def candidates_for(record, index):
    _, name, address, country = record
    counter = collections.Counter()
    toks = get_tokens(name, address)
    for token in toks:
        posting = index.get((country, token))
        if posting:
            for j in posting:
                counter[j] += 1
    if not counter:
        return []
    return [j for j, _ in counter.most_common(MAX_CANDIDATES)]

def load_ground_truth(ground_path):
    gt = load(ground_path)
    truth = {}
    for r in gt.itertuples():
        sid = str(r.source1_entity_id)
        raw = str(r.matched_entity_ids).strip()
        if raw:
            truth[sid] = set(x.strip() for x in raw.split(",") if x.strip())
        else:
            truth[sid] = set()
    return truth

def build_training_pairs(A, S, truth, source_name):
    print(f"Building Training Pairs for {source_name}...")
    index = build_index(S)
    sid_to_idx = {row[0]: j for j, row in enumerate(S)}
    
    records_a = []
    records_b = []
    features_list = []
    labels = []

    rng = np.random.default_rng(42)

    for i in tqdm(range(len(A)), desc=f"Generating candidates for {source_name}"):
        a = A[i]
        true_ids = truth.get(a[0], set())
        true_ids = {x for x in true_ids if x in sid_to_idx}
        
        cand = candidates_for(a, index)
        
        # Positives
        for tid in true_ids:
            j = sid_to_idx.get(tid)
            if j is None: continue
            
            b = S[j]
            records_a.append(a)
            records_b.append(b)
            features_list.append(extract_features(a, b))
            labels.append(1)
            
        # Hard Negatives
        negs = [j for j in cand if S[j][0] not in true_ids]
        if len(negs) > TRAIN_NEGATIVES_PER_SOURCE1:
            negs = list(rng.choice(negs, TRAIN_NEGATIVES_PER_SOURCE1, replace=False))
            
        for j in negs:
            b = S[j]
            records_a.append(a)
            records_b.append(b)
            features_list.append(extract_features(a, b))
            labels.append(0)
            
        if len(labels) >= 100000: # Limit for memory/speed during early dev
            break
            
    print(f"Generated {sum(labels)} Positives and {len(labels) - sum(labels)} Negatives.")
    return records_a, records_b, features_list, labels

def prepare_all_data(data_dir):
    cache_path = os.path.join(data_dir, "train_data_cache.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        import pickle
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    tr1 = prep(load(os.path.join(data_dir, "train", "train_source1.tsv")))
    tr2 = prep(load(os.path.join(data_dir, "train", "train_source2.tsv")))
    tr3 = prep(load(os.path.join(data_dir, "train", "train_source3.tsv")))
    gt = load_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    
    a1, b1, f1, l1 = build_training_pairs(tr1, tr2, gt, "Source 2")
    a2, b2, f2, l2 = build_training_pairs(tr1, tr3, gt, "Source 3")
    
    result = (a1+a2, b1+b2, f1+f2, l1+l2)
    print(f"Saving data to cache at {cache_path}...")
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result
