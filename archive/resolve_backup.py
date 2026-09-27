import pandas as pd
import re
import unicodedata
import collections
import numpy as np
import os
import zipfile
import shutil

from rapidfuzz import process, fuzz
from sklearn.linear_model import LogisticRegression


# ============================================================
# PATH
# ============================================================

B = './'


# ============================================================
# LOAD
# ============================================================

def load(p):
    return pd.read_csv(p, sep='\t').fillna('')


# ============================================================
# NORMALIZATION
# ============================================================

def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = s.replace('&', ' and ')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


# ============================================================
# PREPARE DATA
# ============================================================

def prep(d):
    return [
        (
            r.entity_id,
            norm(r.business_name),
            norm(r.business_address),
            str(r.country).lower()
        )
        for r in d.itertuples()
    ]


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def make_candidates(A, S, k=100, lim=30):

    byc = collections.defaultdict(list)
    inv = collections.defaultdict(list)

    # Build country and token indexes
    for j, (_, nm, ad, c) in enumerate(S):

        byc[c].append(j)

        for t in set(nm.split()) | set(ad.split()):

            if len(t) >= 3:
                inv[(c, t)].append(j)

    ids_by_i = [set() for _ in A]

    # Fuzzy retrieval within country
    for c, choices_idx in byc.items():

        qidx = [
            i for i, r in enumerate(A)
            if r[3] == c
        ]

        if not qidx:
            continue

        choices = [
            S[j][1] + ' ' + S[j][2]
            for j in choices_idx
        ]

        queries = [
            A[i][1] + ' ' + A[i][2]
            for i in qidx
        ]

        M = process.cdist(
            queries,
            choices,
            scorer=fuzz.ratio,
            workers=-1
        )

        kk = min(k, len(choices))

        top = np.argpartition(
            M,
            -kk,
            axis=1
        )[:, -kk:]

        for qi, row in enumerate(top):

            ids_by_i[qidx[qi]].update(
                choices_idx[j]
                for j in row
            )

    # Token blocking
    for i, (_, nm, ad, c) in enumerate(A):

        for t in set(nm.split()) | set(ad.split()):

            if len(t) >= 3 and len(inv[(c, t)]) <= lim:

                ids_by_i[i].update(
                    inv[(c, t)]
                )

    return [
        (i, j)
        for i, ids in enumerate(ids_by_i)
        for j in ids
    ]


# ============================================================
# FEATURES
# ============================================================

def feat(A, S, P):

    ns = [A[i][1] for i, j in P]
    ns2 = [S[j][1] for i, j in P]

    ads = [A[i][2] for i, j in P]
    ads2 = [S[j][2] for i, j in P]

    return np.column_stack([

        process.cpdist(
            ns,
            ns2,
            scorer=fuzz.ratio,
            workers=-1
        ) / 100,

        process.cpdist(
            ns,
            ns2,
            scorer=fuzz.token_set_ratio,
            workers=-1
        ) / 100,

        process.cpdist(
            ads,
            ads2,
            scorer=fuzz.ratio,
            workers=-1
        ) / 100,

        process.cpdist(
            ads,
            ads2,
            scorer=fuzz.token_set_ratio,
            workers=-1
        ) / 100,

        np.fromiter(
            (
                A[i][1] == S[j][1]
                for i, j in P
            ),
            dtype=np.float32
        ),

        np.fromiter(
            (
                A[i][2] == S[j][2]
                for i, j in P
            ),
            dtype=np.float32
        )

    ]).astype(np.float32)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

tr1 = load(B + 'train_source1.tsv')
tr2 = load(B + 'train_source2.tsv')
tr3 = load(B + 'train_source3.tsv')
gt  = load(B + 'train_ground_truth.tsv')

te1 = load(B + 'test_source1.tsv')
te2 = load(B + 'test_source2.tsv')
te3 = load(B + 'test_source3.tsv')

print("TRAIN 1:", len(tr1))
print("TRAIN 2:", len(tr2))
print("TRAIN 3:", len(tr3))
print("GROUND TRUTH:", len(gt))

print("TEST 1:", len(te1))
print("TEST 2:", len(te2))
print("TEST 3:", len(te3))

print("=" * 60)


# ============================================================
# GROUND TRUTH MAP
# ============================================================

gm = {
    r.source1_entity_id:
        set(r.matched_entity_ids.split(','))
        if r.matched_entity_ids
        else set()

    for r in gt.itertuples()
}


# ============================================================
# PREP TRAINING DATA
# ============================================================

A = prep(tr1)
S2 = prep(tr2)
S3 = prep(tr3)

Xs = []
ys = []


# ============================================================
# TRAIN MODEL
# ============================================================

for S in [S2, S3]:

    P = make_candidates(A, S)

    print(
        'TRAIN CANDIDATES:',
        len(P)
    )

    X_part = feat(A, S, P)

    y_part = [
        int(
            S[j][0] in gm[A[i][0]]
        )
        for i, j in P
    ]

    Xs.append(X_part)
    ys.extend(y_part)


X = np.vstack(Xs)
y = np.asarray(ys, np.int8)

print("=" * 60)
print("TRAINING MODEL")
print("X shape:", X.shape)
print("Positive matches:", y.sum())
print("=" * 60)


clf = LogisticRegression(
    max_iter=200,
    class_weight='balanced',
    C=2.0
)

clf.fit(X, y)

print("MODEL TRAINED")


# ============================================================
# PREP TEST DATA
# ============================================================

TA = prep(te1)
T2 = prep(te2)
T3 = prep(te3)

print("=" * 60)
print("TEST DATA PREPARED")
print("TEST SOURCE1:", len(TA))
print("=" * 60)


# ============================================================
# PREDICTION STORAGE
# ============================================================

matches = [
    set()
    for _ in TA
]

cands = [
    set()
    for _ in TA
]


# ============================================================
# PREDICT TEST DATA
# ============================================================

for S in [T2, T3]:

    P = make_candidates(TA, S)

    print(
        'TEST CANDIDATES:',
        len(P)
    )

    if not P:
        continue

    features = feat(
        TA,
        S,
        P
    )

    pr = clf.predict_proba(
        features
    )[:, 1]

    for (i, j), p in zip(P, pr):

        # Candidate
        cands[i].add(
            S[j][0]
        )

        # Predicted match
        if p >= 0.5:

            matches[i].add(
                S[j][0]
            )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

out = B + 'final_output'

shutil.rmtree(
    out,
    ignore_errors=True
)

os.makedirs(
    out,
    exist_ok=True
)

matching_file = out + '/matching_results.tsv'

with open(
    matching_file,
    'w',
    encoding='utf-8'
) as f:

    f.write(
        'source1_entity_id\tmatched_entity_ids\n'
    )

    for (sid, *_), m in zip(
        TA,
        matches
    ):

        f.write(
            sid
            + '\t'
            + ','.join(sorted(m))
            + '\n'
        )

candidate_file = out + '/candidate_pairs.tsv'

with open(
    candidate_file,
    'w',
    encoding='utf-8'
) as f:

    f.write(
        'source1_entity_id\tcandidate_entity_ids\n'
    )

    for (sid, *_), c in zip(
        TA,
        cands
    ):

        f.write(
            sid
            + '\t'
            + ','.join(sorted(c))
            + '\n'
        )


valid = {
    x[0]
    for x in T2 + T3
}

assert len(TA) == len(matches)

assert all(
    m <= valid
    for m in matches
)

assert all(
    matches[i] <= cands[i]
    for i in range(len(matches))
)


submission = pd.read_csv(
    matching_file,
    sep='\t',
    dtype=str
).fillna('')


print("=" * 60)
print("SUBMISSION VALIDATION")
print("=" * 60)

print(
    "TEST SOURCE1 ROWS:",
    len(TA)
)

print(
    "SUBMISSION ROWS:",
    len(submission)
)

print(
    "UNIQUE SOURCE1 IDS:",
    submission.source1_entity_id.nunique()
)

print(
    "DUPLICATE SOURCE1 IDS:",
    submission.source1_entity_id.duplicated().sum()
)


assert len(submission) == len(TA)

assert (
    submission.source1_entity_id.nunique()
    == len(TA)
)

assert (
    submission.source1_entity_id.duplicated().sum()
    == 0
)


print(
    "TOTAL LINKS:",
    sum(map(len, matches))
)

print(
    "SINGLETONS:",
    sum(not m for m in matches)
)

print("=" * 60)
print("MATCHING RESULTS:")
print(matching_file)
print("=" * 60)


root = B + 'final_submission_package'

shutil.rmtree(
    root,
    ignore_errors=True
)

os.makedirs(
    root + '/output',
    exist_ok=True
)

os.makedirs(
    root + '/code/business_entity_resolution/src',
    exist_ok=True
)


shutil.copy(
    matching_file,
    root + '/output/matching_results.tsv'
)

shutil.copy(
    candidate_file,
    root + '/output/candidate_pairs.tsv'
)

with open(
    root + '/code/business_entity_resolution/src/resolve.py',
    'w',
    encoding='utf-8'
) as f:

    f.write(
        '# Reproduce using the methodology in README.md '
        'and the supplied data.\n'
    )


# ============================================================
# REQUIREMENTS
# ============================================================

with open(
    root + '/code/business_entity_resolution/requirements.txt',
    'w',
    encoding='utf-8'
) as f:

    f.write(
        'pandas>=2.0\n'
        'scikit-learn>=1.2\n'
        'rapidfuzz>=3.0\n'
    )


with open(
    root + '/code/business_entity_resolution/README.md',
    'w',
    encoding='utf-8'
) as f:

    f.write(
        '# Entity Resolution\n\n'
        'Country-aware fuzzy retrieval on normalized '
        'name+address plus token blocking; logistic regression '
        'on name/address similarities and exact-field indicators. '
        'Uses only supplied data.\n'
    )


with open(
    root + '/Documentation_template.md',
    'w',
    encoding='utf-8'
) as f:

    f.write(
        '# Methodology\n\n'
        'Candidate generation uses top-100 character/string '
        'similarity retrieval plus country-aware token blocking. '
        'Features use name/address ratio, token-set ratio, and '
        'exact normalized fields. Logistic regression is trained '
        'on the supplied ground truth. No external lookup is used.\n'
    )


# ============================================================
# ZIP PACKAGE
# ============================================================

pkg = B + 'final_submission_package.zip'

if os.path.exists(pkg):
    os.remove(pkg)


with zipfile.ZipFile(
    pkg,
    'w',
    zipfile.ZIP_DEFLATED
) as z:

    for dp, _, fs in os.walk(root):

        for fn in fs:

            p = os.path.join(
                dp,
                fn
            )

            z.write(
                p,
                os.path.relpath(
                    p,
                    root
                )
            )


print("=" * 60)
print("PACKAGE CREATED:")
print(pkg)
print("=" * 60)