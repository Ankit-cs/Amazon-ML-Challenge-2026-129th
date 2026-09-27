import os
import re
import gc
import glob
import math
import unicodedata
import collections

import numpy as np
import pandas as pd

from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIG
# ============================================================

B = "./"

MAX_CANDIDATES = 60
MAX_POSTINGS = 80

TRAIN_NEGATIVES_PER_SOURCE1 = 5
TRAIN_MAX_PAIRS = 500000

TEST_BATCH = 5000
TRAIN_BATCH = 5000

# F0.5 is precision-heavy, so start conservative.
MATCH_THRESHOLD = 0.78


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_file(prefix):
    files = glob.glob(
        os.path.join(B, prefix + "*.tsv")
    )

    if not files:
        raise FileNotFoundError(
            f"Cannot find {prefix}*.tsv in {os.path.abspath(B)}"
        )

    # Prefer exact filename if present.
    exact = os.path.join(B, prefix + ".tsv")

    if exact in files:
        return exact

    # Otherwise use shortest matching name.
    files.sort(key=len)
    return files[0]


TRAIN1 = find_file("train_source1")
TRAIN2 = find_file("train_source2")
TRAIN3 = find_file("train_source3")
GROUND = find_file("train_ground_truth")

TEST1 = find_file("test_source1")
TEST2 = find_file("test_source2")
TEST3 = find_file("test_source3")


print("=" * 70)
print("FILES")
print("=" * 70)

print("TRAIN1 :", TRAIN1)
print("TRAIN2 :", TRAIN2)
print("TRAIN3 :", TRAIN3)
print("GROUND :", GROUND)
print("TEST1  :", TEST1)
print("TEST2  :", TEST2)
print("TEST3  :", TEST3)


# ============================================================
# LOAD
# ============================================================

def load(path):

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str
    ).fillna("")


# ============================================================
# NORMALIZATION
# ============================================================

def norm(s):

    s = unicodedata.normalize(
        "NFKC",
        str(s)
    ).lower()

    s = s.replace("&", " and ")

    # Common business normalization
    s = s.replace(" road ", " rd ")
    s = s.replace(" street ", " st ")
    s = s.replace(" avenue ", " ave ")
    s = s.replace(" boulevard ", " blvd ")
    s = s.replace(" corporation ", " corp ")
    s = s.replace(" incorporated ", " inc ")
    s = s.replace(" limited ", " ltd ")

    s = re.sub(
        r"[^\w\s]",
        " ",
        s
    )

    return re.sub(
        r"\s+",
        " ",
        s
    ).strip()


def prep(df):

    return [
        (
            str(r.entity_id),
            norm(r.business_name),
            norm(r.business_address),
            str(r.country).lower().strip()
        )
        for r in df.itertuples()
    ]


# ============================================================
# TOKENS
# ============================================================

STOP = {
    "the",
    "and",
    "of",
    "for",
    "near",
    "road",
    "rd",
    "street",
    "st",
    "india",
    "usa",
    "us"
}


def get_tokens(name, address):

    result = set()

    for x in (
        name.split() +
        address.split()
    ):

        if len(x) >= 3 and x not in STOP:
            result.add(x)

    return result


# ============================================================
# INDEX
# ============================================================

def build_index(S):

    index = collections.defaultdict(list)

    for j, (_, name, address, country) in enumerate(S):

        toks = get_tokens(
            name,
            address
        )

        for token in toks:

            key = (
                country,
                token
            )

            if len(index[key]) < MAX_POSTINGS:

                index[key].append(j)

    return index


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def candidates_for(
    record,
    index
):

    _, name, address, country = record

    counter = collections.Counter()

    toks = get_tokens(
        name,
        address
    )

    for token in toks:

        posting = index.get(
            (
                country,
                token
            )
        )

        if posting:

            for j in posting:

                counter[j] += 1

    if not counter:
        return []

    # More shared tokens first.
    return [
        j
        for j, _ in counter.most_common(
            MAX_CANDIDATES
        )
    ]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def features(a, b):

    _, n1, ad1, c1 = a
    _, n2, ad2, c2 = b

    nt1 = set(n1.split())
    nt2 = set(n2.split())

    at1 = set(ad1.split())
    at2 = set(ad2.split())

    name_ratio = (
        fuzz.ratio(n1, n2) / 100.0
    )

    name_token = (
        fuzz.token_set_ratio(
            n1,
            n2
        ) / 100.0
    )

    name_partial = (
        fuzz.partial_ratio(
            n1,
            n2
        ) / 100.0
    )

    addr_ratio = (
        fuzz.ratio(
            ad1,
            ad2
        ) / 100.0
    )

    addr_token = (
        fuzz.token_set_ratio(
            ad1,
            ad2
        ) / 100.0
    )

    addr_partial = (
        fuzz.partial_ratio(
            ad1,
            ad2
        ) / 100.0
    )

    name_jac = jaccard(
        nt1,
        nt2
    )

    addr_jac = jaccard(
        at1,
        at2
    )

    exact_name = (
        1.0
        if n1 and n1 == n2
        else 0.0
    )

    exact_address = (
        1.0
        if ad1 and ad1 == ad2
        else 0.0
    )

    country_same = (
        1.0
        if c1 == c2
        else 0.0
    )

    name_len_ratio = (
        min(len(n1), len(n2)) /
        max(len(n1), len(n2))
        if n1 and n2
        else 0.0
    )

    address_len_ratio = (
        min(len(ad1), len(ad2)) /
        max(len(ad1), len(ad2))
        if ad1 and ad2
        else 0.0
    )

    return [
        name_ratio,
        name_token,
        name_partial,
        addr_ratio,
        addr_token,
        addr_partial,
        name_jac,
        addr_jac,
        exact_name,
        exact_address,
        country_same,
        name_len_ratio,
        address_len_ratio
    ]


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth():

    gt = load(GROUND)

    truth = {}

    for r in gt.itertuples():

        sid = str(
            r.source1_entity_id
        )

        raw = str(
            r.matched_entity_ids
        ).strip()

        if raw:
            truth[sid] = set(
                x.strip()
                for x in raw.split(",")
                if x.strip()
            )
        else:
            truth[sid] = set()

    return truth


# ============================================================
# BUILD TRAINING DATA
# ============================================================

def build_training_pairs(
    A,
    S,
    truth,
    source_name
):

    print("=" * 70)
    print(
        "BUILDING TRAINING PAIRS:",
        source_name
    )
    print("=" * 70)

    index = build_index(S)

    sid_to_idx = {
        row[0]: j
        for j, row in enumerate(S)
    }

    X = []
    y = []

    positives = 0
    negatives = 0

    rng = np.random.default_rng(42)

    for start in range(
        0,
        len(A),
        TRAIN_BATCH
    ):

        end = min(
            start + TRAIN_BATCH,
            len(A)
        )

        print(
            f"TRAIN PAIRS "
            f"{start:,} - {end:,} / {len(A):,}"
        )

        for i in range(
            start,
            end
        ):

            a = A[i]

            true_ids = truth.get(
                a[0],
                set()
            )

            # Only labels belonging to this source.
            true_ids = {
                x
                for x in true_ids
                if x in sid_to_idx
            }

            cand = candidates_for(
                a,
                index
            )

            cand_set = set(cand)

            # ------------------------------------------------
            # FORCE TRUE POSITIVES INTO TRAINING
            # ------------------------------------------------

            for tid in true_ids:

                j = sid_to_idx.get(tid)

                if j is None:
                    continue

                X.append(
                    features(
                        a,
                        S[j]
                    )
                )

                y.append(1)

                positives += 1

            # ------------------------------------------------
            # HARD NEGATIVES
            # ------------------------------------------------

            negs = []

            for j in cand:

                if S[j][0] not in true_ids:
                    negs.append(j)

            if len(negs) > TRAIN_NEGATIVES_PER_SOURCE1:

                negs = list(
                    rng.choice(
                        negs,
                        TRAIN_NEGATIVES_PER_SOURCE1,
                        replace=False
                    )
                )

            for j in negs:

                X.append(
                    features(
                        a,
                        S[j]
                    )
                )

                y.append(0)

                negatives += 1

            if len(y) >= TRAIN_MAX_PAIRS:
                break

        if len(y) >= TRAIN_MAX_PAIRS:
            break

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int8
    )

    print(
        "TRAIN FEATURES:",
        X.shape
    )

    print(
        "POSITIVES:",
        int((y == 1).sum())
    )

    print(
        "NEGATIVES:",
        int((y == 0).sum())
    )

    del index

    gc.collect()

    return X, y


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

tr1_df = load(TRAIN1)
tr2_df = load(TRAIN2)
tr3_df = load(TRAIN3)

gt = load_ground_truth()

print(
    "TRAIN 1:",
    len(tr1_df)
)

print(
    "TRAIN 2:",
    len(tr2_df)
)

print(
    "TRAIN 3:",
    len(tr3_df)
)

print(
    "GROUND TRUTH:",
    len(gt)
)

A_train = prep(tr1_df)
S2_train = prep(tr2_df)
S3_train = prep(tr3_df)

del tr1_df
del tr2_df
del tr3_df

gc.collect()


# ============================================================
# TRAIN MODEL
# ============================================================

X1, y1 = build_training_pairs(
    A_train,
    S2_train,
    gt,
    "TRAIN SOURCE 2"
)

X2, y2 = build_training_pairs(
    A_train,
    S3_train,
    gt,
    "TRAIN SOURCE 3"
)

X_train = np.vstack(
    [
        X1,
        X2
    ]
)

y_train = np.concatenate(
    [
        y1,
        y2
    ]
)

del X1
del X2
del y1
del y2

gc.collect()


print("=" * 70)
print("FITTING LOGISTIC REGRESSION")
print("=" * 70)

clf = LogisticRegression(
    max_iter=500,
    C=3.0,
    class_weight="balanced",
    solver="lbfgs"
)

clf.fit(
    X_train,
    y_train
)

print("MODEL TRAINED")

del X_train
del y_train

gc.collect()


# ============================================================
# RELEASE TRAIN SOURCE 2/3
# ============================================================

del S2_train
del S3_train

gc.collect()


# ============================================================
# LOAD TEST SOURCE 1
# ============================================================

print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

te1_df = load(TEST1)
te2_df = load(TEST2)
te3_df = load(TEST3)

print(
    "TEST 1:",
    len(te1_df)
)

print(
    "TEST 2:",
    len(te2_df)
)

print(
    "TEST 3:",
    len(te3_df)
)

A_test = prep(te1_df)
S2_test = prep(te2_df)
S3_test = prep(te3_df)

del te1_df
del te2_df
del te3_df

gc.collect()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUT = os.path.join(
    B,
    "final_output"
)

os.makedirs(
    OUT,
    exist_ok=True
)

candidate_file = os.path.join(
    OUT,
    "candidate_pairs.tsv"
)

matching_file = os.path.join(
    OUT,
    "matching_results.tsv"
)


# ============================================================
# CREATE TEMP MATCH FILES
# ============================================================

tmp2 = os.path.join(
    OUT,
    "_source2_predictions.tsv"
)

tmp3 = os.path.join(
    OUT,
    "_source3_predictions.tsv"
)


# ============================================================
# PROCESS TEST SOURCE
# ============================================================

def process_test_source(
    A,
    S,
    source_name,
    tmp_file,
    candidate_writer
):

    print("=" * 70)
    print(
        "PROCESSING",
        source_name
    )
    print("=" * 70)

    index = build_index(S)

    total_candidates = 0
    total_matches = 0

    with open(
        tmp_file,
        "w",
        encoding="utf-8"
    ) as pred:

        pred.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        for start in range(
            0,
            len(A),
            TEST_BATCH
        ):

            end = min(
                start + TEST_BATCH,
                len(A)
            )

            print(
                f"{source_name}: "
                f"{start:,} - {end:,} / "
                f"{len(A):,}"
            )

            for i in range(
                start,
                end
            ):

                a = A[i]

                cand = candidates_for(
                    a,
                    index
                )

                total_candidates += len(
                    cand
                )

                # Candidate file
                candidate_writer.write(
                    a[0]
                    + "\t"
                    + ",".join(
                        S[j][0]
                        for j in cand
                    )
                    + "\n"
                )

                if not cand:
                    pred.write(
                        a[0] + "\t\n"
                    )
                    continue

                rows = []

                for j in cand:

                    rows.append(
                        features(
                            a,
                            S[j]
                        )
                    )

                X = np.asarray(
                    rows,
                    dtype=np.float32
                )

                probabilities = clf.predict_proba(
                    X
                )[:, 1]

                matched = []

                for j, p in zip(
                    cand,
                    probabilities
                ):

                    if p >= MATCH_THRESHOLD:

                        matched.append(
                            S[j][0]
                        )

                matched = sorted(
                    set(matched)
                )

                total_matches += len(
                    matched
                )

                pred.write(
                    a[0]
                    + "\t"
                    + ",".join(matched)
                    + "\n"
                )

    print(
        source_name,
        "CANDIDATES:",
        total_candidates
    )

    print(
        source_name,
        "MATCHES:",
        total_matches
    )

    del index
    gc.collect()


# ============================================================
# GENERATE CANDIDATES + PREDICTIONS
# ============================================================

print("=" * 70)
print("GENERATING TEST CANDIDATES")
print("=" * 70)

with open(
    candidate_file,
    "w",
    encoding="utf-8"
) as candidate_writer:

    candidate_writer.write(
        "source1_entity_id\tcandidate_entity_ids\n"
    )

    process_test_source(
        A_test,
        S2_test,
        "TEST SOURCE 2",
        tmp2,
        candidate_writer
    )

    # Important:
    # S2 candidates are already written.
    # Source 3 gets appended to same logical row later
    # through an in-memory merge of only final prediction IDs.

# ============================================================
# PROCESS SOURCE 3 INTO SEPARATE CANDIDATE FILE
# ============================================================

tmp_candidates3 = os.path.join(
    OUT,
    "_source3_candidates.tsv"
)

with open(
    tmp_candidates3,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "source1_entity_id\tcandidate_entity_ids\n"
    )

    # We need predictions separately.
    process_test_source(
        A_test,
        S3_test,
        "TEST SOURCE 3",
        tmp3,
        f
    )


# ============================================================
# IMPORTANT:
# REBUILD FINAL CANDIDATE FILE FROM SOURCE2 + SOURCE3
# WITHOUT KEEPING 100M+ IDs IN RAM.
# ============================================================

print("=" * 70)
print("BUILDING FINAL CANDIDATE FILE")
print("=" * 70)


# The previous candidate file contains S2 only.
# Rewrite it using streaming merge.

candidate_s2 = candidate_file
candidate_s3 = tmp_candidates3

candidate_final_tmp = os.path.join(
    OUT,
    "_candidate_final.tsv"
)


def merge_candidate_files(
    file1,
    file2,
    output
):

    with open(
        file1,
        "r",
        encoding="utf-8"
    ) as f1, open(
        file2,
        "r",
        encoding="utf-8"
    ) as f2, open(
        output,
        "w",
        encoding="utf-8"
    ) as out:

        f1.readline()
        f2.readline()

        out.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        while True:

            l1 = f1.readline()
            l2 = f2.readline()

            if not l1 and not l2:
                break

            sid1, ids1 = l1.rstrip("\n").split(
                "\t",
                1
            )

            sid2, ids2 = l2.rstrip("\n").split(
                "\t",
                1
            )

            if sid1 != sid2:

                raise RuntimeError(
                    "Candidate S1 IDs differ: "
                    + sid1
                    + " vs "
                    + sid2
                )

            merged = set()

            if ids1:
                merged.update(
                    x for x in ids1.split(",")
                    if x
                )

            if ids2:
                merged.update(
                    x for x in ids2.split(",")
                    if x
                )

            out.write(
                sid1
                + "\t"
                + ",".join(
                    sorted(merged)
                )
                + "\n"
            )


merge_candidate_files(
    candidate_s2,
    candidate_s3,
    candidate_final_tmp
)

os.replace(
    candidate_final_tmp,
    candidate_file
)


# ============================================================
# MERGE PREDICTIONS
# ============================================================

print("=" * 70)
print("MERGING FINAL PREDICTIONS")
print("=" * 70)


def merge_prediction_files(
    file1,
    file2,
    output
):

    rows = 0
    links = 0

    with open(
        file1,
        "r",
        encoding="utf-8"
    ) as f1, open(
        file2,
        "r",
        encoding="utf-8"
    ) as f2, open(
        output,
        "w",
        encoding="utf-8"
    ) as out:

        f1.readline()
        f2.readline()

        out.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        while True:

            l1 = f1.readline()
            l2 = f2.readline()

            if not l1 and not l2:
                break

            if not l1 or not l2:
                raise RuntimeError(
                    "Prediction files have "
                    "different row counts."
                )

            sid1, ids1 = l1.rstrip(
                "\n"
            ).split(
                "\t",
                1
            )

            sid2, ids2 = l2.rstrip(
                "\n"
            ).split(
                "\t",
                1
            )

            if sid1 != sid2:

                raise RuntimeError(
                    "Prediction S1 IDs differ: "
                    + sid1
                    + " vs "
                    + sid2
                )

            merged = set()

            if ids1:
                merged.update(
                    x for x in ids1.split(",")
                    if x
                )

            if ids2:
                merged.update(
                    x for x in ids2.split(",")
                    if x
                )

            links += len(
                merged
            )

            out.write(
                sid1
                + "\t"
                + ",".join(
                    sorted(merged)
                )
                + "\n"
            )

            rows += 1

            if rows % 100000 == 0:

                print(
                    "Merged:",
                    f"{rows:,}"
                )

    return rows, links


rows, links = merge_prediction_files(
    tmp2,
    tmp3,
    matching_file
)


# ============================================================
# VALIDATION
# ============================================================

print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

expected = len(
    A_test
)

print(
    "EXPECTED S1:",
    expected
)

print(
    "SUBMISSION ROWS:",
    rows
)

print(
    "TOTAL MATCH LINKS:",
    links
)

assert rows == expected


# ============================================================
# CHECK DUPLICATE S1
# ============================================================

seen = set()

duplicates = 0

with open(
    matching_file,
    "r",
    encoding="utf-8"
) as f:

    f.readline()

    for line in f:

        sid = line.split(
            "\t",
            1
        )[0]

        if sid in seen:
            duplicates += 1

        seen.add(sid)


print(
    "UNIQUE S1:",
    len(seen)
)

print(
    "DUPLICATE S1:",
    duplicates
)

assert len(seen) == expected
assert duplicates == 0


# ============================================================
# CLEAN TEMP FILES
# ============================================================

for p in [
    tmp2,
    tmp3,
    tmp_candidates3
]:

    try:
        os.remove(p)
    except FileNotFoundError:
        pass


gc.collect()


# ============================================================
# DONE
# ============================================================

print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "MATCHING:",
    matching_file
)

print(
    "CANDIDATES:",
    candidate_file
)

print(
    "Rows:",
    rows
)

print(
    "Expected:",
    expected
)

print("=" * 70)