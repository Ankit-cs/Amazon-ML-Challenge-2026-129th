import re
import unicodedata
from rapidfuzz import fuzz

from preprocess import norm, get_tokens

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def extract_features(a, b):
    _, n1, ad1, c1 = a
    _, n2, ad2, c2 = b

    nt1 = set(n1.split())
    nt2 = set(n2.split())
    at1 = set(ad1.split())
    at2 = set(ad2.split())

    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_token = fuzz.token_set_ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0

    addr_ratio = fuzz.ratio(ad1, ad2) / 100.0
    addr_token = fuzz.token_set_ratio(ad1, ad2) / 100.0
    addr_partial = fuzz.partial_ratio(ad1, ad2) / 100.0

    name_jac = jaccard(nt1, nt2)
    addr_jac = jaccard(at1, at2)

    exact_name = 1.0 if n1 and n1 == n2 else 0.0
    exact_address = 1.0 if ad1 and ad1 == ad2 else 0.0
    country_same = 1.0 if c1 == c2 else 0.0

    name_len_ratio = min(len(n1), len(n2)) / max(len(n1), len(n2)) if n1 and n2 else 0.0
    address_len_ratio = min(len(ad1), len(ad2)) / max(len(ad1), len(ad2)) if ad1 and ad2 else 0.0

    return [
        name_ratio, name_token, name_partial,
        addr_ratio, addr_token, addr_partial,
        name_jac, addr_jac,
        exact_name, exact_address, country_same,
        name_len_ratio, address_len_ratio
    ]
