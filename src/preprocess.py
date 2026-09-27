import re
import unicodedata

STOP = {
    "the", "and", "of", "for", "near", "road", "rd", "street", "st", 
    "india", "usa", "us"
}

def norm(s):
    """Cleans and standardizes a business string."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("&", " and ")
    
    # Common business normalization
    s = s.replace(" road ", " rd ")
    s = s.replace(" street ", " st ")
    s = s.replace(" avenue ", " ave ")
    s = s.replace(" boulevard ", " blvd ")
    s = s.replace(" corporation ", " corp ")
    s = s.replace(" incorporated ", " inc ")
    s = s.replace(" limited ", " ltd ")
    
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def get_tokens(name, address):
    """Extracts valid tokens from normalized strings."""
    result = set()
    for x in (name.split() + address.split()):
        if len(x) >= 3 and x not in STOP:
            result.add(x)
    return result
