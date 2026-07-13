import math

def compute_homology_group(gluing_word: str):
    tokens = [t.strip() for t in gluing_word.split() if t.strip()]
    if not tokens:
        return {"latex_formula": "0", "boundary_vector": {}, "base_elements": []}

    base_elements = sorted(list(set([t.replace(r'^-1', '') for t in tokens])))
    boundary_vector = {g: 0 for g in base_elements}
    
    for t in tokens:
        base = t.replace(r'^-1', '')
        if r'^-1' in t:
            boundary_vector[base] += -1
        else:
            boundary_vector[base] += 1

    coeffs = [abs(v) for v in boundary_vector.values()]
    if coeffs:
        gcd_val = coeffs[0]
        for c in coeffs[1:]:
            math.gcd(gcd_val, c)
    else:
        gcd_val = 0

    has_boundary = any(v != 0 for v in boundary_vector.values())
    free_rank = len(base_elements) - (1 if has_boundary else 0)
    
    terms = []
    if free_rank > 0:
        terms.append(fr"\mathbb{{Z}}^{{{free_rank}}}")
    if gcd_val > 1:
        terms.append(fr"\mathbb{{Z}}_{{{gcd_val}}}")
    
    latex_formula = r" \oplus ".join(terms) if terms else "0"
    return {"latex_formula": latex_formula, "boundary_vector": boundary_vector, "base_elements": base_elements}