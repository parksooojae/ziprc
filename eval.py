import re

# parse mode's after inference to find final answer
def extract_answer(text):
    if not text or not isinstance(text, str):
        return None
    m = re.search(r"\\boxed\{([^}]+)\}", text)

    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None


# normalize answer type to prevent false negatives
def _norm(s):
    if s is None:
        return None
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return s


# check correctness of model answers
# used to determine which of the two value bins (0 or 1) an answer belongs to
def reward(generated, ground_truth):
    a, b = _norm(extract_answer(generated)), _norm(ground_truth)
    if a is None or b is None:
        return 1.0 if a == b else 0.0
    if isinstance(a, float) and isinstance(b, float):
        return 1.0 if abs(a - b) < 1e-6 else 0.0
    return 1.0 if a == b else 0.0
