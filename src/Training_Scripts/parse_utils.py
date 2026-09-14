import re
import ast
import numpy as np

# Reusable parser for model completions. Defined separately to avoid importing
# heavy training dependencies when only parsing is needed.

def parse_completion_to_coord(response_text: str, use_regex = False) -> np.ndarray | None:
    """
    Parses a string to find and extract a 2D coordinate pair.
    Returns a numpy array on success, or None on failure.
    """
    if use_regex:
        try:
            match = re.search(r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]', response_text)
            if match and np.all(np.isfinite(match)):
                x = float(match.group(1))
                y = float(match.group(2))
                output = 6 * np.array([x, y])
                if np.any(np.abs(output) > 6):
                    return None
                else:
                    return output
            else:
                return None
        except (ValueError, IndexError):
            return None
    else:
        if len(response_text) > 20:
            return None
        try:
            match = np.array(ast.literal_eval(response_text))

            if match.shape == (2,) and np.all(np.isfinite(match)):
                output = 6 * match.astype(float)
                if np.any(np.abs(output) > 6):
                    return None
                else:
                    return output
            else:
                return None
        except Exception:
            return None
