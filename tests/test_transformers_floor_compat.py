# In tests/test_transformers_floor_compat.py

import ast

def _is_transformer_model_load(func_node: ast.AST) -> bool:
    """
    Checks if a function call node is a transformer model or processor load site
    using from_pretrained, matching AutoModel*, AutoProcessor, and concrete
    model task classes (*ForCausalLM, *ForConditionalGeneration, *ForSequenceClassification, etc.).
    """
    if isinstance(func_node, ast.Attribute) and func_node.attr == "from_pretrained":
        value = func_node.value
        # Reconstruct dotted name
        parts = []
        curr = value
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
        name = ".".join(reversed(parts))
        basename = parts[0] if parts else ""

        # Match AutoModel*, AutoProcessor, or concrete task model suffixes
        if (basename.startswith("AutoModel") or 
            basename == "AutoProcessor" or
            basename.endswith("ForCausalLM") or
            basename.endswith("ForConditionalGeneration") or
            basename.endswith("ForSequenceClassification") or
            basename.endswith("Model") and "Transformer" in basename):
            return True
    return False