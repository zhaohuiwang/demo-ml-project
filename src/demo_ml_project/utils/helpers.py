

# src/demo_ml_project/utils/helpers.py

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten a nested dictionary (from Pydantic model_dump) into flat keys like 'training.batch_size'.
    Handles lists by indexing: data.cat_cols.0, data.cat_cols.1, etc.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            for i, val in enumerate(v):
                items.extend(flatten_dict({str(i): val}, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)