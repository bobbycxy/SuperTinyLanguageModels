# stlm/registry.py

REGISTRY = {
    "embedder": {},
    "core": {},
    "attention": {},
    "ffn": {},
    "head": {},
}

def register_component(category, name):
    """
    Decorator to register a class under a category and name.
    Example:
        @register_component("core", "transformer")
        class TinyTransformer(BaseCoreModel):
            ...
    """
    def decorator(cls):
        if category not in REGISTRY:
            raise ValueError(f"Unknown category {category}. Must be one of {list(REGISTRY.keys())}")
        REGISTRY[category][name] = cls
        return cls
    return decorator
