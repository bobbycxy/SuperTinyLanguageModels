#stlm/models/__init__.py
import importlib, pkgutil, pathlib

def _import_all_model_submodules():
    """
    Recursively import all Python submodules under stlm/models so that
    components are registered via @register_component decorators.
    """
    package_dir = pathlib.Path(__file__).resolve().parent  # stlm/models
    prefix_base = "stlm.models"

    # Collect all directories under stlm/models
    subdirs = [str(package_dir)] + [
        str(p) for p in package_dir.rglob("*") if p.is_dir()
    ]

    imported = []
    for d in subdirs:
        # Build import prefix like 'stlm.models.embedders.' or 'stlm.models.cores.components.'
        rel_path = pathlib.Path(d).relative_to(package_dir)
        prefix = (
            f"{prefix_base}." + ".".join(rel_path.parts) + "."
            if rel_path.parts else f"{prefix_base}."
        )

        # Import all Python modules found within this directory
        for module_info in pkgutil.walk_packages([d], prefix=prefix):
            try:
                importlib.import_module(module_info.name)
                imported.append(module_info.name)
            except Exception as e:
                print(f"[WARN] Skipping module {module_info.name}: {e}")
    
    return imported


# Automatically run this once on import
_import_all_model_submodules()