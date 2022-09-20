import pkgutil


def get_models():
    """
    Get a list of all models in eprsim package
    """
    return [m.name for m in (pkgutil.iter_modules(__path__))]
