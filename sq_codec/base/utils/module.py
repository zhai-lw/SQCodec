import importlib


def installed(module_name: str):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_name(obj=None, cls=None):
    if obj is not None:
        cls = obj.__class__
    return cls.__module__ + "." + cls.__name__


def compare_version(v1: str, v2: str) -> bool:
    """
    Compare two version strings.
    :param v1: version string 1
    :param v2: version string 2
    :return: true if v1 >= v2, else false
    """
    v1 = v1.split(".")
    v2 = v2.split(".")
    for vi1, vi2 in zip(v1, v2):
        if int(vi1) > int(vi2):
            return True
        elif int(vi1) < int(vi2):
            return False
    return len(v1) >= len(v2)
