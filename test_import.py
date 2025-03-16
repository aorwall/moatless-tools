import importlib


def import_attribute(name):
    """Simplified version of RQ's import_attribute function"""
    name_bits = name.split(".")
    module_name_bits, attribute_bits = name_bits[:-1], [name_bits[-1]]
    module = None

    print(f"Trying to import: {name}")
    print(f"Module name bits: {module_name_bits}")
    print(f"Attribute bits: {attribute_bits}")

    while len(module_name_bits):
        try:
            module_name = ".".join(module_name_bits)
            print(f"Trying to import module: {module_name}")
            module = importlib.import_module(module_name)
            print(f"Successfully imported module: {module_name}")
            break
        except ImportError as e:
            print(f"Import error for {module_name}: {e}")
            attribute_bits.insert(0, module_name_bits.pop())

    if module is None:
        print(f"Failed to import any module for {name}")
        return None

    attribute_name = ".".join(attribute_bits)
    print(f"Looking for attribute: {attribute_name} in module {module.__name__}")

    if hasattr(module, attribute_name):
        attr = getattr(module, attribute_name)
        print(f"Found attribute {attribute_name} directly in module {module.__name__}")
        return attr

    # Try to handle nested attributes (like Class.method)
    if "." in attribute_name:
        parts = attribute_name.split(".")
        current = module
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                print(f"Could not find part {part} in {current}")
                return None
        return current

    print(f"Attribute {attribute_name} not found in module {module.__name__}")
    return None


# Test the import
func = import_attribute("moatless.evaluation.run_instance.run_instance")
if func:
    print(f"Successfully imported {func.__name__} from {func.__module__}")
else:
    print("Import failed")

# Try alternative import paths
print("\nTrying alternative import paths:")
alt_paths = ["moatless.evaluation.run_instance", "moatless.evaluation", "moatless"]

for path in alt_paths:
    print(f"\nTrying: {path}")
    module = importlib.import_module(path)
    print(f"Module {path} imported successfully")

    if path == "moatless.evaluation.run_instance":
        if hasattr(module, "run_instance"):
            func = getattr(module, "run_instance")
            print(f"Found run_instance in {path}: {func.__name__} from {func.__module__}")
        else:
            print(f"run_instance not found in {path}")
