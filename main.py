import importlib
import os
import sys


def get_model_class():
    module_name, class_name = sys.argv[1].split(".")
    importlib.import_module("models")
    class_obj = getattr(importlib.import_module("." + module_name, "models"), class_name)
    sys.argv.pop(1)
    return class_obj()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = get_model_class()
    model.execute()
