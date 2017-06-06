import os
import sys


def get_model_class():
    class_obj, class_name = None, sys.argv[1]
    try:
        import models
        class_obj = getattr(sys.modules["models"], class_name)
        sys.argv.pop(1)
    except:
        print("Model [{}] not found.\nSupported models:\n\n\t\t{}\n".format(class_name, sys.modules["models"].__all__))
        exit(1)
    return class_obj()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = get_model_class()
    model.execute()
