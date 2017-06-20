import json
import logging
import os

import sys
from pprint import pprint

logger = logging.info
err = logging.error


def setup_from_args_file(file):
    if not file:
        return
    json_dict = json.load(open(file, encoding="utf-8"))
    args = [sys.argv[0]]
    for k, v in json_dict.items():
        args.append("--{}".format(k))
        args.append(str(v))
    sys.argv = args.copy() + sys.argv[1:]


def save_args(args):
    """
    save all arguments.
    """
    save_obj_to_json(args.weight_path, vars(args), filename="args.json")
    pprint(vars(args), indent=4)


def save_obj_to_json(path, obj, filename):
    if not os.path.exists(path):
        os.mkdir(path)
    file = os.path.join(path, filename)
    with open(file, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, sort_keys=True, indent=4)
