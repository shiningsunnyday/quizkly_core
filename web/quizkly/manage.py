#!/usr/bin/env python
import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="Configs paths to the models",
    )
    flags = parser.parse_known_args()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quizkly.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    if len(sys.argv) != 2:
        os.environ.setdefault("MODELS_CONFIG", flags[0].config_file)
        execute_from_command_line(sys.argv[:-1])
    else:
        execute_from_command_line(sys.argv)
