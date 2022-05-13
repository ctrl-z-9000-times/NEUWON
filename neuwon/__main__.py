import argparse
parser = argparse.ArgumentParser(prog="python -m neuwon")
parser.add_argument('model_file', type=str, nargs='?')
args = parser.parse_args()

from neuwon.gui.model_editor.model_editor import ModelEditor
ModelEditor(args.model_file).run()
