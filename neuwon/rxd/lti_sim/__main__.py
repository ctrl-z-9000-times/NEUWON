from neuwon.rxd.lti_sim import main, LinearInput, LogarithmicInput
import argparse
import numpy as np

parser = argparse.ArgumentParser(prog='lti_sim',
        description="Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.",)
parser.add_argument('nmodl_filename',
        metavar='NMODL_FILE',
        help="")
params = parser.add_argument_group('model parameters')
params.add_argument('-t', '--time_step', type=float, required=True,
        help="")
params.add_argument('-c', '--celsius', type=float, default=37.0,
        help="default: 37°")
inputs = parser.add_argument_group('input specification')
inputs.add_argument('-i', '--input', action='append', default=[],
        nargs=3, metavar=('NAME', 'MIN', 'MAX'),
        help="")
inputs.add_argument('--log', nargs='?', action='append', default=[],
        metavar='INPUT',
        help="scale the input logarithmically")
inputs.add_argument('--initial', nargs=2, action='append', default=[],
        metavar=('INPUT', 'VALUE'),
        help="default: the input's minimum value")
sim = parser.add_argument_group('simulator arguments')
sim.add_argument('-e', '--error', type=float, default=1e-4,
        help="maximum error per time step. default: 10^-4")
sim.add_argument('--target', choices=['host','cuda'], default='host',
        help="default: host")
sim.add_argument('-f', '--float', choices=['32','64'], default='64',
        help="default: 64")
sim.add_argument('-o', '--output', type=str, default=True,
        metavar='FILE',
        help="")
parser.add_argument('--plot', action='store_true',
        help="show the matrix")
parser.add_argument('-v', '--verbose', action='count', default=0,
        help="show diagnostic information, give twice for trace info")
args = parser.parse_args()

if   args.float == '32': float_dtype = np.float32
elif args.float == '64': float_dtype = np.float64

# Gather & organize all information about the inputs.
inputs = {}
for (name, minimum, maximum) in args.input:
    inputs[name] = [LinearInput, [name, minimum, maximum, None]]
for name in args.log:
    if name is None:
        if len(inputs) == 1:
            name = next(iter(inputs))
        else:
            parser.error(f'Argument "--log" must specify which input it refers to.')
    elif name not in inputs:
        parser.error(f'Argument "--log {name}" does not match any input name.')
    inputs[name][0] = LogarithmicInput
for name, initial_value in args.initial:
    if name not in inputs:
        parser.error(f'Argument "--initial {name}" does not match any input name.')
    inputs[name][1][3] = float(initial_value)
# Create the input data structures.
inputs = [input_type(*args) for (input_type, args) in inputs.values()]

main(args.nmodl_filename, inputs, args.time_step, args.celsius,
     error=args.error, target=args.target, float_dtype=float_dtype,
     outfile=args.output, verbose=args.verbose, plot=args.plot,)
