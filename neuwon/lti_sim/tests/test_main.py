import subprocess
import os

def test_nav11():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'Nav11.mod')
    output_file = os.path.join(directory, 'tmp.cpp')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['python', '-m', 'lti_sim',
            test_file, '-i', 'v', '-120', '120',
            '-t', '0.1', '-c', '37',
            '--verbose',
            '-o', output_file],
            check=True,)
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)

def test_ampa():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'ampa13.mod')
    output_file = os.path.join(directory, 'tmp.cu')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['python', '-m', 'lti_sim',
            test_file, '-i', 'C', '0', '1e3', '--log',
            '-t', '0.1', '-c', '37',
            '-f32', '--target', 'cuda',
            '--verbose',
            '-o', output_file],
            check=True,)
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)

def test_nmda():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'NMDA.mod')
    output_file = os.path.join(directory, 'tmp.cu')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['python', '-m', 'lti_sim', test_file,
            '-t', '0.1',
            # Test default inputs.
            '--target', 'cuda',
            '-e', '1e-3',
            '-vv',
            '-o', output_file],
            check=True,)
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)
