import sys

def main():
    print('Hodgkin-Huxley Demonstration.')
    print('')
    print_usage_msg = False
    if len(sys.argv) != 2:
        demo = None
    else:
        demo = sys.argv[1]

    if demo == 'propagation':
        from neuwon.examples.HH.propagation import main as demo_main
    elif demo == 'accuracy':
        from neuwon.examples.HH.accuracy import main as demo_main
    else:
        from neuwon.examples.HH.animation import main as demo_main
        if demo != 'animation':
            print_usage_msg = True
 
    if print_usage_msg:
        print('Usage: python -m neuwon.examples.HH {propagation|accuracy|animation}')
        print('')
    demo_main()

if __name__ == '__main__': main()
