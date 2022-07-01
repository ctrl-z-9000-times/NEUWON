import webbrowser
from os.path import dirname, join

website_path = join(dirname(__file__), 'index.html')
webbrowser.open(website_path)

# TODO: Allow CLI arg containing saved file and start with that file opened?
