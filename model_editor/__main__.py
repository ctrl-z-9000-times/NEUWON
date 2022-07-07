import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from os.path import dirname, join

root = dirname(__file__)

class gui_server(BaseHTTPRequestHandler):
 
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        try:
            send_data = open(join(root, self.path[1:])).read()
            self.send_response(200)
        except:
            send_data = "File not found"
            self.send_response(404)
        self.end_headers()
        self.wfile.write(bytes(send_data, 'utf-8'))

    def do_POST(self):
        print(dir(self))

webbrowser.open("localhost:8080/index.html")

httpd = HTTPServer(('localhost', 8080), gui_server)
httpd.serve_forever()

# TODO: Allow CLI arg containing saved file and start with that file opened?
