# simple server script for serving static web app for local testing
if __name__ == "__main__":
    import os
    import http.server
    import socketserver

    PORT = 8000
    DIRECTORY = "."

    os.chdir(DIRECTORY)

    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()