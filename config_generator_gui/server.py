from http.server import HTTPServer, CGIHTTPRequestHandler

if __name__ == '__main__':
    try:
        CGIHTTPRequestHandler.cgi_directories = ['/cgi-bin']

        httpd = HTTPServer(('', 8000),             # localhost:8000
                           CGIHTTPRequestHandler)  # CGI support.

        print(f"Running server. Use [ctrl]-c to terminate.")

        httpd.serve_forever()

    except KeyboardInterrupt:
        print(f"\nReceived keyboard interrupt. Shutting down server.")
        httpd.socket.close()
