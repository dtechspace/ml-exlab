# © Copyright 2020, D-Tech, LLC, All Rights Reserved. 
# Version: 0.5 (initial version), 08/10/2020
# License: The use of this software program is subject to the ML-ExLab 
# license terms and conditions as defined in the LICENSE file.
# Disclaimer: This software is provided "AS IS" without warrantees.  
# D-Tech, LLC has no obligation to provide any maintenence, update 
# or support for this software.  Under no circumstances shall D-Tech,  
# LLC be liable to any parties for direct, indirect, special, incidental,
# or consequential damages, arising out of the use of this software
# and related data and documentation.
#

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
