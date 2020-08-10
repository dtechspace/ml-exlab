#!/usr/bin/env python

import cgi, cgitb 

form = cgi.FieldStorage()
print("Content-type: text/html\n\n")
print(form["config"].value)
