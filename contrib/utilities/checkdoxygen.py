#!/usr/bin/python

# run this script on all headers of deal.II to check that:
# 1. all doxygen groups have matching @{ @} pairs.
# 2. all namespaces have a documentation block.
# 
# usage:
# find include -name "*h" -print | xargs -n 1 contrib/utilities/checkdoxygen.py

import sys

args=sys.argv
args.pop(0)

filename = args[0]
f = open(filename)
lines = f.readlines()
f.close()


count = 0
lineno = 1
for l in lines:
    if "@{" in l:
            count = count + 1
    elif "@}" in l:
             count = count -1
             if (count < 0):
                 sys.exit("Error in file '%s' in line %d"%(filename,lineno));
    lineno = lineno + 1

if (count != 0):
    sys.exit("Error: missing closing braces in file '%s'"%(filename));

in_doxy_comment = False
last_left = -1
lineno = 1
for l in lines:
    line = l.strip()
    if "/**" in line:
        assert(in_doxy_comment==False)
        in_doxy_comment=True
    if "*/" in line:
        in_doxy_comment=False
        last_left = lineno
    if line.startswith("namespace ") and last_left!=lineno-1:
        if line!="namespace internal":
            print "%s: %d: Warning: undocumented namespace '%s'" % (filename, lineno, line)
    lineno = lineno + 1
    
