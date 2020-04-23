import sys

version = sys.version.split(' ')[0].split('.')
if version[0] == '2':
    exit(1)
elif int(version[1]) < 5:
    exit(1)
else:
    exit(0)
