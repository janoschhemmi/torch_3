##
import os

def check_path(path):
    if os.path.exists(path):
        print("path exists")
    else:
        os.mkdir(path)
        print("path created")



