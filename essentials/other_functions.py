##
import os

def check_path(path):
    if os.path.exists(path):
        print("path exists")
    else:

        try:
            os.mkdir(path)
            print("path created")

        except FileNotFoundError:
            os.makedirs(path)
            print("path created")




