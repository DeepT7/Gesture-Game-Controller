import os 

def print_directory_structure(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        print(dirpath)
        for dirname in dirnames:
            print(os.path.join(dirpath, dirname))
        for filename in filenames:
            print(os.path.join(dirpath, filename))

path = './'
print_directory_structure(path)