import os

_directory_stack =[]

def pushd(target):
    oldpath = os.getcwd()
    newpath = os.path.abspath(target) 
    if not newpath:
        print("Cant cd to new path %s" % (target))

    _directory_stack.append(oldpath)
    if not os.path.exists(newpath):
        print("Making directory: %s" % (newpath))
        os.mkdir(newpath)
    os.chdir(newpath)
    print("Current directory: %s" % (os.getcwd()) )

def popd():
    current_top = _directory_stack.pop()
    if current_top:
        print("Changing directory to %s" % (os.getcwd()) )
        os.chdir(current_top)

if __name__ == "__main__":
    print("hworld") 
    pushd("./aatest") 
    pushd("./aatest") 
    popd() 
    popd() 




