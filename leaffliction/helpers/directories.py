
import os


def is_file(full_path):
    """
    check if the full_path is a file
    """
    try:
        return os.path.isfile(full_path)
    except PermissionError:
        return False


def is_dir(full_path):
    """
    check if the full_path is a file
    """
    try:
        return os.path.isdir(full_path)
    except PermissionError:
        return False


def list_dir(dirname):
    """
    list the contents of the directory
    """
    try:
        return [name for name in os.listdir(dirname)]
    except PermissionError:
        return []


def is_last_level(dirname):
    """
    last level is a directory without any subdirectories
    """
    try:
        lst = os.listdir(dirname)
    except PermissionError:
        return True
    for name in lst:
        if is_dir(os.path.join(dirname, name)):
            return False
    return True


def is_above_current_dir(dirname):
    """
    check if the current directory in any level above the current directory
    """
    try:
        current_dir = os.path.abspath('.')
        new_dir = os.path.abspath(dirname)
        if (new_dir.startswith(current_dir)):
            return True
        return False
    except PermissionError:
        return False


def iter_images(dirname: str, directories: [str], function: callable) -> []:
    """
    for every file in the directory list matching the pattern
    '.jpg' or '.jpeg', calls function() passing the full path as argument

    function: the function to call in the files
              it must receive the full path as argument
    returns a list of returns from the function calls
    """
    ret = []
    for directory in directories:
        files = [name for name in list_dir(os.path.join(dirname, directory))
                 if is_file(os.path.join(dirname, directory, name))
                 and name.lower().endswith(('.jpg', '.jpeg'))]
        print("Processing", directory)
        for file in files:
            fullpath = os.path.join(dirname, directory, file)
            ret.append(function(fullpath))
            print('.', end='', flush=True)
        print()
    return ret


def iter_dir(dirname: str, function: callable):
    """
    recursively search for directories that are the penultimate level
    and performs the function in them.

    arguments:
    dirname: the directory to start the search
    function: the function to call in the last level directories
              it must receive dirname and directories as arguments
    returns the result of the function call
    """
    results = []
    for name in os.listdir(dirname):
        if (is_dir(os.path.join(dirname, name))
                and not name.startswith('.')
                and not is_last_level(os.path.join(dirname, name))):
            sub_dir_results = iter_dir(os.path.join(dirname, name), function)
            if sub_dir_results is not None:
                results.extend(sub_dir_results)
    directories = [
        name for name in list_dir(dirname)
        if is_dir(os.path.join(dirname, name))
        and is_last_level(os.path.join(dirname, name))]
    if directories:
        result = function(dirname, directories)
        if result is not None:
            results.append(result)
    return results


def iter_dir_image(dirname: str, function: callable):
    """
    recursively search for directories that are the penultimate level
    and performs the function to every image in them.

    arguments:
    dirname: the directory to start the search
    function: the function to call in the images
              it must receive the full path as argument
    """
    def wrapper(dirname, directories):
        return iter_images(dirname, directories, function)
    return iter_dir(dirname, wrapper)


if __name__ == '__main__':
    """
    test the iter_dir_image function, passing test() as argument, which
    prints the full path of the images and also returns a list of the full
    paths as a list
    """

    def test(fullpath):
        print(fullpath)
        return fullpath

    ret = iter_dir_image('.', test)
    print()
    print(ret)
