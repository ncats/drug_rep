import os

BASE_DIRECTORY = 'X:/Desktop - Data Drive/NCATS/src/'

def resolve_path(relative_path: str) -> str:
    '''Resolve the relative file path using os.path.abspath()'''
    return os.path.abspath(os.path.join(BASE_DIRECTORY, relative_path))