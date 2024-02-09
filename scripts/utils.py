import os


def get_checkpoints_list(path):
    ''' Returns checkpoint's list in specified folder '''
    flist = os.listdir(path+'/')
    extentions = ['.safetensors', '.ckpt']
    checkpoints = [file for file in flist if os.path.splitext(file)[1] in extentions]
    if len(checkpoints) == 0:
        return None

    return checkpoints
