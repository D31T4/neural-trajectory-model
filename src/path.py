from pathlib import Path

# path to project root
ROOT = Path(__file__).parent.parent.absolute()

def default_shanghai_dataset_path():
    '''
    path to shanghai dataset
    '''
    return str(ROOT.joinpath('data/shanghai'))

def default_tdrive_dataset_path():
    '''
    path to T-drive dataset
    '''
    return str(ROOT.joinpath('data/tdrive'))