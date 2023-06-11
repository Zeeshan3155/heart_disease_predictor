import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s: %(message)s]')

project_name = 'heart_disease_predictor'

folder_paths = [
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/pipeline/training_pipe.py',
    f'src/{project_name}/pipeline/prediction_pipe.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/utils.py',
    f'src/{project_name}/exception/__init__.py',
    f'src/{project_name}/exception/exception.py',
    f'src/{project_name}/logging/__init__.py',
    f'src/{project_name}/logging/logger.py',
    'artfacts/',
    'data/',
    'app.py',
    'setup.py',
    'requirements.txt'
]

for path in folder_paths:
    file_path = Path(path)
    filedir, filename = os.path.split(path)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info("creating directory")

    if (not os.path.exists(path)) or (os.path.getsize(path)==0):
        with open(path,'w') as fp:
            pass

    else:
        logging.info(f"{path} already exists")
