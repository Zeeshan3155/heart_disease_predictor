import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s: %(message)s]')

project_name = 'heart_disease_predictor'

folder_paths = [
    f'src/__init__.py',
    f'src/components/__init__.py',
    f'src/pipeline/__init__.py',
    f'src/pipeline/training_pipe.py',
    f'src/pipeline/prediction_pipe.py',
    f'src/utils/__init__.py',
    f'src/utils/utils.py',
    f'src/exception/__init__.py',
    f'src/exception/exception.py',
    f'src/logging/__init__.py',
    f'src/logging/logger.py',
    'artifacts/',
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
