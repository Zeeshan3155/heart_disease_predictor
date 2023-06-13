import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle

def save_obj(obj_path,obj):
    try:
        dir_path = os.path.dirname(obj_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(obj_path,"wb") as fp:
            pickle.dump(obj,fp)
        
        logging.info("Object saved")

    except Exception as e:
        raise CustomException(e,sys)


def load_object(object_path):
    try:
        with open(object_path,"rb") as fp:
            return pickle.load(fp)
    except Exception as e:
        raise CustomException(e,sys)