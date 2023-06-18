from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

'''Initiating data ingestion'''
data_ingestion = DataIngestion()
train_path, test_path, raw_data_path = data_ingestion.initiate_data_ingestion()


'''Initiating data transformation'''
transformer = DataTransformation(raw_data_path)
train_arr, test_arr = transformer.initiate_data_transformation(train_path,test_path)

'''Initiating model trainer'''
model_trainer = ModelTrainer()
model = model_trainer.initiate_model_trainer(train_arr,test_arr)
