import mlflow.pyfunc
import os
import pickle
import sklearn
import pandas as pd
import cloudpickle
from sklearn.ensemble import RandomForestRegressor
import shutil

# Define the model class
class WineQualityRegressionPyfunc(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        print(os.path.join(os.path.dirname(__file__)))
        with open(context.artifacts["model"], 'rb') as file:
            pickle_model = pickle.load(file)
        self.model = pickle_model

        with open(context.artifacts["preprocessor_X"], 'rb') as f:
                preprocessor_X = cloudpickle.load(f)
        self.preprocessor_X = preprocessor_X

        with open(context.artifacts["preprocessor_Y"], 'rb') as f:
                preprocessor_Y = cloudpickle.load(f)
        self.preprocessor_Y = preprocessor_Y
                
    def predict(self, context, model_input):
        project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 
        src_folder = "/code/"
        
        tmp_folder = '/tmp/'
        os.mkdir(project_dir + tmp_folder)
        model_input = pd.DataFrame(model_input)
        pred_X = pd.DataFrame(self.preprocessor_X.transform(model_input), columns = model_input.columns)
        pred_Y_transform = self.model.predict(pred_X)
        pred_Y = self.preprocessor_Y.inverse_transform(pred_Y_transform.reshape(1, -1))

        shutil.rmtree(project_dir + tmp_folder)   

        return pred_Y
        
