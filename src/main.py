# main app
from dataHelpers import PreprocessData
from train_model import excecuteTrainning
from predict_model import excecuteInference
from metrics import excecuteModelQuality
import os
import datetime

def searchDataDates():
    '''
    Funcion para ir a buscar en los archivos de datos/base de datos el rango de fechas diponible
    para que el usuario decida hasta que fecha entrenar la primera vez.
    '''
    
    return True



def main():

    #initialize the state of the app
    ini_state = 0
    stoper = 0

    while stoper != 1:

        # check the state of the app to ask for the time window.
        if ini_state == 0:

            while True:
                try:
                    strInput = input('Enter the end date for trainning data dd/mm/YYYY')
                    end_day = datetime.datetime.strptime(strInput, '%d/%m/%Y')
                except:
                    print('Please use the proper format dd/mm/YYYY')

        
        else:

            adding_months = datetime.timedelta(nextMonths*365/12)
            
            end_day = end_day + adding_months


        # Preprocess the raw data
        '''
        Deberia devolver algun identificador con el cual poder ir a buscar el dataset correcto
        a la hora de realizar el entrenamiento
        '''
        print('Starting the preprocessing of the data.')
        idDataset = PreprocessData()
        print('End the preprocessing of the data.')


        # Check for the last model available
        print('Searching for a model/s')
        models_path = 'ml-ops-project\models' 
        models_extension = '.pickle'  

        files = os.listdir(models_path)

        if models_extension in files:
            print(f"Model/s found in the directory '{models_path}'")

            idModel = models_path

            # last model found, do inference with the new data
            '''
                Deberia devolver algun identificador con el cual poder ir a buscar el dataset 
                con las predicciones del modelo para calcular metricas

                VER COMO CAPTURAR EL TRAINSET de cada modelo para poder separar
                los datos conocidos por el modelo por aquellos que el modelo nunca vio.
                -> OOJO CON LOS idDataset
            '''
            idOutput = excecuteInference(idDataset, idModel)

        else:
            print(f"Model/s not found in the directory '{models_path} excecuting Trainning'")

            # model not found, excecute trainning
            print('Starting new model trainning')
            idModel, idTestSet = excecuteTrainning(idDataset)
            print('End new model trainning')

            # Inferencing with the model trained
            print('Doing Inferencing')
            idOutput = excecuteInference(idTestSet, idModel)
            print('Inferencing Done')

        # Calculating metrics
        metrics = excecuteModelQuality(idOutput)


        # Model quality
        while True:

            if metrics['ROC'] > 80 and metrics['accuracy'] > 70 and metrics['recall'] > 0.8 :
                
                print('Model quality test successfuly done!')

                # Exit Loop
                break

            else:
                print('Model quality not accepted, Trainning a new one!')

                print('Starting new model trainning')
                idModel, idTestSet = excecuteTrainning(idDataset)
                print('End new model trainning')


                print('Doing Inferencing')
                idOutput = excecuteInference(idTestSet, idModel)
                print('Inferencing Done')

                # Calculating metrics
                metrics = excecuteModelQuality(idOutput)


        # Ask the user if we add a new month in advance of data
        stoper = input('If you want to end the script enter 1, in any other case put any caracter')

        if stoper != 1:
            try:
                nextMonths = int(input('Enter an integer representing the amouth of months added in the next round'))

            except:
                while type(nextMonths) != int:
                    try:
                        nextMonths = int(input('Enter an integer representing the amouth of months added in the next round'))
                    except:
                        print('Please enter an integer, not any other caracter')



