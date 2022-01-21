import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation



class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            #code change
            # wafer_names=data['Wafer']
            # data=data.drop(labels=['Wafer'],axis=1)

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data = preprocessor.dropUnnecessaryColumns(data,['DATE','Precip','WETBULBTEMPF','DewPointTempF','StationPressure'])

            # replacing '?' values with np.nan as discussed in the EDA part

            data = preprocessor.replaceInvalidValuesWithNull(data)

            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)

            #scale the prediction data
            data_scaled = pandas.DataFrame(preprocessor.standardScalingData(data),columns=data.columns)

            #data=data.to_numpy()
            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')

            ##Code changed
            #pred_data = data.drop(['Wafer'],axis=1)
            clusters=kmeans.predict(data_scaled)#drops the first column for cluster prediction
            data_scaled['clusters']=clusters
            clusters=data_scaled['clusters'].unique()
            result=[] # initialize blank list for storing predicitons
            # with open('EncoderPickle/enc.pickle', 'rb') as file: #let's load the encoder pickle file to decode the values
            #     encoder = pickle.load(file)

            for i in clusters:
                cluster_data= data_scaled[data_scaled['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                for val in (model.predict(cluster_data.values)):
                    result.append(val)
            result = pandas.DataFrame(result,columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path

            # old code
            # i=0
            # for row in data:
            #     cluster_number=kmeans.predict([row])
            #     model_name=file_loader.find_correct_model_file(cluster_number[0])
            #
            #     model=file_loader.load_model(model_name)
            #     #row= sparse.csr_matrix(row)
            #     result=model.predict([row])
            #     if (result[0]==-1):
            #         category='Bad'
            #     else:
            #         category='Good'
            #     self.predictions.write("Wafer-"+ str(wafer_names[i])+','+category+'\n')
            #     i=i+1
            #     self.log_writer.log(self.file_object,'The Prediction is :' +str(result))
            # self.log_writer.log(self.file_object,'End of Prediction')
            #print(result)




