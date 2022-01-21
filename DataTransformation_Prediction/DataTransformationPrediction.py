from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Logger


class dataTransformPredict:

     """
                  This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

                  Written By: iNeuron Intelligence
                  Version: 1.0
                  Revisions: None

                  """

     def __init__(self):
          self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
          self.logger = App_Logger()


     def addQuotesToStringValuesInColumn(self):

          """
                                  Method Name: addQuotesToStringValuesInColumn
                                  Description: This method replaces the missing values in columns with "NULL" to
                                               store in the table. We are using substring in the first column to
                                               keep only "Integer" data for ease up the loading.
                                               This column is anyways going to be removed during prediction.

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                                          """

          try:
               log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
               onlyfiles = [f for f in listdir(self.goodDataPath)]
               for file in onlyfiles:
                    data = pandas.read_csv(self.goodDataPath + "/" + file)
                    # list of columns with string datatype variables
                    # column = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick',
                    #           'pregnant',
                    #           'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                    #           'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured',
                    #           'TT4_measured',
                    #           'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG', 'referral_source', 'Class']

                    data['DATE'] = data["DATE"].apply(lambda x: "'" + str(x) + "'")

                    # there are "hyphen" in our column name which results in failure when inserting the column names in the table
                    # so we are changing the column names by replacing the '-'
                    # for col in data.columns:
                    #      new_col = col.replace('-', '')
                    #      data = data.rename(columns={col: new_col})
                    #
                    # for col in data.columns:
                    #      data[col] = data[col].apply(lambda x: "'" + str(x) + "'")
                    # #csv.update("'"+ csv['Wafer'] +"'")
                    # csv.update(csv['Wafer'].astype(str))
                    # csv['Wafer'] = csv['Wafer'].str[6:]
                    data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                    self.logger.log(log_file, " %s: Quotes added successfully!!" % file)

          except Exception as e:
               log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               #log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
               log_file.close()
               raise e
          log_file.close()
