import time
import config.Config as config
import os

class Start:
    def __init__(self):
        while True:
            print('\nITP4514 Assignment - News Classification and Summarization')
            print('------ Models ------')
            print(' - Classification - ')
            print('[1] - Naive Bayes (NB)')
            print('[2] - Gated Recurrent Unit (GRU)')
            print(' - Summerization - ')
            print('[3] - Bidirectional Encoder Representations from Transformers (BERT)')
            print('[4] - T5')
            print('\n[x] - Exit')
            option = input('Select a model: ')
            if option == '1':
                self.__nb()
            elif option == '2':
                self.__gru()
            elif option == '3':
                self.__bert()
            elif option == '4':
                self.__t5()
            elif option == 'x':
                exit()
            else:
                print('Invalid input, please try again.\n')

    def __nb(self):
        print('\n - Naive Bayes (NB) - ')
        print('[1] - Start Classification')
        print('      Classify the new.')
        print('[2] - Start Model Evaluation')
        print('      Train and test the model.')
        print('[3] - Start Feature Analysis') 
        print('      Output the features of each category.')
        print('\n[b] - Back')
        option = input('Select an option: ')
        if option == '1':
            import NB_Classification as NB
            NB.NB_Classification(self.__readTestData())
        elif option == '2':
            import NB_ModelEvaluation as NB
            NB.NB_ModelEvaluation()
        elif option == '3':
            import NB_FeatureAnalysis as NB
            NB.NB_FeatureAnalysis()
        elif option == 'b':
            Start()
        else:
            print('Invalid input, please try again.')
            self.__nb()

    def __gru(self):
        print('\n - Gated Recurrent Unit (GRU) - ')
        print('[1] - Start Classification')
        print('      Classify the new.')
        print('[2] - Start Model Tuning')
        print('      Tune and test the model.')
        print('\n[b] - Back')
        option = input('Select an option: ')
        if option == '1':
            pass
        elif option == '2':
            import GRU_ModelTuning as GRU
            GRU.GRU_ModelTuning()
        elif option == 'b':
            Start()
        else:
            print('Invalid input, please try again.')
            self.__gru()

    def __bert(self):
        pass

    def __t5(self):
        pass

    def __readTestData(self):
        print('\nPlease put the test data in the [data/testData] folder.')
        input('Press enter to continue...')
        file_list = os.listdir(config.testDataFolderPath)
        files = []
        for file_name in file_list:
            file_path = os.path.join(config.testDataFolderPath, file_name)
            if file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    files.append(
                        {
                            'file_name': file_name,
                            'file_content': self.__cleaneTestData(f.read())
                        }
                    )
        
        print('\nYou have', len(files), 'test data.')
        for index, file in enumerate(files):
            print(f'  [{index + 1}] -', file['file_name'])

        while True:
            index = input('Please select a test data: ')
            if index.isdigit() and int(index) <= len(files) and int(index) > 0:
                return files[int(index) - 1]
            else:
                print('Invalid input, please try again.\n')

    def __cleaneTestData(self, input_string):
        special_chars = ['\u2013', '\u2014', '\u00ad', '\u2018', '\u2019', '\u201c', '\u201d', '\u00AD', '\n', '\t', '\r', '\f']
        for char in special_chars:
            input_string = input_string.replace(char, "")
        return input_string


if __name__ == '__main__':
    Start()