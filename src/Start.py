import time

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
        print('      Classify the new(s).')
        print('[2] - Start Model Evaluation')
        print('      Train and test the model.')
        print('[3] - Start Feature Analysis') 
        print('      Output the features of each category.')
        print('\n[b] - Back')
        option = input('Select an option: ')
        if option == '1':
            import NB_Classification as NB
        elif option == '2':
            import NB_ModelEvaluation as NB
            NB.NB_ModelEvaluation()
        elif option == '3':
            import NB_FeatureAnalysis as NB
            NB.NB_FeatureAnalysis()
        elif option == 'b':
            Start()
        else:
            print('Invalid input, please try again.\n')
            self.__nb()

    def __gru(self):
        pass

    def __bert(self):
        pass

    def __t5(self):
        pass


if __name__ == '__main__':
    Start()