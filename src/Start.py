import time

class Start:
    def __init__(self):
        while True:
            print('\nITP4514 Assignment - News Classification and Summarization')
            print('------ Models ------')
            print(' - Classification - ')
            print('[0] - Naive Bayes (NB)')
            print('[1] - Gated Recurrent Unit (GRU)')
            print(' - Summerization - ')
            print('[2] - Bidirectional Encoder Representations from Transformers (BERT)')
            print('[3] - T5')
            print('\n[4] - Exit')
            option = input('Select a model: ')
            if option == '0':
                self.__nb()
            elif option == '1':
                self.__gru()
            elif option == '2':
                self.__bert()
            elif option == '3':
                self.__t5()
            elif option == '4':
                exit()
            else:
                print('Invalid input, please try again.\n')

    def __nb(self):
        print('\n - Naive Bayes (NB) - ')
        print('[0] - Start Classification')
        print('      Classify the new(s).')
        print('[1] - Start Model Evaluation')
        print('      Train and test the model.')
        print('[2] - Start Feature Analysis') 
        print('      Output the features of each category.')
        print('\n[3] - Back')
        option = input('Select an option: ')
        if option == '0':
            import NB_Classification as NB
        elif option == '1':
            import NB_ModelEvaluation as NB
            NB.NB_ModelEvaluation()
        elif option == '2':
            pass
        elif option == '3':
            return True
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