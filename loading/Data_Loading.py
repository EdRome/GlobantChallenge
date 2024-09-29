class CreateTrainingBatches:
    """
    This class is used to yield an iterator of fixed size. Size is calculated during initialization and used the
    training size and the number of batches to produce the output. Number of batches is configurable
    """
    
    def __init__(self, train_set, num_batches=4):
        self.__train_set = train_set
        # Define the batch size and the number of batches
        self.__num_batches = num_batches
        self.__train_size = len(self.__train_set)
        self.__batch_size = round(self.__train_size / self.__num_batches)
        
    def create_batches(self):
        # Yield an iterator
        for i in range(self.__num_batches):
            yield self.__train_set[i*self.__batch_size:(i+1)*self.__batch_size]
