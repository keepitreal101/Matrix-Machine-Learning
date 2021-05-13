Note: The loader file has been copied from https://github.com/MichalDanielDobrzanski/DeepLearningPython.git which is an updated version of Michael Nielson's file in 
      https://github.com/mnielsen/neural-networks-and-deep-learning.git
      The data used to train the network is MNIST data. 
      
This program uses matrix multiplication over mini-batches using the library NumPy in backpropagation instead of iteration over a mini-batch to improve time-efficiency. It is a solution to one of the problems posed in Michael Neilson's Book about Deep Learning. On my local machine the program runs twice as fast as its generic counterpart. The learing rate for the program is also set a bit higher because it is adjusted down as the network reaches its most optimal parameters. 
