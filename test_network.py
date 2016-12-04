import network
import mnist_loader

EPOCHS = 30
MINI_BATCH_SIZE = 10
ETA = 1.0

train, validate, test = mnist_loader.load_data_wrapper()

nn = network.Network(sizes=[784, 30, 10])
nn.SGD(training_data=train, epochs=EPOCHS, mini_batch_size=MINI_BATCH_SIZE,
       eta=ETA, test_data=test)
