import Data
from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.utilities           import percentError

def loadDataSet(x,y):
    dataset = ClassificationDataSet(x.shape[1], y.shape[1], nb_classes=10)
    dataset.setField('input', x)
    dataset.setField('target', y)
    return dataset

X,Y = Data.loadData("TrainBig.csv")
train = loadDataSet(X,Y)
train._convertToOneOfMany() #One vs All

#train
print "Training"
neuralnet = buildNetwork( train.indim, 20, train.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer(neuralnet, dataset=train, momentum=0.1, verbose=True, weightdecay=0.01)
#trainer.trainUntilConvergence()
trainer.trainEpochs( 20 )


#test
print "Testing"
X,Y = Data.loadData("Test.csv")
test = loadDataSet(X,Y)
test._convertToOneOfMany() #One vs All

trnresult = percentError( trainer.testOnClassData(), train['class'] )

print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult

#activate

#result = neuralnet.activateOnDataset(dataset)
#print result