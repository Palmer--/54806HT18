import sipaModels as sipa
import modelEvaluator as eval

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = sipa.getConvModel()
eval.evaluateModel(model, 20, "Convolutional Model")

model = sipa.getConvModelBigKernel()
eval.evaluateModel(model, 20, "Big Kernel Model")

model = sipa.getConvModelSmallKernel()
eval.evaluateModel(model, 20, "Small Kernel Model")

model = sipa.getConvModelMultiLayer()
eval.evaluateModel(model, 20, "MultiLayer Model")


