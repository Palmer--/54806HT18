import sipaModels as sipa
import modelEvaluator as eval

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = sipa.getBaseModel()
eval.evaluateModel(model, 20, "Base Model")

model = sipa.getConvModelBigKernel()
eval.evaluateModel(model, 20, "Big Kernel Model")

model = sipa.getConvModelSmallKernel()
eval.evaluateModel(model, 20, "Small Kernel Model")

model = sipa.getConvModelBigStrides()
eval.evaluateModel(model, 20, "Big Strides Model")

model = sipa.getConvModelMultiLayer()
eval.evaluateModel(model, 20, "MultiLayer Model")

model = sipa.getConvModelNoDropout()
eval.evaluateModel(model, 20, "No dropout Model")

model = sipa.getConvModelValidPadding()
eval.evaluateModel(model, 20, "Valid padding Model")


