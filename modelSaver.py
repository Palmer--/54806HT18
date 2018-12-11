import keras
from keras.models import model_from_json

savePath = "E:\\KerasModels\\"

def SaveModel(model : keras.models.Model, name : str):
    destination = savePath + name
    jsonModel = model.to_json()
    with open(destination + ".json", 'w') as file:
        file.write(jsonModel)
    model.save_weights(destination + ".h5")

def LoadModel(name : str):
    source = savePath + name
    file = open(source + ".json", 'r')
    fileData = file.read()
    model = model_from_json(fileData)
    model.load_weights(source + ".h5")
    return model
