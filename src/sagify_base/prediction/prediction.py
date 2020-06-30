import os
import joblib

_MODEL_PATH = os.path.join('/opt/ml/', 'model')  # Path where all your model(s) live in

class ModelService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            with open(os.path.join(_MODEL_PATH,'decision-tree-model.pkl'),'rb') as inp:
                cls.model = joblib.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        clf = cls.get_model()
        return clf.predict(input)


def predict(json_input):
    """
    Prediction given the request input
    :param json_input: [dict], request input
    :return: [dict], prediction
    """

    # TODO Transform json_input and assign the transformed value to model_input
    model_input = json_input['features']
    prediction = ModelService.predict(model_input)
    result = {'result': prediction[0]}
    print(result)
    return result
