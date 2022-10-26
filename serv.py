import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

model_runner = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5").to_runner()

svc = bentoml.Service("mlzoom_classifier", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
def classify(user_prof):
    prediction = model_runner.predict.run(user_prof)
    print(prediction)
    result = prediction[0]
    return result
    