# Mask RCNN Api
A simple Flask web api to host a Mask RCNN model and do predictions. 

## How to use
In `app.py`, configure your model by replacing the value of the model's name with yours. Copy your model in the `models
` folder.

Post a base64 encoded image to the url and it will return a json array with the prediction information.