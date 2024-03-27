# So what should we do?

What we want is to set up a domain for RestAPI and have a Python model in the backend using Flask.
The idea was to set up a specific domain for a POST request, where that POST request would contain the Base64 encoded picture send by the mobile application.
Then the image would be decoded and put through the Python model, afterwards, the prediction of HEAVY, MEDIUM or LIGHT bleeding would be return as a 202 Response to the Request from the mobile app.

To illustrate this:
1, Mobile App ----- Base64 image --> Flask Server
2, Flask Server charges the Python model in its memory, the image is then treated by the model ---> prediction
3, Flask Server ----- Response 202 + prediction -----> Mobile App

There are some steps to this:
1, Set up a basic Flask application, and test basic POST request using Postman
2, Get the model at https://huggingface.co/Khai2002/menstrual_image_model/tree/main
3, Set up the model inside of Flask app
4, Create a domain for POST request that will receive a Base64 picture
5, Define function that will pass that image through the model and returns the prediction as a 202 Response
6, Uploading this Flask app to a public domain so that the mobile app can call it. You should use ngrok, read more about ngrok to understand how it works.

You can base of this project for what you're going to do. Just know that not everything in this project should be replicated and the most important files are kanji.py and THE END of script.js

Good luck!

