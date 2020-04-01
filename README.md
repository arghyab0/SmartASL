# SmartASL
The **aim** of this project is to provide a platform that interprets American Sign Language to text and speech in real-time. We planned to implement it as an Android application considering the huge userbase. Users are expcted to launch the app and point his mobile camera at the person communicating in ASL and the app will display the sentence conveyed in ASL which can be translated to mutiple languages. To make the app as accessible as possible, no additional hardware or a working internet connection is required. We believe that communication should always be two-sided and in that light, we added a module that converts speech to text so that a person can communicate to the person with hearing and speaking impairment. There is also a helper [website](http://ArghyaBiswas0.github.io/SmartASL/helperwebsite/index.html) to learn ASL interactively and provide us feedback.

This prototype was developed as a 12hour hackathon build in Solve4Bharat hackathon organized by the PanIIT association in IISc Bangalore.


#### Notes:

- I directly imported the Kaggle dataset to my Colab environment using the dataset API. You can also download the dataset locally from [here](https://www.kaggle.com/datamunge/sign-language-mnist) and then upload it to the environment you're working on. I personally found the previous method to be much faster.

- I have used Tensorflow version 2.1.0 and that or a later version is what I recommend.

- Training the model for 10 epochs \(approximately 11s per epoch) gave me a training accuracy of 99.5% and validation accuracy of 99.2%

- To integrate the Tensorflow model in the Android app, I am coverting the Tensorflow model to a TFLite model in the notebook and then copying the .tflite file to the assets directory for the Android app.
