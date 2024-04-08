# Whistle-Classification-Quality-Assurance-GUI
This repository contains code implementing a simple graphical user interface (GUI) which biologists can use as an aid for quality assurance when selecting whistles. A user can have a machine learning model analyse wav files (specified by selection tables) to provide the user with a second opinion about their whistle selections.

# Setting up Virtual Environment/Conda Environment

It is very important that the Python library versions specified in the `requirements.txt` file be used as the code in this repository is not guaranteed to successfully generate a working Python executable. For the specific `requirements.txt`, the code has been tested to run with both conda and virtual environments. It is also very impotrant that the correct version of Python be used. Python 3.8 was used for development. TensorFlow no longer supports this version of Python (see https://www.tensorflow.org/install/pip) 
