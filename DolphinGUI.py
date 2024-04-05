# -*- coding: utf-8 -*-

from tkinter import Frame, Tk, Button, messagebox, Entry, Label
from tkinter.filedialog import askopenfilename, askdirectory

import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg

#import matplotlib_inline

import numpy as np
from numpy import asarray

import tensorflow as tf
from tensorflow import keras

#tf.get_logger().setLevel('ERROR')
#import tensorflow.python.keras.backend
#import tensorflow.compat.v1
import librosa.display

import os
import codecs
import inspect
from PIL import Image, ImageTk

import csv

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import shap

# disabling certain newer tf features for compatibility with shap
#tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution() 
 
# TODO: Find a way to use shap values to recreate audible (wavfile) version of what highlighted spectrograms indicate
# TODO: Figure out why explain function is so memory intensive
# NOTE: Loading transfer learning models with the new tensorflow folder saving format seems to give an error for model loading,
#       so it is safer to save them in an h5 format for now. Custom-made models seem to be fine with being loaded in 
#       the newer format though. 

# create super class for more general model to which other SHAP methods may be applied
class ClassifierModel:
    """ This class is responsible for creating a dolphin acoustics classifier model.
    """

    def __init__(self, model):
        self.model = model
        self.class_names = None

    def get_label_indices(self, test_data, multiclass_test_labels):

        """This function returns the indices of labels for predicted labels and actual labels. 
        The label indices should correspond to self.class_names.
        

        Parameters
        ----------
        test_data : numpy array
            Array of data for testing
        multiclass_test_labels : numpy array
            Correct labels corresponding to test_data arrays

        Returns
        -------
        tuple
            Tuple of arrays with test label indices and predicted label indices (in that order)

        """

        probability_scores = self.model.predict(test_data)  # list of predicted class probabilities for each image
                                                            # model.predict better for batches but equivalent to model()
        predicted_label_indices = probability_scores.argmax(axis = 1)
        test_label_indices = multiclass_test_labels.argmax(axis = 1)
        return test_label_indices, predicted_label_indices 
    
    def report_stats(self, test_data, multiclass_test_labels):
        """This function returns statistical metrics about the model's performance, 
        including precisiion, recall and F1 score.
        

        Parameters
        ----------
        test_data : numpy array
            Array of data for testing
        multiclass_test_labels : numpy array
            Correct labels corresponding to test_data arrays

        Returns
        -------
        string
            String giving statistical metrics on the model's performance

        """
        # get indices of labels for predicted labels and actual labels 
        test_label_indices, predicted_label_indices = self.get_label_indices(test_data, multiclass_test_labels)

        return classification_report(test_label_indices, predicted_label_indices, target_names = self.class_names, 
                                    labels = np.unique(np.concatenate([predicted_label_indices, test_label_indices])))

    def show_confusion_matrix(self, test_data, multiclass_test_labels):

        """
        This function plots a confusion matrix for some given data by comparing the 
        model's predictions with provided test labels.

        Parameters
        ----------
        test_data : numpy array
            Array of data of for testing
        multiclass_test_labels : numpy array
            Correct labels corresponding to test_data array

        Returns
        -------
        None.

        """

        # get indices of labels for predicted labels and actual labels 
        test_label_indices, predicted_label_indices = self.get_label_indices(test_data, multiclass_test_labels)
        
        # adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions
        # create confusion matrix object and plot its results
        cm = confusion_matrix(test_label_indices, predicted_label_indices, 
                              labels = np.unique(np.concatenate([predicted_label_indices, test_label_indices])))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                         display_labels=self.class_names)
        display.plot()

# create image-related subclass extending ClassifierModel class
class ImageClassifierModel(ClassifierModel):

    """This class is responsible for creating an image classifier model.
    """

    def __init__(self, model, image_data, data_labels, **kwargs):
        """
        Parameters
        ----------
        model : tensorflow model
            Machine learning model for classification
        image_data : numpy array
            Array of image arrays of for testing
        data_labels : list
            Labels corresponding to the class of each image
        data_labels : numpy array
            Correct labels corresponding to test_data image arrays
        class_names : list, optional
            Set of (unique) image labels, deduced 
            from unique values of data_labels if not specified.
        
        Returns
        -------
        None.

        """
        
        super().__init__(model)
        self.image_data = image_data
        self.data_labels = data_labels
        
        # get unique label names if not directly specified
        self.class_names = kwargs.get("class_names", list(set(data_labels))) 

    def explain(self, test_images , multiclass_test_labels = None, background_images = None, 
        background_image_limit: int = 20, test_image_limit: int = 4):

        """
        This function displays an image highlighting relevant pixels to the model'shap 
        compatibility  classification decision.
        
        Note: This function can be quite computationally expensive 
        depending on the choice of machine learning model. 
        It is advised that the number of background images be chosen carefully, 
        with a low number (perhaps 20 or less) being safer to start with if unsure. 

        Parameters
        ----------
        test_images : numpy array
            Array of image arrays of for testing
        background_images : numpy array, optional
            Images used to get background understanding of the model's expected predictions.
            The default is self.image_data.
        multiclass_test_labels : numpy array
            Correct image labels corresponding to test_images arrays
        background_image_limit : int, optional
            Maximum number of background images used in calculation.
            The default is 20.
        test_image_limit : int, optional
            Maximum number of test images to be explained, starting at index 0, 
            and ending at index before test_image_limit. The default is 4.

        Returns
        -------
        None.

        """
        
        if background_images is None:
            background_images = self.image_data

        # adapted from https://github.com/slundberg/shap#deep-learning-example-with-deepexplainer-tensorflowkeras-models
        
        # select a set of background examples to take an expectation over
        if len(background_images) > background_image_limit:
            background_images = background_images[0 : background_image_limit] #[np.random.choice(train_images.shape[0], background_image_limit, replace=False)]
        
        # select test images to be explained
        if len(test_images) > test_image_limit:
            test_images = test_images[0 : test_image_limit]
    
        # explain predictions of the model
        explainer = shap.DeepExplainer(self.model, background_images)
    
        # get shap values from explainer
        # TODO: investigate behaviour of check_additivity
        # (check_additivity=True seems to lead to failure for single test image)
        shap_values = explainer.shap_values(test_images, check_additivity=False) 
    
        # create labels for image plots
        labels = np.array(self.class_names)
        labels = np.tile(labels, shap_values[0].shape[0]) # duplicate labels for every row of images
        labels = np.reshape(labels, (shap_values[0].shape[0], len(shap_values))) # reshape array appropriately 
                                                                                 # for shap compatibility 

        # print predicted labels and actual labels for each test image
        if multiclass_test_labels is not None:
            test_label_indices, predicted_label_indices = self.get_label_indices(test_images, multiclass_test_labels)
            print("Predicted Labels\t\tActual Labels\t\t(in order of appearance of images below)")
            for i in range(len(predicted_label_indices)):
                print(self.class_names[predicted_label_indices[i]] + "\t\t" + self.class_names[test_label_indices[i]])

        # plot the image explanations
        shap.image_plot(shap_values, test_images[0:1]/255, labels = labels, show=False)
        plt.savefig(os.path.join(os.getcwd(), "cetacexplain.jpg"))

def show_image(image_array_data):
      """
      Display image from numpy array image representation

      Parameters
      ----------
      image_array_data : numpy array
          array image representation

      Returns
      -------
      None.

      """
      plt.imshow(image_array_data)

def convert_time_series_to_wavfile():
      """This funcion converts a time series of frequency values to a wave file
      """
      pass

def save_spectrogram_image(
        input_path,
        output_path,
        image_name,
        sampling_rate=48000,
        n_fft=512,
        dpi=96,  # this should be dpi of your own screen
        max_freq=22000,  # for cropping
        min_freq=3000,  # for cropping
        img_size=(413, 202)):
        """
        This function takes in the above parameters and
        generates a spectrogram from a given sample recording and
        saves the spectrogram image
        """

        f_step = sampling_rate / n_fft
        min_bin = int(min_freq / f_step)
        max_bin = int(max_freq / f_step)

        # Generate image
        x, sr = librosa.load(input_path, sr=sampling_rate)

        X = librosa.stft(x, n_fft=n_fft)  # Apply fourier transform
        # Crop image vertically (frequency axis) from min_bin to max_bin
        X = X[min_bin:max_bin, :]

        # TODO change refs
        Xdb = librosa.amplitude_to_db(
            abs(X), ref=np.median
        )  # Convert amplitude spectrogram to dB-scaled spec
        fig = plt.figure(
            frameon=False, figsize=(img_size[0] / dpi, img_size[1] / dpi), dpi=dpi
        )  # Reduce image

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(Xdb, cmap="gray_r", sr=sr,
                                x_axis="time", y_axis="hz")
        #plt.show()

        # Save image
        fig.savefig(os.path.join(output_path, str(image_name) + ".png"))
        plt.close(fig)
    
def image_to_array(image_file):
    """This function coverts a PIL image to a numpy array.
    """
    # load image
    image = Image.open(image_file).convert("RGB")

    # convert image to numpy array
    return asarray(image)

def get_prediction(image_file, model):
    """ This function returns a string of the predicted dolphin species
    """

    classes = ["No Whistle", "Whistle"]
    image_array = image_to_array(image_file)
    image_array = np.array([image_array.tolist()])
    predicted_class = np.argmax(model.predict(image_array))

    return classes[predicted_class]

class Whistle_GUI():
    
    """ This class is responsible for running the graphical user interface
        allowing users to interact with AI model that classifies spectrograms
        based on the perceived presence or absence of whistles in audio files.
    """
    

    def __init__(self):
        
        # initialize tkinter window
        self.root = Tk()
        self.frame = Frame(self.root, height=750, width=750)
        self.root.title("Whistle-Noise Classifer")

        # initialize label for filepath to audio wav file
        self.whistle_wav_path = None

        # initialize label for filepath to folder in which selection tables are kept
        self.selection_table_folder = None
        
        # initalize spectrogram image
        self.spectrogram_image = None
        
        # initialize predicted image class
        self.prediction = None
        
        # intialize machine learning model
        self.model = tf.keras.models.load_model(
            os.path.join(os.getcwd(),"2024_10_02_xception_rosie_fine_tuned.h5"), compile=False)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy()]
            )
        
        # initialize background images and corresponding labels
        self.background_images = []
        self.background_image_labels = np.array([])
        image_dir = os.path.join(os.getcwd(), "Background_Images")
        for file in os.listdir(image_dir):
            self.background_images.append(image_to_array(os.path.join(image_dir, file)))
            if file.endswith("n.png"):
                self.background_image_labels = np.append(self.background_image_labels, "No Whistle")
                
            elif file.endswith("y.png"):
                self.background_image_labels = np.append(self.background_image_labels, "Whistle")
       
        self.background_images = np.array(self.background_images, dtype=float) # convert to numpy array
    
        print(len(self.background_image_labels), "background images found")
        
        print(self.background_images.shape)
        
        # initialize explainer
        self.explainer = ImageClassifierModel(self.model, self.background_images,
                                                      self.background_image_labels)

        
        
        
        # create textbox for entry of saved file name
        #self.textbox = Entry(width=50)
        #self.textbox.pack()
        #self.textbox.place(relx=0.5, rely=0.7, anchor="center")

        
        

        # add relevant items to frame
        self.create_frame_objects()
        
        
    
    def start(self):
        
        try:
            self.frame.pack()
            self.frame.update() # TODO
            
            # centre frame on screen
            screen_height = self.root.winfo_screenheight()
            screen_width = self.root.winfo_screenwidth()
            
            screen_centre = ((screen_width/2) - (self.frame.winfo_width()/2),
                             (screen_height/2) - (self.frame.winfo_height()/2))
            self.root.geometry(f"{self.frame.winfo_width()}x{self.frame.winfo_height()}"+
                               f"+{int(screen_centre[0])}+{int(screen_centre[1])}")
            
            
            # start main loop
            self.root.mainloop()
        except:
            messagebox.showerror(message="An error has occurred. Please close this window and try again.")
            if self.root.winfo_exists():
                self.root.destroy()
            
    def select_wav_file(self):
        """
        Requests path of a wav file from a user and generates AI-based 
        explanation of spectrogram to show user insight behind the machine 
        learning model's classification.

        Returns
        -------
        None.

        """
        
        filepath = askopenfilename(filetypes=[("Wav file", "*.wav")])

        
        if (self.whistle_wav_path != None):
            self.whistle_wav_path["text"] = filepath
        else:
            self.whistle_wav_path = Label(self.root, text=filepath) 
            self.whistle_wav_path.pack()
            self.whistle_wav_path.place(relx=0.5, rely=0.3, anchor="center")
            
            
            print("starting explanation")
            self.explain_audio(filepath)
            print("audio explained")
            self.image = Image.open(os.path.join(os.getcwd(),'cetacexplain.jpg'))
            self.photo = ImageTk.PhotoImage(self.image)
            self.spectrogram_image = Label(self.root, image=self.photo, borderwidth=0)
            self.spectrogram_image.pack()
            self.spectrogram_image.image = self.image
            self.spectrogram_image.place(relx=0.5, rely=0.5, anchor="center", 
                                         relwidth=0.5, relheight=0.39)

            self.prediction = Label(self.root, 
                                    text=get_prediction(os.path.join(os.getcwd(),"explained_image.png"), 
                                                        self.model))
            self.prediction.pack()
            self.prediction.place(relx=0.5, rely=0.7)
            
    def select_folder(self):
        folder_name = askdirectory()
        
        if (self.selection_table_folder != None):
            self.selection_table_folder["text"] = "Analysed selections in "+folder_name
        else:
            self.selection_table_folder = Label(self.root, text=folder_name)
            self.selection_table_folder.pack()
            self.selection_table_folder.place(relx=0.5, rely=0.8, anchor="center")
            self.read_seletion_table(folder_name)
            
    def explain_audio(self, audio_path):
        
        # create random sample of background images for explainer to gauge expected model output
        # and create random test images for explanation
        save_spectrogram_image(audio_path, os.getcwd(), "explained_image")
        
        # convert spectrogram to numpy array
        spectrogram_arr = np.array([image_to_array(os.path.join(os.getcwd(), 
                                        "explained_image.png"))], dtype=float)
        self.explainer.explain(spectrogram_arr)
    
    def read_seletion_table(self, folder_name):
        print("started")
        main_dir = folder_name
        dir_list = os.listdir(main_dir)
        selection_tables = [] # initialise list of selection tables

        
        
        for dir in dir_list:
            print(dir)
        
            # ignore any Python notebooks in directory
            if dir.endswith(".ipynb"):
                continue
            
            # find all selection table file paths
            elif os.path.isdir(os.path.join(main_dir, dir)):
                
                for file in os.listdir(os.path.join(main_dir, dir)):
        
                    if file.endswith(".selections.txt"):
                        selection_tables.append(os.path.join(main_dir, dir, file))
                        print(file)
        
        
        removed_end_str_length = 26 # length to be removed from end of string
        

        
        for selection_table in selection_tables:
        
            # get column data from selection tables
            columns = np.genfromtxt(selection_table, dtype=str, delimiter="\t", skip_header=True).T
            selection_nums, annotations = columns[0], columns[-1]
        
            # standardise number of zeroes in whistle selection numbers
            selection_nums = np.array([num_string.zfill(2) if len(num_string) else num_string for num_string in selection_nums])
        
            # use annotations column in selection table to group whistles
            with open(os.path.join(os.getcwd(), selection_table[:-removed_end_str_length]+"_review.csv"), "wt") as file:
                
                # write header
                writer = csv.writer(file, delimiter=",")
                writer.writerow(["wav file", "annotation", "AI Prediction"])
                
                for i, annotation in enumerate(annotations):
                    wav_filename = selection_table[:-removed_end_str_length] + "sel_" + selection_nums[i] + ".wav"
                    save_spectrogram_image(wav_filename, os.getcwd(), os.path.join(os.getcwd(), "spectrogram"))
                    predicition = get_prediction(os.path.join(os.getcwd(), "spectrogram.png"), self.model)
                    writer.writerow([wav_filename, annotation, predicition])
                       
        print("finished")
    
    def update_label(self, label, image_file, model):
        """ This function updates the species label being displayed in the tkinter gui by 
        obtaining the machine learning model's most recent precition on the audio input
        """

        new_label = get_prediction(image_file, model)
        label.config(text=new_label)

        
        # close application
        self.root.destroy()

    def create_frame_objects(self):
        
        # create button to specify selection table
        file_select_button = Button(self.root, text="Specify Wav File Path for AI explanaion", command=self.select_wav_file)
        file_select_button.place(relx=0.3, rely=0.2, anchor="center")

        # create button to select folder with relevant selection tables
        folder_select_button = Button(self.root, text="Specify Folder containing selection tables", command=self.select_folder)
        folder_select_button.place(relx=0.7, rely=0.2, anchor="center")
        

        # create label final instructions
        instructions = Label(self.root, text="To analyse selection tables,"
                             +" specify a folder with subfolders for different encounters.\n"
                             + "Each encounter folder should contain associated wav files and selection tables." 
                             + "CSV files ending in '_review.txt' will be written\n in the enccounter folders "
                             + "containing comparisons between AI and your assigned labellings.\nTo analyse an "
                             + "audio spectrogram in detail, an highlighted image can be generated when a "
                             + "wav file path is specified."
                             )
        
        instructions.pack()
        instructions.place(relx=0.5, rely=0.1, anchor="center")
         

def main():

    # create and start GUI
    gui = Whistle_GUI()
    gui.start()
    

if __name__ == "__main__":
    main()
