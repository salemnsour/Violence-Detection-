#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from keras.applications import VGG16
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
import sys
import h5py


# # Downloading the Dataset

# In[4]:


def print_progress(count, max_count):
    # Percentage completion.
    pct_complete = count / max_count

    # Status-message. Note the \r which means the line should
    # overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


# In[5]:


def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """

    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


# In[6]:


def maybe_download_and_extract(url, download_dir):
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a tar-ball file.
    :param url:
        Internet URL for the tar-file to download.
        Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/CIFAR-10/"
    :return:
        Nothing.
    """

    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")




# In[7]:


def download_data(in_dir, url):
    
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    maybe_download_and_extract(url,in_dir)


# In[141]:


#directory that where the data will be downloaded
in_dir = "data4"


# In[9]:


#the URL for the Dataset
url_hockey = "http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip"


# In[10]:


#To download the data and decompress it we will use this function
download_data(in_dir,url_hockey)


# In[13]:


import os

# Define the path to the problematic file
file_to_remove = "data4/HockeyFights.zip"

# Check if the file exists and remove it
if os.path.exists(file_to_remove):
    os.remove(file_to_remove)
    print(f"Removed file: {file_to_remove}")
else:
    print(f"File not found: {file_to_remove}")

# List the files in the directory to verify removal


# Some Data Dimensions

# In[140]:


# Frame size  
img_size = 224

img_size_touple = (img_size, img_size)

# Number of channels (RGB)
num_channels = 3

# Flat frame size
img_size_flat = img_size * img_size * num_channels

# Number of classes for classification (Violence-No Violence)
num_classes = 2

# Number of files to train
_num_files_train = 1

# Number of frames per video
_images_per_file = 20

# Number of frames per training set
_num_images_train = _num_files_train * _images_per_file

# Video extension
video_exts = ".avi"


# Function used to get 20 frames from a video file

# In[16]:


def get_frames(current_dir, file_name):
    
    in_file = os.path.join(current_dir, file_name)
    
    images = []
    
    vidcap = cv2.VideoCapture(in_file)
    
    success,image = vidcap.read()
        
    count = 0

    while count<_images_per_file:
                
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                                 interpolation=cv2.INTER_CUBIC)
    
        images.append(res)
    
        success,image = vidcap.read()
    
        count += 1
        
    resul = np.array(images)
    
    resul = (resul / 255.).astype(np.float16)
        
    return resul
    
    


# Function that labels the video names 

# In[17]:


def label_video_names(in_dir):
    
    # list containing video names
    names = []
    # list containin video labels [1, 0] if it has violence and [0, 1] if not
    labels = []
    
    
    for current_dir, dir_names,file_names in os.walk(in_dir):
        
        for file_name in file_names:
            
            if file_name[0:2] == 'fi':
                labels.append([1,0])
                names.append(file_name)
            elif file_name[0:2] == 'no':
                labels.append([0,1])
                names.append(file_name)
                     
            
    c = list(zip(names,labels))
    # Suffle the data (names and labels)
    shuffle(c)
    
    names, labels = zip(*c)
            
    return names, labels


# In[18]:


#First get the names and labels of the whole videos
names, labels = label_video_names(in_dir)
     


# In[19]:


#The names of the videos
names


# # Plotting a video frames

# In[20]:


frames = get_frames(in_dir, names[999])


# In[21]:


visible_frame = (frames*255).astype('uint8')


# In[22]:


plt.imshow(visible_frame[3])


# In[23]:


plt.imshow(visible_frame[15])


# # Pre-Trained Model: VGG16

# The VGG16 model contains a convolutional part and a fully-connected (or dense) part which is used for classification. If include_top=True then the whole VGG16 model is downloaded which is about 528 MB

# weights=‘imagenet’: This indicates that the model should use the weights that were pre-trained on the ImageNet dataset. The ImageNet dataset contains millions of labeled images across thousands of categories, making it a comprehensive resource for training robust image classification models.

# In[144]:


image_model = VGG16(include_top=True, weights='imagenet')


# Here is the model summary

# In[145]:


image_model.summary()


# As we can see that the input size must be (224,224,3) that as we specified at the Data dimension section

# In[26]:


input_shape = image_model.input_shape[1:3]
input_shape


# # The VGG16 model flowchart

# The following chart shows how the data flows when using the VGG16 model for Transfer Learning. First we input and process 20 video frames in batch with the VGG16 model. Just prior to the final classification layer of the VGG16 model, we save the so-called Transfer Values to a cache-file.
# 
# The reason for using a cache-file is that it takes a long time to process an image with the VGG16 model. If each image is processed more than once then we can save a lot of time by caching the transfer-values.
# 
# When all the videos have been processed through the VGG16 model and the resulting transfer-values saved to a cache file, then we can use those transfer-values as the input to LSTM neural network. We will then train the second neural network using the classes from the violence dataset (Violence, No-Violence), so the network learns how to classify images based on the transfer-values from the VGG16 model.

# In[27]:


from tensorflow.keras import backend as K
# We will use the output of the layer prior to the final
# classification-layer which is named fc2. This is a fully-connected (or dense) layer.
transfer_layer = image_model.get_layer('fc2')

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

transfer_values_size = K.int_shape(transfer_layer.output)[1]


print("The input of the VGG16 net have dimensions:",K.int_shape(image_model.input)[1:3])

print("The output of the selecter layer of VGG16 net have dimensions: ", transfer_values_size)


# A function to proccess a 20 frames from a video through the pretrained VGG16 model and return the transfered values

# In[28]:


def get_transfer_values(current_dir, file_name):
    
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    
    image_batch = get_frames(current_dir, file_name)
      
    # Pre-allocate output-array for transfer-values.
    # Note that we use 16-bit floating-points to save memory.
    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = \
            image_model_transfer.predict(image_batch)
            
    return transfer_values


# A function that generates each video through VGG16 model

# In[29]:


def proces_transfer(vid_names, in_dir, labels):
    
    
    count = 0
    
    tam = len(vid_names)
    
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)
    while count<tam:
        
        video_name = vid_names[count]
        
        image_batch = np.zeros(shape=shape, dtype=np.float16)
    
        image_batch = get_frames(in_dir, video_name)
        
         # Note that we use 16-bit floating-points to save memory.
        shape = (_images_per_file, transfer_values_size)
        transfer_values = np.zeros(shape=shape, dtype=np.float16)
        
        transfer_values = \
            image_model_transfer.predict(image_batch)
         
        labels1 = labels[count]
        
        aux = np.ones([20,2])
        
        labelss = labels1*aux
        
        yield transfer_values, labelss
        
        count+=1
    


# A function to save the transfered Values for later use 

# In[30]:


def make_files(n_files):
    
    gen = proces_transfer(names_training, in_dir, labels_training)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File('prueba1.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            print_progress(numer, n_files)
        
            numer += 1


# In[31]:


def make_files_test(n_files):
    
    gen = proces_transfer(names_test, in_dir, labels_test)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File('pruebavalidation1.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            print_progress(numer, n_files)
        
            numer += 1


# # Splitting the dataset into training and testing

# In[32]:


training_set = int(len(names)*0.8)
test_set = int(len(names)*0.2)

names_training = names[0:training_set]
names_test = names[training_set:]

labels_training = labels[0:training_set]
labels_test = labels[training_set:]


# Here we are going to proccess all of the frames of the videos through the VGG16 model and then saving the transfered values into the disk

# In[33]:


make_files(training_set)


# In[34]:


make_files_test(test_set)


# In[35]:


test_set


# In[36]:


training_set


# In[1]:


'''
Reimporting the libraries for easier Access

%matplotlib inline
import cv2
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from keras.applications import VGG16
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
import sys
import h5py
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization
from keras.optimizers import Adam
'''


# # Loading the cached transfer values into the memory

# We have already saved all the videos transfer values into disk. But we have to load those transfer values into memory in order to train the LSTM net.

# In[2]:


def process_alldata_training():
    
    joint_transfer=[]
    frames_num=20
    count = 0
    
    with h5py.File('prueba1.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count+frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target
     


# In[3]:


def process_alldata_test():
    
    joint_transfer=[]
    frames_num=20
    count = 0
    
    with h5py.File('pruebavalidation1.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count+frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target


# In[4]:


data, target = process_alldata_training()



# In[5]:


data_test, target_test = process_alldata_test()


# # Defining the LSTM model

# LSTM is an advanced Type of RNN designed to handle long-term dependencies more effectively.
# 
# Define LSTM architecture:
# 
# When defining the LSTM architecture we have to take into account the dimensions of the transfer values. From each frame the VGG16 network obtains as output a vector of 4096 transfer values. From each video we are processing 20 frames so we will have 20 x 4096 values per video. The classification must be done taking into account the 20 frames of the video. If any of them detects violence, the video will be classified as violent.

# In[114]:


chunk_size = 4096 # The size of each chunk of data (the size of each frame)

n_chunks = 20 # This defines the number of chunks in the sequence (20 frames)

rnn_size = 265 # This is the number of units (neurons) in the LSTM layer.

model5 = Sequential() # Initialize the sequential model

# LSTM layer with 'rnn_size' units. The input shape is (n_chunks, chunk_size)
model5.add(LSTM(rnn_size, input_shape=(n_chunks, chunk_size)))

# This helps to stabilize and accelerate the training process and Provides a regularizing effect
model5.add(BatchNormalization())

# This helps to prevent overfitting by randomly shutting down some neurons by 50%
model5.add(Dropout(0.5))

model5.add(Dense(2048))# Fully connected layer with 2048 units(neuron)

model5.add(BatchNormalization())# Add batch normalization layer

model5.add(Activation('sigmoid'))# Sigmoid activation function for non-linearity

model5.add(Dropout(0.3))# Dropping neurons by 30%

model5.add(Dense(64))# Fully connected layer with 64 units(neuron)

model5.add(Activation('sigmoid'))# Sigmoid activation function for non-linearity

model5.add(Dropout(0.2))# Dropping neurons by 20%

model5.add(Dense(2))# Fully connected layer with 2 units(neuron) for binary classification

model5.add(Activation('sigmoid'))

learning_rate = 0.000005  # learning rate for the optimizer

optimizer = Adam(learning_rate=learning_rate)# Adam optimizer with the specified learning rate

# Compile the model with binary cross-entropy loss and Adam optimizer
model5.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Stop training when the loss has stopped improving for 7 epochs
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor = "loss", patience = 7, restore_best_weights= True)

# Reduce learning rate when the validation loss has stopped improving for 2 epochs
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss", factor = 0.6, patience = 2,
    min_lr = 0.0000005, verbose = 1)


# # Model training

# In[115]:


epoch = 100
batchS = 64

history5 = model5.fit(np.array(data[0:750]), np.array(target[0:750]), epochs=epoch, 
                    batch_size=batchS,validation_data=(np.array(data[750:]),np.array(target[750:])),
                      callbacks = [early_stopping_callback,reduce_lr_callback])


# # Model testing an evaluating

# We are here testing the performance of the model by sevral metrics , and we have assured the test set have not been seen by the model

# In[122]:


result = model5.evaluate(np.array(data_test), np.array(target_test))


# In[147]:


from sklearn.metrics import classification_report 
print("Classification Report:")
print(classification_report(target_test, y_pred, target_names=['NonViolence', 'Violence']))


# In[148]:


from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix

# Get the predictions for your test data
y_pred = model5.predict(np.array(data_test))

y_pred = (y_pred > 0.5).astype("int32")
# Calculate the confusion matrix
mcm = multilabel_confusion_matrix(target_test, y_pred)

print("Confusion Matrix:")
print(mcm)


# In[149]:


import numpy as np
import matplotlib.pyplot as plt

# Plot the confusion matrix for each label
for i, cm in enumerate(mcm):
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted Negative', 'Predicted Positive'])
    plt.yticks([0, 1], ['True Negative', 'True Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for Label {i}')
    
    # Add text annotations
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            plt.text(x, y, f'{cm[y, x]}', ha='center', va='center', color='black')
    
    plt.show()


# # Showing the accuracy and loss over epochs  

# In[117]:


plt.plot(history5.history['accuracy'])
plt.plot(history5.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('destination_path.eps', format='eps', dpi=1000)
plt.show()

# summarize history for loss
plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('destination_path1.eps', format='eps', dpi=1000)
plt.show()


# # Saving the best weights for the model

# We are saving the Weights so w dont have to Train the model further

# In[139]:


weights_path = 'best_weights1.h5'
with h5py.File(weights_path, 'w') as f:
    for layer in model5.layers:
        g = f.create_group(layer.name)
        weights = layer.get_weights()
        for i, weight in enumerate(weights):
            g.create_dataset(f"weight_{i}", data=weight)

print(f"Model weights saved to: {weights_path}")




# In[137]:


with h5py.File(weights_path, 'r') as f:
    for layer in model5.layers:
        weights = [f[layer.name][f"weight_{i}"][:] for i in range(len(f[layer.name]))]
        layer.set_weights(weights)

