{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fashion MNIST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import struct\n",
    "import numpy as np\n",
    "\n",
    "def read_idx(filename):\n",
    "    \"\"\"Credit: https://gist.github.com/tylerneylon\"\"\"\n",
    "    with open(filename, 'rb') as f:\n",
    "        zero, data_type, dims = struct.unpack('>HBB', f.read(4))\n",
    "        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))\n",
    "        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = read_idx(\"./fashion/fashion/train-images-idx3-ubyte\")\n",
    "y_train = read_idx(\"./fashion/fashion/train-labels-idx1-ubyte\")\n",
    "x_test = read_idx(\"./fashion/fashion/t10k-images-idx3-ubyte\")\n",
    "y_test = read_idx(\"./fashion/fashion/t10k-labels-idx1-ubyte\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's inspect our dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Initial shape or dimensions of x_train (60000, 28, 28)\n",
      "Number of samples in our training data: 60000\n",
      "Number of labels in our training data: 60000\n",
      "Number of samples in our test data: 10000\n",
      "Number of labels in our test data: 10000\n",
      "\n",
      "Dimensions of x_train:(28, 28)\n",
      "Labels in x_train:(60000,)\n",
      "\n",
      "Dimensions of x_test:(28, 28)\n",
      "Labels in y_test:(10000,)\n"
     ]
    }
   ],
   "source": [
    "# printing the number of samples in x_train, x_test, y_train, y_test\n",
    "print(\"Initial shape or dimensions of x_train\", str(x_train.shape))\n",
    "\n",
    "print (\"Number of samples in our training data: \" + str(len(x_train)))\n",
    "print (\"Number of labels in our training data: \" + str(len(y_train)))\n",
    "print (\"Number of samples in our test data: \" + str(len(x_test)))\n",
    "print (\"Number of labels in our test data: \" + str(len(y_test)))\n",
    "print()\n",
    "print (\"Dimensions of x_train:\" + str(x_train[0].shape))\n",
    "print (\"Labels in x_train:\" + str(y_train.shape))\n",
    "print()\n",
    "print (\"Dimensions of x_test:\" + str(x_test[0].shape))\n",
    "print (\"Labels in y_test:\" + str(y_test.shape))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's view some sample images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 6 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Let's do the same thing but using matplotlib to plot 6 images \n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Plots 6 images, note subplot's arugments are nrows,ncols,index\n",
    "# we set the color map to grey since our image dataset is grayscale\n",
    "plt.subplot(331)\n",
    "random_num = np.random.randint(0,len(x_train))\n",
    "plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))\n",
    "\n",
    "plt.subplot(332)\n",
    "random_num = np.random.randint(0,len(x_train))\n",
    "plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))\n",
    "\n",
    "plt.subplot(333)\n",
    "random_num = np.random.randint(0,len(x_train))\n",
    "plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))\n",
    "\n",
    "plt.subplot(334)\n",
    "random_num = np.random.randint(0,len(x_train))\n",
    "plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))\n",
    "\n",
    "plt.subplot(335)\n",
    "random_num = np.random.randint(0,len(x_train))\n",
    "plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))\n",
    "\n",
    "plt.subplot(336)\n",
    "random_num = np.random.randint(0,len(x_train))\n",
    "plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))\n",
    "\n",
    "# Display out plots\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's create our model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x_train shape: (60000, 28, 28, 1)\n",
      "60000 train samples\n",
      "10000 test samples\n",
      "Number of Classes: 10\n",
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       \n",
      "_________________________________________________________________\n",
      "batch_normalization_1 (Batch (None, 26, 26, 32)        128       \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     \n",
      "_________________________________________________________________\n",
      "batch_normalization_2 (Batch (None, 24, 24, 64)        256       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 12, 12, 64)        0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 9216)              0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 128)               1179776   \n",
      "_________________________________________________________________\n",
      "batch_normalization_3 (Batch (None, 128)               512       \n",
      "_________________________________________________________________\n",
      "dropout_2 (Dropout)          (None, 128)               0         \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 10)                1290      \n",
      "=================================================================\n",
      "Total params: 1,200,778\n",
      "Trainable params: 1,200,330\n",
      "Non-trainable params: 448\n",
      "_________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "from keras.datasets import mnist\n",
    "from keras.utils import np_utils\n",
    "import keras\n",
    "from keras.datasets import mnist\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D, BatchNormalization\n",
    "from keras import backend as K\n",
    "\n",
    "# Training Parameters\n",
    "batch_size = 128\n",
    "epochs = 3\n",
    "\n",
    "# Lets store the number of rows and columns\n",
    "img_rows = x_train[0].shape[0]\n",
    "img_cols = x_train[1].shape[0]\n",
    "\n",
    "# Getting our date in the right 'shape' needed for Keras\n",
    "# We need to add a 4th dimenion to our date thereby changing our\n",
    "# Our original image shape of (60000,28,28) to (60000,28,28,1)\n",
    "x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n",
    "x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n",
    "\n",
    "# store the shape of a single image \n",
    "input_shape = (img_rows, img_cols, 1)\n",
    "\n",
    "# change our image type to float32 data type\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')\n",
    "\n",
    "# Normalize our data by changing the range from (0 to 255) to (0 to 1)\n",
    "x_train /= 255\n",
    "x_test /= 255\n",
    "\n",
    "print('x_train shape:', x_train.shape)\n",
    "print(x_train.shape[0], 'train samples')\n",
    "print(x_test.shape[0], 'test samples')\n",
    "\n",
    "# Now we one hot encode outputs\n",
    "y_train = np_utils.to_categorical(y_train)\n",
    "y_test = np_utils.to_categorical(y_test)\n",
    "\n",
    "# Let's count the number columns in our hot encoded matrix \n",
    "print (\"Number of Classes: \" + str(y_test.shape[1]))\n",
    "\n",
    "num_classes = y_test.shape[1]\n",
    "num_pixels = x_train.shape[1] * x_train.shape[2]\n",
    "\n",
    "# create model\n",
    "model = Sequential()\n",
    "\n",
    "model.add(Conv2D(32, kernel_size=(3, 3),\n",
    "                 activation='relu',\n",
    "                 input_shape=input_shape))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Conv2D(64, (3, 3), activation='relu'))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "model.add(Dropout(0.25))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(128, activation='relu'))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(num_classes, activation='softmax'))\n",
    "\n",
    "model.compile(loss = 'categorical_crossentropy',\n",
    "              optimizer = keras.optimizers.Adadelta(),\n",
    "              metrics = ['accuracy'])\n",
    "\n",
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's train our model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/3\n",
      "60000/60000 [==============================] - 220s 4ms/step - loss: 0.4570 - accuracy: 0.8414 - val_loss: 1.0266 - val_accuracy: 0.6811\n",
      "Epoch 2/3\n",
      "60000/60000 [==============================] - 213s 4ms/step - loss: 0.2869 - accuracy: 0.8982 - val_loss: 0.2543 - val_accuracy: 0.9078\n",
      "Epoch 3/3\n",
      "60000/60000 [==============================] - 217s 4ms/step - loss: 0.2341 - accuracy: 0.9160 - val_loss: 0.2517 - val_accuracy: 0.9076\n",
      "Test loss: 0.2516735388636589\n",
      "Test accuracy: 0.9075999855995178\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(x_train, y_train,\n",
    "          batch_size=batch_size,\n",
    "          epochs=epochs,\n",
    "          verbose=1,\n",
    "          validation_data=(x_test, y_test))\n",
    "\n",
    "score = model.evaluate(x_test, y_test, verbose=0)\n",
    "print('Test loss:', score[0])\n",
    "print('Test accuracy:', score[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's test out our model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'cv2'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-7-7f3be4dd479e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mcv2\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mgetLabel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minput_class\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[0mnumber\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minput_class\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'cv2'"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "def getLabel(input_class):\n",
    "    number = int(input_class)\n",
    "    if number == 0:\n",
    "        return \"T-shirt/top \"\n",
    "    if number == 1:\n",
    "        return \"Trouser\"\n",
    "    if number == 2:\n",
    "        return \"Pullover\"\n",
    "    if number == 3:\n",
    "        return \"Dress\"\n",
    "    if number == 4:\n",
    "        return \"Coat\"\n",
    "    if number == 5:\n",
    "        return \"Sandal\"\n",
    "    if number == 6:\n",
    "        return \"Shirt\"\n",
    "    if number == 7:\n",
    "        return \"Sneaker\"\n",
    "    if number == 8:\n",
    "        return \"Bag\"\n",
    "    if number == 9:\n",
    "        return \"Ankle boot\"\n",
    "\n",
    "def draw_test(name, pred, actual, input_im):\n",
    "    BLACK = [0,0,0]\n",
    "\n",
    "    res = getLabel(pred)\n",
    "    actual = getLabel(actual)   \n",
    "    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, 4*imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)\n",
    "    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)\n",
    "    cv2.putText(expanded_image, \"Predicted - \" + str(res), (152, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0), 1)\n",
    "    cv2.putText(expanded_image, \"   Actual - \" + str(actual), (152, 90) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 1)\n",
    "    cv2.imshow(name, expanded_image)\n",
    "\n",
    "\n",
    "for i in range(0,10):\n",
    "    rand = np.random.randint(0,len(x_test))\n",
    "    input_im = x_test[rand]\n",
    "    actual = y_test[rand].argmax(axis=0)\n",
    "    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)\n",
    "    input_im = input_im.reshape(1,28,28,1) \n",
    "    \n",
    "    ## Get Prediction\n",
    "    res = str(model.predict_classes(input_im, 1, verbose = 0)[0])\n",
    "\n",
    "    draw_test(\"Prediction\", res, actual, imageL) \n",
    "    cv2.waitKey(0)\n",
    "\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
