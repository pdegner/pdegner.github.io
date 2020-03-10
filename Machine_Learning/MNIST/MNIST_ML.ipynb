{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "z6UHmLYVhWAN"
   },
   "source": [
    "# Digit Classification with KNN and Naive Bayes\n",
    "\n",
    "This was a project for my machine learning class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "iJ9ayCvyhWAP"
   },
   "outputs": [],
   "source": [
    "# This tells matplotlib not to try opening a new window for each plot.\n",
    "%matplotlib inline\n",
    "\n",
    "# Import a bunch of libraries.\n",
    "import time\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.ticker import MultipleLocator\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.datasets import fetch_openml\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.naive_bayes import BernoulliNB\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "# Set the randomizer seed so results are the same each time.\n",
    "np.random.seed(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "sO1t0ypThWAR"
   },
   "source": [
    "Load the data. Notice that the data gets partitioned into training, development, and test sets. Also, a small subset of the training data called mini_train_data and mini_train_labels gets defined, which you should use in all the experiments below, unless otherwise noted."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "3yK9DacchWAS"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "data shape:  (70000, 784)\n",
      "label shape: (70000,)\n"
     ]
    }
   ],
   "source": [
    "# Load the digit data from https://www.openml.org/d/554 or from default local location '~/scikit_learn_data/...'\n",
    "X, Y = fetch_openml(name='mnist_784', return_X_y=True, cache=False)\n",
    "\n",
    "\n",
    "# Rescale grayscale values to [0,1].\n",
    "X = X / 255.0\n",
    "\n",
    "# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this\n",
    "# permutation to X and Y.\n",
    "# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering.\n",
    "shuffle = np.random.permutation(np.arange(X.shape[0]))\n",
    "X, Y = X[shuffle], Y[shuffle]\n",
    "\n",
    "print('data shape: ', X.shape)\n",
    "print('label shape:', Y.shape)\n",
    "\n",
    "# Set some variables to hold test, dev, and training data.\n",
    "test_data, test_labels = X[61000:], Y[61000:]\n",
    "dev_data, dev_labels = X[60000:61000], Y[60000:61000]\n",
    "train_data, train_labels = X[:60000], Y[:60000]\n",
    "mini_train_data, mini_train_labels = X[:1000], Y[:1000]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "atc2JpWKhWAV"
   },
   "source": [
    "### Show a 10x10 grid that visualizes 10 examples of each digit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "436UeH7JhWAW"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 100 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axs = plt.subplots(10, 10) #sets up Nx10 grid\n",
    "for i in range(10):\n",
    "    yy = np.where(Y == str(i))[0][:10] #Get N indicies of the number of interest\n",
    "\n",
    "    #add image to its place\n",
    "    j = 0\n",
    "    for index in yy: \n",
    "        axs[i, j].imshow(np.reshape(X[index], (28,28)), interpolation='nearest', cmap = \"gray_r\") \n",
    "        axs[i, j].axis(\"off\")\n",
    "        j += 1\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "EMQAHr7QhWAX"
   },
   "source": [
    "### Produce k-Nearest-Neighbors model with k = [1,3,5,7,9].  \n",
    "\n",
    "Evaluate and show the performance of each model. For the 1-Nearest Neighbor model, show precision, recall, and F1 for each label. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "-it5pn8-hWAY"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Error rate when k = 1 is 0.116, and 2 is the most mislabeled.\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.95      0.95      0.95       106\n",
      "           1       0.89      0.98      0.93       118\n",
      "           2       0.90      0.79      0.84       106\n",
      "           3       0.93      0.87      0.90        97\n",
      "           4       0.91      0.85      0.88        92\n",
      "           5       0.86      0.88      0.87        88\n",
      "           6       0.92      0.92      0.92       102\n",
      "           7       0.85      0.94      0.89       102\n",
      "           8       0.83      0.77      0.80        94\n",
      "           9       0.80      0.86      0.83        95\n",
      "\n",
      "    accuracy                           0.88      1000\n",
      "   macro avg       0.88      0.88      0.88      1000\n",
      "weighted avg       0.89      0.88      0.88      1000\n",
      "\n",
      "Error rate when k = 3 is 0.124, and 8 is the most mislabeled.\n",
      "Error rate when k = 5 is 0.118, and 2 is the most mislabeled.\n",
      "Error rate when k = 7 is 0.123, and 2 is the most mislabeled.\n",
      "Error rate when k = 9 is 0.125, and 2 is the most mislabeled.\n"
     ]
    }
   ],
   "source": [
    "k_values = [1, 3, 5, 7, 9]\n",
    "\n",
    "for k in k_values:\n",
    "    # Fit model when k = 1 and output extra analysis\n",
    "    if k == 1:\n",
    "        K1 = KNeighborsClassifier(n_neighbors=k).fit(mini_train_data, mini_train_labels)\n",
    "        predictions = K1.predict(dev_data)\n",
    "        report = classification_report(dev_labels, predictions)\n",
    "        errors = []\n",
    "        for i in range(len(dev_labels)):\n",
    "            if dev_labels[i] != predictions[i]:\n",
    "                errors.append(dev_labels[i])\n",
    "        print(\"Error rate when k = \"+str(k)+\" is \"+str(len(errors)/len(dev_data)) + \\\n",
    "              \", and \"+max(set(errors), key = errors.count)+\" is the most mislabeled.\")\n",
    "        print(report)\n",
    "\n",
    "    # Fit models when k >1\n",
    "    else:\n",
    "        K2 = KNeighborsClassifier(n_neighbors=k).fit(mini_train_data, mini_train_labels)\n",
    "        predictions = K2.predict(dev_data)\n",
    "        errors = []\n",
    "        for i in range(len(dev_labels)):\n",
    "            if dev_labels[i] != predictions[i]:\n",
    "                errors.append(dev_labels[i])\n",
    "        print(\"Error rate when k = \"+str(k)+\" is \"+str(len(errors)/len(dev_data)) + \\\n",
    "              \", and \"+max(set(errors), key = errors.count)+\" is the most mislabeled.\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "7b6YEAzzhWAa"
   },
   "source": [
    "### Produce 1-Nearest Neighbor models using training data of various sizes.  \n",
    "\n",
    "Evaluate and show the performance of each model.  Additionally, show the time needed to measure the performance of each model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "gEpNzDEjhWAa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Error rate when size = 100 is 0.02\n",
      "Elapesed time for 100 observations is 47.23378896713257 seconds or 0.7872298161188761 minutes.\n",
      "Error rate when size = 200 is 0.035\n",
      "Elapesed time for 200 observations is 54.242591857910156 seconds or 0.9040431976318359 minutes.\n",
      "Error rate when size = 400 is 0.035\n",
      "Elapesed time for 400 observations is 67.64456796646118 seconds or 1.1274094661076863 minutes.\n",
      "Error rate when size = 800 is 0.02875\n",
      "Elapesed time for 800 observations is 95.25564813613892 seconds or 1.5875941356023153 minutes.\n",
      "Error rate when size = 1600 is 0.018125\n",
      "Elapesed time for 1600 observations is 107.99214196205139 seconds or 1.7998690327008566 minutes.\n",
      "Error rate when size = 3200 is 0.0090625\n",
      "Elapesed time for 3200 observations is 107.93150806427002 seconds or 1.7988584677378336 minutes.\n",
      "Error rate when size = 6400 is 0.00453125\n",
      "Elapesed time for 6400 observations is 108.11047697067261 seconds or 1.8018412828445434 minutes.\n",
      "Error rate when size = 12800 is 0.002265625\n",
      "Elapesed time for 12800 observations is 108.08983707427979 seconds or 1.8014972845713297 minutes.\n",
      "Error rate when size = 25600 is 0.0011328125\n",
      "Elapesed time for 25600 observations is 108.19199085235596 seconds or 1.8031998475392659 minutes.\n",
      "[0.98, 0.965, 0.965, 0.97125, 0.981875, 0.9909375, 0.99546875, 0.997734375, 0.9988671875]\n"
     ]
    }
   ],
   "source": [
    "train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]\n",
    "accuracies = []\n",
    "\n",
    "# Train a model for each size of data\n",
    "for size in train_sizes:\n",
    "    start = time.time()\n",
    "    K1 = KNeighborsClassifier(n_neighbors=1).fit(train_data, train_labels)\n",
    "    predictions = K1.predict(dev_data[:size])\n",
    "\n",
    "    # Count number of errors and calculate accuracies\n",
    "    errors = []\n",
    "    for i in range(min(size, len(dev_labels))):\n",
    "        if dev_labels[i] != predictions[i]:\n",
    "            errors.append(dev_labels[i])\n",
    "    print(\"Error rate when size = \"+str(size)+\" is \"+str(len(errors)/size))\n",
    "    accuracies.append((size-len(errors))/size)\n",
    "    end = time.time()\n",
    "    print(\"Elapesed time for \"+str(size)+\" observations is \"+str(end-start)+\" seconds or \"+str((end-start)/60)+\" minutes.\")\n",
    "\n",
    "print(accuracies)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "B56lVsKNhWAc"
   },
   "source": [
    "### Produce a regression model that predicts accuracy of a 1-Nearest Neighbor model given training set size. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracies = [0.98,0.965,0.965,0.97125,0.981875,0.9909375,0.99546875,0.997734375,0.9988671875] # Y\n",
    "train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600] # X\n",
    "odds = [i/(1-i) for i in accuracies] # Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-squared standard model: 0.54\n"
     ]
    },
    {
     "data": {
      "image/png": "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80lJSWHBggXUrFlT01QTiLqYJCpFcXMSib8NGzZw//33k5aWxtNPP01aWhpZWVkA1KpVS8khwagFIVEpipuTSHwtWrTo0PUKXbt2pV+/fjRs2DDOUUk8KUFIVLR0ROk0f/581q9fzy233ELTpk157LHH6N69OzVr1jz6i6XUUxeTRCWRl44obdydDz74gFatWtG8eXMeeeQRcnNzKVOmDI888oiSgxyiBCFRScSlI0qjOXPmcNFFF9G2bVvWrl3LiBEjWLp0KWXK6KNAfk5dTBKVRFo6orTZt28fe/fupXLlypQrV47s7GzGjx/PbbfdRnJycrzDk2JMNwwSKaV27drFSy+9xIgRI2jXrh1jxowBONSdJAIF3zBILQiRUua7777jueee48UXXyQrK4urr76aW2655dB2JQeJlhKESCnzxBNPMGrUKDp16sSAAQNIT4/4z6HIUelfCZESbsWKFXTt2pV58+YBMHDgQFatWsXf/vY3JQc5IUoQIiXUp59+Srt27WjUqBFTpkxh9erVANSoUYMGDTS7TE6cuphESqBOnToxZcoUTj/9dJ588kl69epF5cqV4x2WlDJKECIlwMGDB3nvvfdo3749SUlJXHXVVbRs2ZLf/e53nHrqqfEOT0opJQiRYmzfvn1MmDCBYcOGsW7dOt577z1+9atf0atXr3iHJglAYxAixdC+ffsYMmQItWvX5t5776Vq1apMnTqV66+/Pt6hSQJRC0KkGNm/fz/JycmULVuWsWPH0rhxYwYOHEjLli211LYUOSUIkWJg3bp1DB8+nBkzZvDVV19xyimnsGjRIk477bR4hyYJTF1MInG0bNkybrvtNurXr8+4ceO47rrr2Lt3L4CSg8SdWhAicbJs2TIaN25M+fLleeihh+jTpw/VqlWLd1gihyhBiBSR3NxcZs6cyTfffEPv3r1p1KgRY8aMoXPnzlSqVCne4Yn8jFZzLYamLtmsZbVLkYMHD/Lmm28yZMgQVqxYwbnnnsvy5cspW1b/n0n8FbSaq8YgipmpSzYz8N3lbM7KxoHNWdkMfHc5U5dsjndochw+/vhj6tevT9euXcnNzWXixIksW7ZMyUFKBCWIYmZYRibZB3IOK8s+kMOwjMw4RSTHKisri40bNwJQrVo1qlWrxrRp01i+fDldu3alXLlycY5QJDpKEMXMlqzsYyqX4mPr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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The estimated accuracy for 60000 training points is 1.0464348933893306\n",
      "The estimated accuracy for 120000 training points is 1.1166064492908208\n",
      "The estimated accuracy for 1000000 training points is 2.1457892691793443\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "R-squared for log sizes: 0.73\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The estimated accuracy for 60000 training points is 367.8106324657777\n",
      "The estimated accuracy for 120000 training points is 734.6834729418389\n",
      "The estimated accuracy for 1000000 training points is 6115.485133257403\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "R-squared for log sizes vs. odds: 0.64\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The estimated accuracy for 60000 training points is 0.9999998621639768\n",
      "The estimated accuracy for 120000 training points is 0.9999999310852483\n",
      "The estimated accuracy for 1000000 training points is 0.999999991730574\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "R-squared for odds: 1.00\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The estimated accuracy for 60000 training points is 0.9995081755520061\n",
      "The estimated accuracy for 120000 training points is 0.9997531893050895\n",
      "The estimated accuracy for 1000000 training points is 0.9999702871842772\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "train_sizes = np.array(train_sizes).reshape(-1, 1)\n",
    "predict_on = np.array([60000, 120000, 1000000]).reshape(-1, 1)\n",
    "accuracies = np.array(accuracies).reshape(-1,1)\n",
    "x = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600] # Unshaped train_sizes\n",
    "y = [0.98,0.965,0.965,0.97125,0.981875,0.9909375,0.99546875,0.997734375,0.9988671875] # Unshaped accuracies\n",
    "\n",
    "### train_sizes vs. accuracies\n",
    "regr = LinearRegression()\n",
    "regr.fit(train_sizes, accuracies)\n",
    "predictions = regr.predict(accuracies)\n",
    "print('R-squared standard model: %.2f'% regr.score(train_sizes, accuracies))\n",
    "\n",
    "# plot a graph\n",
    "coef = np.polyfit(x, y, 1)\n",
    "fn_line = np.poly1d(coef)\n",
    "plt.plot(x, y, 'o', x, fn_line(x), '--k')\n",
    "plt.title(\"Standard Model\")\n",
    "plt.xlabel(\"Train Sizes\")\n",
    "plt.ylabel(\"Accuracies\")\n",
    "plt.show()\n",
    "\n",
    "# Predict\n",
    "new_pred = regr.predict(predict_on)\n",
    "for i in range(len(predict_on)):\n",
    "    print(\"The estimated accuracy for \"+ str(predict_on[i][0])+ \" training points is \"+ str(new_pred[i][0]))\n",
    "print('''\n",
    "\n",
    "\n",
    "\n",
    "''')\n",
    "\n",
    "\n",
    "### log trin_sizes vs. accuracies\n",
    "log_train_sizes = [np.log(i)[0] for i in train_sizes]\n",
    "regr = LinearRegression()\n",
    "regr.fit(np.array(log_train_sizes).reshape(-1,1), accuracies)\n",
    "predictions = regr.predict(accuracies)\n",
    "print('R-squared for log sizes: %.2f'% regr.score(np.array(log_train_sizes).reshape(-1,1), accuracies))\n",
    "\n",
    "# plot a graph\n",
    "coef = np.polyfit(log_train_sizes, y, 1)\n",
    "fn_line = np.poly1d(coef)\n",
    "plt.plot(log_train_sizes, y, 'o', log_train_sizes, fn_line(log_train_sizes), '--k')\n",
    "plt.title(\"Log train sizes vs. accuracies\")\n",
    "plt.xlabel(\"Log Train Sizes\")\n",
    "plt.ylabel(\"Accuracies\")\n",
    "plt.show() \n",
    "\n",
    "# Predict\n",
    "new_pred = regr.predict(predict_on)\n",
    "for i in range(len(predict_on)):\n",
    "    print(\"The estimated accuracy for \"+ str(predict_on[i][0])+ \" training points is \"+ str(new_pred[i][0]))\n",
    "print('''\n",
    "\n",
    "\n",
    "\n",
    "''')\n",
    "\n",
    "\n",
    "### log train_sizes vs. odds\n",
    "odds = [a/(1-a) for a in y]\n",
    "regr = LinearRegression()\n",
    "regr.fit(np.array(log_train_sizes).reshape(-1,1), np.array(odds).reshape(-1,1))\n",
    "predictions = regr.predict(regr.predict(np.array(odds).reshape(-1,1)))\n",
    "print('R-squared for log sizes vs. odds: %.2f'% regr.score(np.array(log_train_sizes).reshape(-1,1), odds))\n",
    "\n",
    "#plot a graph\n",
    "# note: the graph shows train_sizes vs. odds, NOT train_sizes vs. accuracy\n",
    "coef = np.polyfit(log_train_sizes, odds, 1)\n",
    "fn_line = np.poly1d(coef)\n",
    "plt.plot(log_train_sizes, odds, 'o', log_train_sizes, fn_line(log_train_sizes), '--k')\n",
    "plt.title(\"Log train sizes vs. odds\")\n",
    "plt.xlabel(\"Log Train Sizes\")\n",
    "plt.ylabel(\"Odds\")\n",
    "plt.show()\n",
    "\n",
    "# Predict\n",
    "# Final printed answer is accuracy, not odds\n",
    "new_pred = regr.predict(predict_on)\n",
    "un_odds = [i[0]/(1+i[0]) for i in new_pred]\n",
    "for i in range(len(predict_on)):\n",
    "    print(\"The estimated accuracy for \"+ str(predict_on[i][0])+ \" training points is \"+ str(un_odds[i]))\n",
    "print('''\n",
    "\n",
    "\n",
    "\n",
    "''')    \n",
    "\n",
    "\n",
    "\n",
    "### trian_sizes vs. odds\n",
    "regr = LinearRegression()\n",
    "regr.fit(train_sizes, odds)\n",
    "predictions = regr.predict(np.array(odds).reshape(-1,1))\n",
    "print('R-squared for odds: %.2f'% regr.score(train_sizes, odds))\n",
    "\n",
    "# plot a graph\n",
    "# note: the graph shows train_sizes vs. odds, NOT train_sizes vs. accuracy\n",
    "coef = np.polyfit(x, odds, 1)\n",
    "fn_line = np.poly1d(coef)\n",
    "plt.plot(x, odds, 'o', x, fn_line(x), '--k')\n",
    "plt.title(\"Train sizes vs. odds\")\n",
    "plt.xlabel(\"Train Sizes\")\n",
    "plt.ylabel(\"Odds\")\n",
    "plt.show()\n",
    "\n",
    "# Predict\n",
    "# Final printed answer is accuracy, not odds\n",
    "new_pred = regr.predict(predict_on)\n",
    "un_odds = [(i)/(1+i) for i in new_pred]\n",
    "for i in range(len(predict_on)):\n",
    "    print(\"The estimated accuracy for \"+ str(predict_on[i][0])+\" training points is \"+ str(un_odds[i]))\n",
    "print('''\n",
    "\n",
    "\n",
    "\n",
    "''')    \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "HYYYL9cGhWAe"
   },
   "source": [
    "**In the first model, the train_sizes vs. accuracies, the estimated accuracies for all the large training sizes were greater than 1 (which is impossible) and an R-squared of .54. The graph seemed to be a logrithmic, rather than linear curve, so I took the log of train_sizes and ran the model again.**\n",
    "\n",
    "**The second model (log of train_sizes vs. accuracies) improved on the first, with an R-squared of .73. The line does seem to fit the data better. However, the results are well over 1.**\n",
    "\n",
    "**The third model looks at log of accuracies vs. odds. Odds is a transformation of accuracies, which scales them from one to infinity rather than from zero to one using this formula: odds(y) = y/(1-y). The R-squared value decreased a to .63. Here, the output below the graph is accuracy, not odds. The accuracies are now within the range of possibility.**\n",
    "\n",
    "**The final model, train_sizes vs. odds, gives the best R-squared of almost 1. Here, the output below the graph is accuracy, not odds. All of the accuracies are below 1.**\n",
    "\n",
    "**There is a common trend in all these models: the accuracy increases with more data points, and we know that accuracy will approach 1 with more and more data.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "geAQJjGRhWAe"
   },
   "source": [
    "### Produce a 1-Nearest Neighbor model and show the confusion matrix. \n",
    "\n",
    "Which pair of digits does the model confuse most often? Show the images of these most often confused digits.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[101   0   1   0   0   0   1   1   2   0]\n",
      " [  0 116   1   0   0   0   0   0   1   0]\n",
      " [  1   4  84   2   2   0   2   4   6   1]\n",
      " [  0   2   0  84   0   6   0   2   3   0]\n",
      " [  0   0   1   0  78   0   0   2   0  11]\n",
      " [  2   0   0   1   1  77   5   0   2   0]\n",
      " [  1   2   1   0   1   2  94   0   1   0]\n",
      " [  0   1   1   0   0   0   0  96   0   4]\n",
      " [  1   5   4   3   1   3   0   1  72   4]\n",
      " [  0   1   0   0   3   2   0   7   0  82]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## This one determines which digits are most frequently confused for each other.\n",
    "    \n",
    "# Train model and produce matrix\n",
    "K1 = KNeighborsClassifier(n_neighbors=1).fit(mini_train_data, mini_train_labels)\n",
    "predictions = K1.predict(dev_data)\n",
    "matrix = confusion_matrix(dev_labels, predictions)\n",
    "print(matrix)\n",
    "\n",
    "# Count mislabeled images and print an example of each of the most mislabeled\n",
    "mislabel_count = []\n",
    "m = 0 # Flag highest number of mistakes\n",
    "for digit1 in range(10):\n",
    "    for digit2 in range((digit1 +1), 10):\n",
    "        confused_pair = matrix[digit1, digit2] + matrix[digit2, digit1]\n",
    "        if confused_pair > m:\n",
    "            m = confused_pair\n",
    "            mislabel_count = [digit1, digit2]\n",
    "        elif confused_pair == m: # if there is a tie\n",
    "            mislabel_count.append(digit1)\n",
    "            mislabel_count.append(digit2)\n",
    "for label in mislabel_count:\n",
    "    yy = np.where(Y==str(label))[0][:1]\n",
    "    plt.imshow(np.reshape(X[yy], (28,28)), interpolation = 'nearest', cmap = \"gray_r\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Bq36xaQohWAf"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[101   0   1   0   0   0   1   1   2   0]\n",
      " [  0 116   1   0   0   0   0   0   1   0]\n",
      " [  1   4  84   2   2   0   2   4   6   1]\n",
      " [  0   2   0  84   0   6   0   2   3   0]\n",
      " [  0   0   1   0  78   0   0   2   0  11]\n",
      " [  2   0   0   1   1  77   5   0   2   0]\n",
      " [  1   2   1   0   1   2  94   0   1   0]\n",
      " [  0   1   1   0   0   0   0  96   0   4]\n",
      " [  1   5   4   3   1   3   0   1  72   4]\n",
      " [  0   1   0   0   3   2   0   7   0  82]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## This one determines which digits are most frequently misidentified.'''\n",
    "    \n",
    "# Train model and produce matrix\n",
    "K1 = KNeighborsClassifier(n_neighbors=1).fit(mini_train_data, mini_train_labels)\n",
    "predictions = K1.predict(dev_data)\n",
    "matrix = confusion_matrix(dev_labels, predictions)\n",
    "print(matrix)\n",
    "\n",
    "# Count mislabeled images and print an example of each of the most mislabeled\n",
    "mislabel_count = []\n",
    "for num in matrix:\n",
    "    mislabel_count.append(sum(num) - max(num))\n",
    "max_mis = max(mislabel_count)\n",
    "most_mislabeled = [i for i, j in enumerate(mislabel_count) if j == max_mis]\n",
    "for label in most_mislabeled:\n",
    "    yy = np.where(Y==str(label))[0][:1]\n",
    "    plt.imshow(np.reshape(X[yy], (28,28)), interpolation = 'nearest', cmap = \"gray_r\")\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "tgqMKb-hhWAh"
   },
   "source": [
    "### Blurring\n",
    "\n",
    "Implement a simplified Gaussian blur by just using the 8 neighboring pixels like this: the smoothed value of a pixel is a weighted combination of the original value and the 8 neighboring values.\n",
    "\n",
    "Apply your blur filter in 3 ways:\n",
    "- Filter the training data but not the dev data\n",
    "- Filter the dev data but not the training data\n",
    "- Filter both training data and dev data\n",
    "\n",
    "Show the accuracy resulting no filter and from each way you apply the filter."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "lSKHmHGshWAi"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "When no data is filtered, accuracy is 0.116\n",
      "When only training data is filtered, accuracy is 0.879\n",
      "When only dev data is filtered, accuracy is 0.906\n",
      "When all data is filtered, accuracy is 0.879\n"
     ]
    }
   ],
   "source": [
    "# I assume it is okay to not blur the first or last line since they are all black.\n",
    "# I assume k=1\n",
    "\n",
    "# Filter the training data\n",
    "train_filtered = []\n",
    "for image in mini_train_data:\n",
    "    train_filtered.append(image.copy())\n",
    "for image in train_filtered:\n",
    "    for i in range(28, len(image)-29):\n",
    "        image[i] = mini_train_data[0][i]*.2 + mini_train_data[0][i-29]*.1 + mini_train_data[0][i-28]*.1 + mini_train_data[0][i-27]*.1 + mini_train_data[0][i-1]*.1 + mini_train_data[0][i+1]*.1 + mini_train_data[0][i+27]*.1 + mini_train_data[0][i+28]*.1 + mini_train_data[0][i+29]*.1\n",
    "\n",
    "# Filter the dev data\n",
    "dev_filtered = []\n",
    "for image in dev_data:\n",
    "    dev_filtered.append(image.copy())\n",
    "for image in dev_filtered:\n",
    "    for i in range(28, len(image)-29):\n",
    "        image[i] = dev_data[0][i]*.2 + dev_data[0][i-29]*.1 + dev_data[0][i-28]*.1 + dev_data[0][i-27]*.1 + dev_data[0][i-1]*.1 + dev_data[0][i+1]*.1 + dev_data[0][i+27]*.1 + dev_data[0][i+28]*.1 + dev_data[0][i+29]*.1\n",
    "\n",
    "# No filtered data\n",
    "K1 = KNeighborsClassifier(n_neighbors=1).fit(mini_train_data, mini_train_labels)\n",
    "predictions = K1.predict(dev_data)\n",
    "errors = []\n",
    "for i in range(len(dev_labels)):\n",
    "    if dev_labels[i] != predictions[i]:\n",
    "        errors.append(dev_labels[i])\n",
    "print(\"When no data is filtered, accuracy is \"+ str(len(errors)/len(mini_train_data)))\n",
    "\n",
    "# Filtered training data but not dev data\n",
    "K1 = KNeighborsClassifier(n_neighbors=1).fit(train_filtered, mini_train_labels)\n",
    "predictions = K1.predict(dev_data)    \n",
    "errors = []\n",
    "for i in range(len(dev_labels)):\n",
    "    if dev_labels[i] != predictions[i]:\n",
    "        errors.append(dev_labels[i])\n",
    "print(\"When only training data is filtered, accuracy is \"+ str(len(errors)/len(mini_train_data)))\n",
    "\n",
    "# Filtered dev data but not training data\n",
    "K1 = KNeighborsClassifier(n_neighbors=1).fit(mini_train_data, mini_train_labels)\n",
    "predictions = K1.predict(dev_filtered)\n",
    "errors = []\n",
    "for i in range(len(dev_labels)):\n",
    "    if dev_labels[i] != predictions[i]:\n",
    "        errors.append(dev_labels[i])\n",
    "print(\"When only dev data is filtered, accuracy is \"+ str(len(errors)/len(mini_train_data)))\n",
    "\n",
    "# Filtered both training data and dev data\n",
    "K1 = KNeighborsClassifier(n_neighbors=1).fit(train_filtered, mini_train_labels)\n",
    "predictions = K1.predict(dev_filtered)\n",
    "errors = []\n",
    "for i in range(len(dev_labels)):\n",
    "    if dev_labels[i] != predictions[i]:\n",
    "        errors.append(dev_labels[i])\n",
    "print(\"When all data is filtered, accuracy is \"+ str(len(errors)/len(mini_train_data)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "LtgepWfAhWAk"
   },
   "source": [
    "### Naive Bayes\n",
    "\n",
    "For the first model, map pixel values to either 0 or 1. Use some reasonable threshold to separate white from black.  Use `BernoulliNB` to produce the model.\n",
    "\n",
    "For the second model, map pixel values to either 0, 1, or 2, representing white, gray, or black. Use some reasonable thresholds to separate white from gray from black.  Use `MultinomialNB` to produce the model. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "eGpH-4IQhWAk"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "In the black and white binomial trial, accuracy is 0.809\n",
      "In the black, gray, and white multinomial trial, accuracy is 0.807\n"
     ]
    }
   ],
   "source": [
    "##Pixels are black and white\n",
    "bw_train = []\n",
    "for image in mini_train_data:\n",
    "    bw_train.append(image.copy())\n",
    "for image in bw_train:\n",
    "    for i in range(len(image)):\n",
    "        if i <5:\n",
    "            i = 0\n",
    "        else:\n",
    "            i = 1\n",
    "\n",
    "bw_dev = []\n",
    "for image in dev_data:\n",
    "    bw_dev.append(image.copy())\n",
    "for image in bw_dev:\n",
    "    for i in range(len(image)):\n",
    "        if i <5:\n",
    "            i = 0\n",
    "        else:\n",
    "            i = 1\n",
    "\n",
    "##Pixels are black, gray, and white\n",
    "bgw_train = []\n",
    "for image in mini_train_data:\n",
    "    bgw_train.append(image.copy())\n",
    "for image in bgw_train:\n",
    "    for i in range(len(image)):\n",
    "        if i < (1/3):\n",
    "            i = 0\n",
    "        elif i > (2/3):\n",
    "            i = 2\n",
    "        else:\n",
    "            i = 1\n",
    "\n",
    "bgw_dev = []\n",
    "for image in dev_data:\n",
    "    bgw_dev.append(image.copy())\n",
    "for image in bgw_dev:\n",
    "    for i in range(len(image)):\n",
    "        if i < (1/3):\n",
    "            i = 0\n",
    "        elif i > (2/3):\n",
    "            i = 2\n",
    "        else:\n",
    "            i = 1\n",
    "\n",
    "# Run black and white binomial trial\n",
    "bw = BernoulliNB()\n",
    "bw.fit(bw_train, mini_train_labels)\n",
    "print(\"In the black and white binomial trial, accuracy is \"+ str(bw.score(bw_dev, dev_labels)))\n",
    "\n",
    "# Run black, gray, and white multinomial trial\n",
    "bgw = MultinomialNB()\n",
    "bgw.fit(bgw_train, mini_train_labels)\n",
    "print(\"In the black, gray, and white multinomial trial, accuracy is \"+ str(bgw.score(bgw_dev, dev_labels)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "zNLrgggohWAm"
   },
   "source": [
    "**The multinomial model did not improve on the results because the multinomial model introduces more noise, which then leads to overfitting.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "PqjbRLg7hWAm"
   },
   "source": [
    "### LaPlace Smoothing\n",
    "\n",
    "Search across several values for the LaPlace smoothing parameter (alpha) to find its effect on a Bernoulli Naive Bayes model's performance.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#This cell makes Python ignore one of the useless warnings I was getting\n",
    "import warnings\n",
    "warnings.simplefilter(action='ignore', category=FutureWarning) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/pattidegner/opt/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:814: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.\n",
      "  DeprecationWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.804 0.818 0.822 0.823 0.82  0.818 0.817 0.807 0.77 ]\n",
      "\n",
      "Best alpha =  {'alpha': 0.01}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/pattidegner/opt/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:814: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.\n",
      "  DeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "def LaPlace_alpha(alphas):\n",
    "    '''Tests various alphas to determine which one produces the best model.'''\n",
    "    \n",
    "    model = GridSearchCV(BernoulliNB(), alphas)\n",
    "    model.fit(mini_train_data, mini_train_labels)\n",
    "    print(model.cv_results_['mean_test_score'])\n",
    "    return model.fit(mini_train_data, mini_train_labels)\n",
    "    \n",
    "\n",
    "alphas = {'alpha': [1.0e-10, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}\n",
    "nb = LaPlace_alpha(alphas)\n",
    "print()\n",
    "print(\"Best alpha = \", nb.best_params_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "1yEg9keThWAp"
   },
   "source": [
    "**The best alpha is .01 with an accuracy of .823. The accuracy of the model when it is near zero is .804. It makes sense that zero is not the best because then the boundary line between digits is less smooth (overfitted). The higher alpha smoothes the line and increases accuracy, but only to a point. After that it decreases accuracy as the line doesn't fit the data.** "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "B07GDiDdhWAq"
   },
   "source": [
    "### Guassian Naive Bayes\n",
    "\n",
    "Apply a simple fix to this model so that the model accuracy is around the same as for a Bernoulli Naive Bayes model. Show the model accuracy before your fix and the model accuracy after your fix. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "gBLbTMWChWAq"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Initial model has accuracy 0.593\n",
      " avg theta = 0.13233917230617082 and avg sigma =0.05299707504881932\n",
      "\n",
      "Final model has accuracy 0.817 with a smoothing factor of 0.1\n",
      " avg theta = 0.13233917230617082 and avg sigma =0.07314040555249945\n"
     ]
    }
   ],
   "source": [
    "# Create basic model\n",
    "model = GaussianNB()\n",
    "model.fit(mini_train_data, mini_train_labels)\n",
    "theta_list = []\n",
    "for theta in model.theta_:\n",
    "    theta_list.append(np.mean(theta))\n",
    "sigma_list = []\n",
    "for sigma in model.sigma_:\n",
    "    sigma_list.append(np.mean(sigma))\n",
    "print(\"Initial model has accuracy \" + str(model.score(dev_data, dev_labels))+\\\n",
    "      \"\\n avg theta = \"+str(np.mean(theta_list))+\\\n",
    "      \" and avg sigma =\"+str(np.mean(sigma_list)))\n",
    "\n",
    "# Identify best smoothing\n",
    "smoothed_list = []\n",
    "var_smoothing_options = [1.0e-10, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]\n",
    "for smooth in var_smoothing_options:\n",
    "    model2 = GaussianNB(var_smoothing = smooth)\n",
    "    model2.fit(mini_train_data, mini_train_labels)\n",
    "    smoothed_list.append(model2.score(dev_data, dev_labels))\n",
    "\n",
    "# Create final model with smoothing factor that maximizes accuracy\n",
    "best = smoothed_list.index(max(smoothed_list))\n",
    "final_model = GaussianNB(var_smoothing = var_smoothing_options[best])\n",
    "final_model.fit(mini_train_data, mini_train_labels)\n",
    "theta_list = []\n",
    "for theta in final_model.theta_:\n",
    "    theta_list.append(np.mean(theta))\n",
    "sigma_list = []\n",
    "for sigma in final_model.sigma_:\n",
    "    sigma_list.append(np.mean(sigma))\n",
    "print()\n",
    "print(\"Final model has accuracy \" + str(final_model.score(dev_data, dev_labels))+\\\n",
    "      \" with a smoothing factor of \"+str(var_smoothing_options[best])+\"\\n avg theta = \"+\\\n",
    "        str(np.mean(theta_list))+\" and avg sigma =\"+str(np.mean(sigma_list)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "1SyHTEJohWAt"
   },
   "source": [
    "**The simple fix used here is to introduce a smoothing parameter to the Gaussian model. The average mean (theta) stays the same between the original model and the model with smoothing, but the varince (sigma) increases as the smoothing parameter increases.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "dgZMuc1VhWAt"
   },
   "source": [
    "### Generate digit images\n",
    "\n",
    "Produce a Bernoulli Naive Bayes model and then use it to generate a 10x20 grid with 20 example images of each digit. Each pixel output should be either 0 or 1, based on comparing some randomly generated number to the estimated probability of the pixel being either 0 or 1.  Show the grid."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "ktii-Mp-hWAu"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 200 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axs = plt.subplots(10, 20) #sets up Nx10 grid\n",
    "model = BernoulliNB().fit(X,Y)\n",
    "\n",
    "# Generate images\n",
    "for i in range(10):\n",
    "    for j in range(20):\n",
    "        num = np.exp(model.feature_log_prob_[i])\n",
    "        for k, pix in enumerate(num):\n",
    "            n = np.random.rand()\n",
    "            if pix>n:\n",
    "                num[k] = 1\n",
    "            else:\n",
    "                num[k] = 0\n",
    "\n",
    "        # Put each image in its place\n",
    "        axs[i, j].imshow(num.reshape(28,28), interpolation = \"nearest\", cmap = \"gray_r\")\n",
    "        axs[i, j].axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "SuQd1fTGhWAw"
   },
   "source": [
    "**These images are hazier due to the randomization aspect. Most of them bear a resemblance to the actual number, but for some, especially 4, 5, and 9, it is hard to tell what the number is supposed to be. Even though the pixel is compared to a random number, we can still see what the number is supposed ot be, for the most part. This shows how powerful Naive Bayes can be.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "ksHMg73uhWAx"
   },
   "source": [
    "### Calibration Analysis\n",
    "\n",
    "Recall that a strongly calibrated classifier is rougly 90% accurate when the posterior probability of the predicted class is 0.9. A weakly calibrated classifier is more accurate when the posterior probability of the predicted class is 90% than when it is 80%. A poorly calibrated classifier has no positive correlation between posterior probability and accuracy.  \n",
    "\n",
    "Produce a Bernoulli Naive Bayes model.  Evaluate performance: partition the dev set into several buckets based on the posterior probabilities of the predicted classes - think of a bin in a histogram- and then estimate the accuracy for each bucket. So, for each prediction, find the bucket to which the maximum posterior probability belongs, and update \"correct\" and \"total\" counters accordingly.  Show the accuracy for each bucket."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "a1N-St12hWAy"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "p(pred) is 0.0000000000000 to 0.5000000000000    total =   0    accuracy = 0.000\n",
      "p(pred) is 0.5000000000000 to 0.9000000000000    total =  31    accuracy = 0.355\n",
      "p(pred) is 0.9000000000000 to 0.9990000000000    total =  67    accuracy = 0.433\n",
      "p(pred) is 0.9990000000000 to 0.9999900000000    total =  59    accuracy = 0.458\n",
      "p(pred) is 0.9999900000000 to 0.9999999000000    total =  46    accuracy = 0.652\n",
      "p(pred) is 0.9999999000000 to 0.9999999990000    total =  62    accuracy = 0.774\n",
      "p(pred) is 0.9999999990000 to 0.9999999999900    total =  33    accuracy = 0.788\n",
      "p(pred) is 0.9999999999900 to 0.9999999999999    total =  43    accuracy = 0.791\n",
      "p(pred) is 0.9999999999999 to 1.0000000000000    total = 659    accuracy = 0.938\n"
     ]
    }
   ],
   "source": [
    "buckets = [0.5, 0.9, 0.999, 0.99999, 0.9999999, 0.999999999, 0.99999999999, 0.9999999999999, 1.0]\n",
    "correct = [0 for i in buckets]\n",
    "total = [0 for i in buckets]\n",
    "\n",
    "# Create model. I assume alpha = .001 is okay.\n",
    "model = BernoulliNB(alpha = .001).fit(mini_train_data, mini_train_labels)\n",
    "\n",
    "for i in range(len(dev_data)):\n",
    "\n",
    "    # Determine bin\n",
    "    max_guess = max(model.predict_proba(dev_data)[i])\n",
    "    bucket = 15\n",
    "    for bins in buckets:\n",
    "        if bucket < 10:\n",
    "            continue\n",
    "        elif bins < max_guess:\n",
    "            continue\n",
    "        else:\n",
    "            bucket = buckets.index(bins)\n",
    "\n",
    "    # Determine if correct label, if so add to correct bin\n",
    "    guess = model.predict(dev_data)[i]\n",
    "    true_label = dev_labels[i]\n",
    "    if guess == true_label:\n",
    "        correct[bucket] += 1\n",
    "\n",
    "    # Add to total\n",
    "    total[bucket]+=1\n",
    "\n",
    "\n",
    "for i in range(len(buckets)):\n",
    "    accuracy = 0.0\n",
    "    if (total[i] > 0): accuracy = correct[i] / total[i]\n",
    "    print('p(pred) is %.13f to %.13f    total = %3d    accuracy = %.3f' % (0 if i==0 else buckets[i-1], buckets[i], total[i], accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "h-4qQsrrhWA1"
   },
   "source": [
    "**When the posterior probability is .9, the accuracy is only 43.3%, not 90% as we would expect from a strongly calibrated classifier. However, the accuracy increases as the posterior probability increases. This correlation indicates that this is a weakly calibrated classifier. If it were poorly classified, the accuracy would not increase with the posterior probability.**"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "name": "firstname_lastname_p1.ipynb",
   "private_outputs": true,
   "provenance": [
    {
     "file_id": "https://github.com/MIDS-W207/Master/blob/master/Projects/firstname_lastname_p1.ipynb",
     "timestamp": 1557957807607
    }
   ],
   "version": "0.3.2"
  },
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
   "version": "3.7.4"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": false,
   "sideBar": true,
   "skip_h1_title": true,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
