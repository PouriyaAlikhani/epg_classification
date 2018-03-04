import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import itertools
import sys




d0 = {3:'JD29d003.abf',
        8:'JD29d008.abf',
        13:'JD29d013.abf',
        18:'JD29d018.abf',
        23:'JD29d023.abf',
        28:'JD29d028.abf',
        33:'JD29d033.abf',
        38:'JD29d038.abf',
        43:'JD29d043.abf',
        48:'JD29d048.abf'}
d0 = {  2:'JD29003.abf',
        12:'JD29006.abf',
        22:'JD29007.abf',
        32:'JD29016.abf',
        42:'JD29018.abf'}


def findPath(k, dic):
    """
    DESCRIPTION
        this method returns the subpath of the each file
    """
    if (cwd == 'C:\Users\Pouria\Documents\Python Scripts\JD\\'):
        return 'data\\Baseline\\'
    elif (cwd == '/data/users/palikh/JoeDent/'):
        return 'data/Baseline/'


def balanced_index(dic, n_samples=40000, window_size = 0.5, pump_type = 'major'):
    def load_r(file, pump_type = 'major'):
        if (pump_type == 'minor'):
            record = pd.read_pickle(cwd + subPath + file.replace('.abf', ' Full Annotation.cPickle'))
        elif (pump_type == 'major'):
            record = pd.read_pickle(cwd + subPath + file.replace('.abf', ' Annotation.cPickle'))
        return record
    files = sorted(dic.values())
    n_files = len(files)
    r = load_r(file = files[0])
    classes = list(r['class'].unique())
    Total_Length = r.shape[0]
    N = np.zeros(shape = (n_files, len(classes)))
    R = np.zeros(shape = (n_files, len(classes)))
    all_idx = np.zeros(shape=(n_samples)) - 1
    file_idx = 0
    for file in files:
        r = load_r(file, 'major')
        # N contains the number of samples selected from different classes in a given recording file.
        N[file_idx, :] = [r[r['class'] == class_idx].shape[0] for class_idx in classes]
        file_idx += 1
    for class_idx in classes:
        R[:, class_idx] = [int(N[file_idx, class_idx]/N[:, class_idx].sum() * n_samples / len(classes)) for file_idx in range(n_files)]
    R[0, 0] += n_samples - R.sum()
    file_idx = 0
    current_idx = 0
    for file in files:
        r = load_r(file, 'major')
        for class_idx in classes:
            step = int(R[file_idx, class_idx])
            all_idx[current_idx:current_idx+step] = np.random.choice(a=r[r['class']==class_idx].index, size=step) + Total_Length * file_idx
            current_idx += step
        file_idx += 1
    # return all_idx
    # return all_idx
    print 'R[<file>, <class>]\n', R
    np.random.shuffle(all_idx)
    pd.to_pickle(all_idx, '%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_samples))
    return extract_recording_by_idx(all_idx=all_idx.astype(int), dict1=dic, window_size = window_size, pump_type = pump_type)



def tabulate_single_output(window_size, dict1 = d0, n_window = 100000, pump_type = 'minor'):
    """
    DESCRIPTION
        this method extracts n_window number of windows of length window_size (in seconds) from the files indicated in dic1 and returns them as a numpy matrix.

    DEPENDENCIES
    - Packages
        numpy package
        pandas package(immidiately converted to numpy matrix)
    - Files
        = Full Annotations as cPickle files, passed as a dictionary to dict1
        = ML_all_idx.cPickle
    """
    window_size = int(window_size * 10000)
    total_recording_length = 3100000
    n_frame = total_recording_length // window_size
    middle_idx = window_size//2
    keys = sorted(dict1.keys())
    n_files = len(keys)
    if (os.path.isfile('%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_window))):
        all_idx = np.array(sorted(pickle.load(open('%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_window), 'rb'))))
        print('Read from file')
    else:
        import random
        all_idx = np.array(random.sample(range(total_recording_length * n_files), k = n_window))
        np.random.shuffle(all_idx)
        pickle.dump(all_idx, open('%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_window), 'wb'))
        all_idx = np.sort(all_idx)
        print('Create New Index File')
    # data = 0
    # data = extract_recording_by_idx(all_idx=all_idx, dict1=dict1, window_size=window_size)
    # return all_idx
    return extract_recording_by_idx(all_idx=all_idx, dict1=dict1, window_size = window_size, pump_type = pump_type)




def extract_recording_by_idx_fixed(all_idx, dict1, window_size, pump_type = 'minor'):
    """
    DESCRIPTION
        This method collects the desired portion of all recordings that correspond to the list of indeces it is given.
        This method uses a fixed length of recording.
    """
    all_idx = np.array(all_idx)
    # window_size = int(window_size * 10000)
    n_window = len(all_idx)
    total_recording_length = 3000000
    # in case window_size is in second units
    if window_size < 100:
        window_size = int(window_size * 10000)
    n_frame = total_recording_length // window_size
    middle_idx = window_size//2
    files = sorted(dict1.values())
    n_files = len(files)
    data = np.zeros(shape=(n_window, window_size + 1))
    current_vector = 0
    for file_idx in range(n_files):
        current_file_idx = file_idx
        if (pump_type == 'minor'):
            record = pd.read_pickle(cwd + subPath + files[file_idx].replace('.abf', ' Full Annotation.cPickle')).as_matrix()[:,1:]
        elif (pump_type == 'major'):
            record = pd.read_pickle(cwd + subPath + files[file_idx].replace('.abf', ' Annotation.cPickle')).as_matrix()[:,1:]
        if (record[:, 0].min() < 0):
            record[:, 0] -= record[:, 0].min()
        Mat = np.ones(shape = (total_recording_length+window_size - 1, 2))*(-2)
        Mat[middle_idx:middle_idx + total_recording_length,:] = record.copy()
        # following extracts the middle indeces for a given file index
        local_idx = all_idx[all_idx / total_recording_length == file_idx] % total_recording_length
        # verifies that the current file includes some windows
        if (local_idx.shape[0]):
            data[current_vector:current_vector + local_idx.shape[0], -1] = Mat[local_idx + middle_idx, 1]
            # following extracts the middle indeces with a bound
            local_idx_range = np.array([xrange(idx_instant, idx_instant + 2*middle_idx) for idx_instant in local_idx])
            data[current_vector:current_vector+local_idx.shape[0], :-1] = Mat[local_idx_range ,0]
        else:
            print ('No windows from %s.' % (files[file_idx]))

        current_vector += local_idx.shape[0]
    print 'data distribution\n', pd.Series(data[:,-1]).value_counts()
    return data




def extract_recording_by_idx(all_idx, dict1, window_size, pump_type = 'minor'):
    """
    DESCRIPTION
        This method collects the desired portion of all recordings that correspond to the list of indeces it is given.
        This method can be used for different total length of recording.
    """
    all_idx = np.array(all_idx)
    # window_size = int(window_size * 10000)
    n_window = len(all_idx)
    # total_recording_length = 3000000
    # in case window_size is in second units
    if window_size < 100:
        window_size = int(window_size * 10000)
    # n_frame = total_recording_length // window_size
    middle_idx = window_size//2
    files = sorted(dict1.values())
    n_files = len(files)
    data = np.zeros(shape=(n_window, window_size + 1))
    current_vector = 0
    for file_idx in range(n_files):
        current_file_idx = file_idx
        if (pump_type == 'minor'):
            record = pd.read_pickle(cwd + subPath + files[file_idx].replace('.abf', ' Full Annotation.cPickle')).as_matrix()[:,1:]
        elif (pump_type == 'major'):
            record = pd.read_pickle(cwd + subPath + files[file_idx].replace('.abf', ' Annotation.cPickle')).as_matrix()[:,1:]
        # print(file_idx, record.shape)
        total_recording_length = record.shape[0]
        if (record[:, 0].min() < 0):
            record[:, 0] -= record[:, 0].min()
        Mat = np.ones(shape = (total_recording_length+window_size - 1, 2))*(-2)
        Mat[middle_idx:middle_idx + total_recording_length,:] = record.copy()
        # following extracts the middle indeces for a given file index
        local_idx = all_idx[all_idx // total_recording_length == file_idx] % total_recording_length
        # verifies that the current file includes some windows
        if (local_idx.shape[0]):
            data[current_vector:current_vector + local_idx.shape[0], -1] = Mat[local_idx + middle_idx, 1]
            # following extracts the middle indeces with a bound
            local_idx_range = np.array([range(idx_instant, idx_instant + 2*middle_idx) for idx_instant in local_idx])
            data[current_vector:current_vector+local_idx.shape[0], :-1] = Mat[local_idx_range ,0]
        else:
            print ('No windows from %s.' % (files[file_idx]))

        current_vector += local_idx.shape[0]
    print 'data distribution\n', pd.Series(data[:,-1]).value_counts()
    return data

# validation_ratio = 0.2; test_set_ratio = 0.1; clf = MLPClassifier; n_folds = 5; dic = d0








import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
train_scores, valid_scores = validation_curve(Ridge(max_iter=20), X, y, "alpha", np.logspace(-7, 3, 3), verbose = True)
# train_scores = 
plt.plot(train_scores, color='b')
plt.plot(valid_scores, color='r')
plt.show()


def classifier_single_output(data, validation_ratio = 0.2,n_iter = 100, test_set_ratio = 0.1, clf =MLPClassifier , n_folds = 5, dic = d0, classifier_lab='NN'):
    """
    DESCRIPTION
        this method receives a numpy matrix; ratio of leftout validation portion; and ratio of test portion
    """
    subPath = findPath(k=dic.keys()[0], dic=dic)
    n_window = data.shape[0]
    window_size = data.shape[1]
    valid_start = int(n_window*test_set_ratio)
    test_array = np.zeros(shape = (n_folds, int(test_set_ratio*n_window)))

    X = data[valid_start:,:-1]
    y = data[valid_start:,-1]

    kf = KFold(n_splits = n_folds)
    kf.get_n_splits(X)

    fold_idx = 0
    for train_idx, valid_idx in kf.split(X=X, y=y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        print 'fold_idx:', fold_idx
        add_report(window_size = window_size, fold_idx=fold_idx, subPath = subPath)
        rft = clf(verbose=True, max_iter=n_iter, learning_rate='adaptive', solver='adam', learning_rate_init=0.0001)
        # validation_curve(estimator=rft, X_train, y_train, param_name="alpha",param_range=np.logspace(-7, 3, 3), groups=None, cv=None, scoring=None, n_jobs=1, pre_dispatch=’all’, verbose=0)
        rft.fit(X = X_train, y = y_train)
        valid_pred = rft.predict(X = X_valid)
        train_pred = rft.predict(X = X_train)
        # pickle.dump([valid_idx, valid_pred], open('%s%svalid prediction %s - window size %.1f fold index %d.cPickle' % (cwd, subPath, str(clf).split('(')[0], float(window_size)/10000.0    , fold_idx), 'wb'))

        test_array[fold_idx] = rft.predict(X = data[:valid_start,:-1])
        # test_array[fold_idx, 1, :] = test_pred.T
        # pickle.dump([test_array[0, 0, :].astype(int), test_pred], open('%s%stest prediction %s - window size %.1f fold index %d.cPickle' % (cwd, subPath, str(clf).split('(')[0], float(window_size)/10000.0, fold_idx), 'wb'))
        fold_idx += 1
    # test_result = np.array([np.argmax(np.bincount(i.astype(int))) for i in test_array.T])
    pickle.dump(test_array, open('%s%stest pred %s - ws_%.1f n_%d.cPickle' % (cwd, subPath, classifier_lab, float(window_size)/10000.0, n_window), 'wb'))
    # return test_result, test_array






def load_pred(window_size=0.5, n_window=100000, classifier_lab='NN'):
    """
    DESCRIPTION
        this method load the predicted labels of the test group
    """
    return pickle.load(open('%s%stest pred %s - ws_%.1f n_%d.cPickle' % (cwd, subPath, classifier_lab, window_size, n_window), 'rb'))



def load_idx(n_window=100000, test=False, test_set_ratio = 0.1):
    if (test):
        return np.array(sorted(pickle.load(open('%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_window), 'rb'))))[:int(n_window * test_set_ratio)]
    return np.array(sorted(pickle.load(open('%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_window), 'rb'))))





def add_report(window_size, fold_idx, subPath):
    """
    DESCRIPTION
        this method adds a report line to keep track of the progress.
    """
    # checks if the OS is UNIX based
    if ('/' in cwd):
        # report file was not previously created
        if (os.path.isfile(cwd + subPath + 'ML_Result/EPG Classification Progress Report.txt')):
            f = open(cwd + subPath + 'ML_Result/EPG Classification Progress Report.txt', 'rb')
            string = f.read()
            f.close()
            string += '\nWindow Size: %.1f - fold_idx: %d' % (float(window_size)/10000.0, fold_idx)
        # report file was previously created
        else:
            string = 'Window Size: %.1f - fold_idx: %d' % (float(window_size)/10000.0, fold_idx)
        f = open(cwd + subPath + 'ML_Result/EPG Classification Progress Report.txt', 'w')
    # in case OS is windows
    else:
        # report file was not previously created
        if (os.path.isfile(cwd + subPath + 'ML_Result\\EPG Classification Progress Report.txt')):
            f = open(cwd + subPath + 'ML_Result\\EPG Classification Progress Report.txt', 'rb')
            string = f.read()
            f.close()
            string = '\nWindow Size: %.1f - fold_idx: %d' % (float(window_size)/10000.0, fold_idx)
        # report file was previously created
        else:
            string = 'Window Size: %.1f - fold_idx: %d' % (float(window_size)/10000.0, fold_idx)                
        f = open(cwd + subPath + 'ML_Result\\EPG Classification Progress Report.txt', 'w')
    f.write(string)
    f.close()





def score_single_output(dic, window_size, classifier_lab='NN', n_window=100000, pump_type='minor', all_idx = -1):
    """
    DESCRIPTION
        for the entire length of a given recording this method computes a confusion matrix for annotation of autoEPG.
    MODES
        0:  compare AutoEPG predicted peaks with their real region
        1:  compare clf preditions for AutoEPG chosen peaks with their real region
        2:  compare all clf predictions with the real regions
        3:  compare non baseline region (according to clf prediction) with their true region => accuracy
        4:  compare the precition of clf for regions that are not in baseline (according to the true annotation of regions) => precision
    """
    from sklearn.metrics import confusion_matrix
    test_array = pickle.load(open('%s%stest pred %s - ws_%.1f n_%d.cPickle' % (cwd, subPath, classifier_lab, window_size, n_window), 'rb'))
    test_pred = np.array([np.argmax(np.bincount(i.astype(int))) for i in test_array.T])
    test_size = test_pred.shape[0]
    test_idx = (np.array(pickle.load(open('%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_window), 'rb'))[:test_size])).astype(int)
    # if type(all_idx) == int:
    #     test_idx = np.array(sorted(pickle.load(open('%s%sML_all_idx_so_%d.cPickle' % (cwd, subPath, n_window), 'rb'))[:test_size]))
    # else:
    #     test_idx = np.array(sorted(all_idx[:test_size]))
    # return test_pred
    test_true = extract_recording_by_idx(all_idx = test_idx, dict1 = dic, window_size = int(window_size*10000), pump_type=pump_type)[:,-1]
    # return test_true
    if pump_type == 'minor':
        labs = np.array([1, 2, 3, 4, 5, 0])
        labs_figure = np.array(['e', 'E', 'P', 'R', 'r', 'I'])
    elif pump_type == 'major':
        labs = np.array([1, 2, 3, 0])
        labs_figure = np.array(['E', 'P', 'R', 'I'])
    cnf_matrix = confusion_matrix(y_true = test_true, y_pred = test_pred, labels = labs)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm=cnf_matrix, classes=labs_figure, classifier_lab=classifier_lab, title='%s vs True, %d windows of %d features, ' % (classifier_lab, n_window, int(window_size*10000)))
    fig1 = plt.savefig('%s%sconfusion_matrix %s %s_vs_true ws_%.1f n_%d.png' % (cwd, subPath, pump_type, classifier_lab, window_size, n_window), bbox_inches="tight", bbox_extra_artists=[])
    print('FILE LOCATION: %s%sconfusion_matrix %s %s_vs_true ws_%.1f n_%d.png' % (cwd, subPath, pump_type, classifier_lab, window_size, n_window))
    plt.clf()
    del fig1



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          classifier_lab = ''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from matplotlib import gridspec
    nclass = len(classes)
    if cm[:nclass, :].sum() == 0:
        title = title + '\n FNR: $\infty$, TPR: $\infty$'
    else:
        title = title + '\n FNR: %.3f' % (float(cm[:nclass-1, nclass-1].sum())/float(cm[:nclass-1, :].sum()))
        title = title + ' TPR: %.3f' % (float(cm[:nclass-1, :nclass-1].sum())/float(cm[:nclass-1, :].sum()))
    # if cm[:, :4].sum() == 0:
    #     title = title + ', TPR: $\infty$'
    # else:
    #     title = title + ', TPR: %.3f' % (float(cm[5, :4].sum())/float(cm[:, :4].sum()))
    if (classifier_lab != ''):
        classifier_lab = classifier_lab + ' '
    # gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])
    plt.subplot()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.subplot()
    plt.colorbar()
    # if classes[0] == int:
    #     d = {0:"I", 1:"e", 2:"E", 3:"P", 4:"R", 5:"r"}
    #     for i in range(len(classes)):
    #         classes[i] = d[classes[i]]
    y_classes = ['%s\n|%s|: %d' % (classes[i], classes[i], int(cm[i, :].sum())) for i in range(len(classes))]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, y_classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    global cmx
    cmx = cm
    print(cm)

    thresh = cm.max() / 2.
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:.3f}".format(round(cm[i, j], 3)), clip_on=False,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], clip_on=False,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel(classifier_lab + 'Predicted label', fontsize=12)


if __name__ == '__main__':
    #directories
        ## CTF desktops
    if os.path.isdir('C:\\Users\palikh\Downloads\Python_Scripts\\'):
        cwd = 'C:\\Users\palikh\Downloads\Python_Scripts\\'
        ## laptop Spectre
    elif os.path.isdir('C:\Users\Pouria\Documents\Python Scripts\JD\\'):
        cwd = 'C:\Users\Pouria\Documents\Python Scripts\JD\\'
        ## laptop JW
    elif os.path.isdir('/Downloads/Python_Scripts/'):
        cwd = '/home/hoopoo685/Downloads/Python_Scripts/'
        ## mcgill servers
    elif os.path.isdir('/data/users/palikh/JoeDent/'):
        cwd = '/data/users/palikh/JoeDent/'

    subPath = findPath(k=3, dic = d0)


    A = [0.5]   
    for window_length in A:
        # d = tabulate_single_output(window_size=window_length, n_window = 1000, pump_type = 'major'); print "Tabulate End"
        # d = balanced_index(d0, n_samples = 4000)
        classifier_single_output(d, validation_ratio = 0.2, n_iter = 100, test_set_ratio = 0.1, clf =MLPClassifier , n_folds = 5, dic = d0, classifier_lab='NN'); print "Classifier End"
        score_single_output(dic=d0, window_size=window_length, classifier_lab='NN', n_window=1000, pump_type = 'major', all_idx = d); print "Score End"