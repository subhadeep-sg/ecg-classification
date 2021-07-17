import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

# Libraries for de-noising ecg signal
from scipy.signal import butter, iirnotch, lfilter
import scipy.signal as signal
import pywt

# Libraries for peak detection
from scipy.ndimage import label

# Libraries for organizing the feature extracted input
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.preprocessing import MultiLabelBinarizer
import itertools

# Libraries for model training
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, plot_confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import warnings

warnings.filterwarnings("ignore")

# Constants
fs = 1000
cutoff_high = 0.1  # 0.5
cutoff_low = 50  # 2
powerline = 60

columns = []
for i in range(54):
    columns.append('Column ' + str(i))

df = pd.DataFrame(columns=['filename',
                           'age',
                           'gender',
                           'diagnosis',
                           'ecg_signal',
                           'Lead I',
                           'Lead II',
                           'Lead III',
                           'aVR',
                           'aVL',
                           'aVF',
                           'V1',
                           'V2',
                           'V3',
                           'V4',
                           'V5',
                           'V6'])


def get_df():
    return df


def get_extract_df():
    return pd.DataFrame(columns=columns)


def butter_highpass(cutoff, fs_highpass, order_highpass):
    nyq = 0.5 * fs_highpass
    normal_cutoff = cutoff / nyq
    b, a = butter(order_highpass, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a


def butter_lowpass(cutoff, fs_lowpass, order_lowpass=5):
    nyq = 0.5 * fs_lowpass
    normal_cutoff = cutoff / nyq
    b, a = butter(order_lowpass, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def notch_filter(cutoff, q):
    nyq = 0.5 * fs
    freq = cutoff / nyq
    b, a = iirnotch(freq, q)
    return b, a


def hamilton_detector(ecg_signal, fs_hamilt):
    """
    P.S. Hamilton,
    Open Source ECG Analysis Software Documentation, E.P.Limited, 2002.
    """
    f1 = 8 / fs_hamilt
    f2 = 16 / fs_hamilt

    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    diff = abs(np.diff(ecg_signal))

    b = np.ones(int(0.08 * fs_hamilt))
    b = b / int(0.08 * fs_hamilt)
    a = [1]

    ma = signal.lfilter(b, a, diff)

    ma[0:len(b) * 2] = 0

    n_pks = []
    n_pks_ave = 0.0
    s_pks = []
    s_pks_ave = 0.0
    qrs = [0]
    rr = []
    rr_ave = 0.0

    th = 0.0

    i = 0
    idx = []
    peaks = []

    for i in range(len(ma)):

        if 0 < i < len(ma) - 1:
            if (ma[i - 1] < ma[i]) and (ma[i + 1] < ma[i]):
                peak = i
                peaks.append(i)

                if ma[peak] > th and (peak - qrs[-1]) > 0.3 * fs_hamilt:
                    qrs.append(peak)
                    idx.append(i)
                    s_pks.append(ma[peak])
                    if len(n_pks) > 8:
                        s_pks.pop(0)
                    s_pks_ave = np.mean(s_pks)

                    if rr_ave != 0.0:
                        if qrs[-1] - qrs[-2] > 1.5 * rr_ave:
                            missed_peaks = peaks[idx[-2] + 1:idx[-1]]
                            for missed_peak in missed_peaks:
                                if missed_peak - peaks[idx[-2]] > int(0.360 * fs_hamilt) and ma[missed_peak] > 0.5 * th:
                                    qrs.append(missed_peak)
                                    qrs.sort()
                                    break

                    if len(qrs) > 2:
                        rr.append(qrs[-1] - qrs[-2])
                        if len(rr) > 8:
                            rr.pop(0)
                        rr_ave = int(np.mean(rr))

                else:
                    n_pks.append(ma[peak])
                    if len(n_pks) > 8:
                        n_pks.pop(0)
                    n_pks_ave = np.mean(n_pks)

                th = n_pks_ave + 0.45 * (s_pks_ave - n_pks_ave)

                i += 1

    qrs.pop(0)

    return qrs, rr


class Denoiser:
    def __init__(self, order):
        self.fs = fs
        self.cutoff_high = cutoff_high
        self.cutoff_low = cutoff_low
        self.powerline = powerline
        self.order = order

    def final_filter(self, data):
        b, a = butter_highpass(self.cutoff_high, self.fs, order_highpass=self.order)
        x = lfilter(b, a, data)
        d, c = butter_lowpass(self.cutoff_low, self.fs, order_lowpass=self.order)
        y = lfilter(d, c, x)
        f, e = notch_filter(self.powerline, 30)
        z = lfilter(f, e, y)
        return z


class Read:
    def __init__(self, hea, mat):
        self.mat = mat
        self.infile = open(hea).readlines()
        self.ecg = loadmat(mat)

    def extract_info(self):
        age = str.rstrip(self.infile[13][6:])
        gender = self.infile[14][6]
        diag = str.rstrip(self.infile[15][5:])
        return age, gender, diag

    def insert_info(self, xdf):
        deno = Denoiser(5)
        age, gender, diag = self.extract_info()
        new_row = {'filename': self.mat,
                   'age': age,
                   'gender': gender,
                   'diagnosis': diag.split(','),
                   'ecg_signal': deno.final_filter(self.ecg['val']),
                   'Lead I': deno.final_filter(self.ecg['val'][0]),
                   'Lead II': deno.final_filter(self.ecg['val'][1]),
                   'Lead III': deno.final_filter(self.ecg['val'][2]),
                   'aVR': deno.final_filter(self.ecg['val'][3]),
                   'aVL': deno.final_filter(self.ecg['val'][4]),
                   'aVF': deno.final_filter(self.ecg['val'][5]),
                   'V1': deno.final_filter(self.ecg['val'][6]),
                   'V2': deno.final_filter(self.ecg['val'][7]),
                   'V3': deno.final_filter(self.ecg['val'][8]),
                   'V4': deno.final_filter(self.ecg['val'][9]),
                   'V5': deno.final_filter(self.ecg['val'][10]),
                   'V6': deno.final_filter(self.ecg['val'][11]),
                   }
        xdf = xdf.append(new_row, ignore_index=True)
        return xdf


class Encoder:
    def __init__(self, xdf, mappings_scored):
        self.df = xdf
        self.mappings_scored = mappings_scored
        self.d = dict()
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.mappings_scored['SNOMED CT Code'] = self.mappings_scored['SNOMED CT Code'].astype(object)

    def one_hot_encode(self):
        one_hot = MultiLabelBinarizer()
        y = one_hot.fit_transform(self.df['diagnosis'])  # .str.split(','))
        print("The classes we will look at are encoded as SNOMED CT codes:")
        print(one_hot.classes_)

        y = np.delete(y, -1, axis=1)
        print("classes: {}".format(y.shape[1]))
        return y, one_hot.classes_[0:-1]

    def snomed_codes(self):
        for j in range(27):
            self.d[self.mappings_scored.iloc[j]["SNOMED CT Code"]] = self.mappings_scored.iloc[j]["Dx"]
        keys_values = self.d.items()
        self.d = {str(key): str(value) for key, value in keys_values}

    def organize(self):
        self.snomed_codes()
        _, classes = self.one_hot_encode()
        new_cl = classes

        for c in new_cl:
            if c not in self.d.keys():
                new_cl = new_cl[new_cl != c]
        new_cl = list(new_cl)
        self.df['Updated_Diagnosis'] = self.df.diagnosis

        # Iterating through the dataframe row by row to edit each cell of diagnosis
        for index, row in self.df.iterrows():
            new_list = []
            for element in (self.df['diagnosis'].loc[index]):
                if element in new_cl:
                    new_list.append(element)

            if len(new_list) == 0:
                self.df['Updated_Diagnosis'].loc[index] = 0
            else:
                self.df['Updated_Diagnosis'].loc[index] = new_list

        self.df = self.df.loc[self.df.Updated_Diagnosis != 0]
        self.df = self.df.drop(columns='diagnosis')

        self.df = self.df.join(
            pd.DataFrame.sparse.from_spmatrix(
                self.mlb.fit_transform(self.df.pop('Updated_Diagnosis')),
                index=self.df.index,
                columns=self.mlb.classes_))

        return self.df


def table(indexes, avg, local, pre, post, row):
    parameters = []
    if len(indexes) == 1:
        for j in range(50):
            parameters.append(0)
    else:
        for j in range(50):
            parameters.append(row[indexes[j]])
    parameters.append(avg)
    parameters.append(local)
    parameters.append(pre)
    parameters.append(post)

    extract_df = pd.DataFrame(columns=columns)
    data_to_append = {}
    for j in range(54):
        data_to_append[extract_df.columns[j]] = parameters[j]
    return data_to_append


def get_plot_ranges(start=10, end=20, n=5):
    """
    Make an iterator that divides into n or n+1 ranges.
    - if end-start is divisible by steps, return n ranges
    - if end-start is not divisible by steps, return n+1 ranges, where the last range is smaller and ends at n
    """
    distance = end - start
    for i in np.arange(start, end, np.floor(distance / n)):
        yield int(i), int(np.minimum(end, np.floor(distance / n) + i))


def detect_peaks(ecg_signal, threshold=0.3, qrs_filter=None):
    """
    Peak detection algorithm using cross corrrelation and threshold
    """
    if qrs_filter is None:
        # create default qrs filter, which is just a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    peaks = []
    for i in range(len(similarity)):
        if similarity[i] > threshold:
            peaks.append(i)

    # return peaks (values in ms) using threshold
    return peaks, similarity


def group_peaks(p, threshold=50):
    """
    The peak detection algorithm finds multiple peaks for each QRS complex.
    Here we group collections of peaks that are very near (within threshold) and we take the median index
    """
    # initialize output
    output = np.empty(0)

    # label groups of sample that belong to the same peak
    peak_groups, num_groups = label(np.diff(p) < threshold)

    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output


class Extract:

    def __init__(self):
        self.sampfrom = 0
        self.sampto = 4900
        self.nr_plots = 1
        self.index_ecg = np.arange(self.sampto)
        self.arbit_index = 3

    def feature_extraction(self, row):

        for start, stop in get_plot_ranges(self.sampfrom, self.sampto, self.nr_plots):
            cond_slice = (self.index_ecg >= start) & (self.index_ecg < stop)
            ecg_slice = row[cond_slice]

        #ecg_slice = Denoiser(3).final_filter(ecg_slice)  # Originally order = 3 for prev result

        # Getting peaks and the rr interval

        peaks, similarity = detect_peaks(ecg_slice)
        peaks = np.array(peaks)
        grouped_peaks = group_peaks(peaks)
        rr = np.diff(grouped_peaks)
        rr_corrected = rr.copy()
        rr_corrected[np.abs(zscore(rr)) > 2] = np.median(rr)

        # Segmentation
        segment_peaks = grouped_peaks

        if len(grouped_peaks) < 10:
            return [0], 0, 0, 0, 0

        window = (grouped_peaks[1] - grouped_peaks[0]) / 2
        segment_peaks = (segment_peaks + window).astype(int)

        segment_list = []
        for i in range(len(segment_peaks) - 1):
            segment_list.append(ecg_slice[segment_peaks[i]:segment_peaks[i + 1]])

        # int(len(segment_peaks)/2)

        sample_indexes = np.arange(0, segment_list[self.arbit_index].shape[0],
                                   (segment_list[self.arbit_index].shape[0] / 50)).round().astype(int)
        sample_indexes += segment_peaks[self.arbit_index]
        avg_rr_interval = np.average(rr_corrected)
        pre_rr_interval = segment_list[self.arbit_index - 1].shape[0]
        post_rr_interval = segment_list[self.arbit_index + 1].shape[0]
        local_rr = segment_list[self.arbit_index].shape[0]

        return sample_indexes, avg_rr_interval, pre_rr_interval, post_rr_interval, local_rr


def plot_roc(y_test, pred):
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, pred, pos_label=1)
    # fpr2, tpr2, thresh2 = roc_curve(y_test, pred, pos_label=1)

    # roc curve for tpr = fpr
    random_prob = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_prob, pos_label=1)

    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Balanced Bagging Classifier')

    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()


class MLmodel:
    def __init__(self, lead_name, data, model):
        self.lead_name = lead_name
        self.data = data
        self.model = model

    def obtain_model_input(self):
        lead_df = pd.DataFrame(columns=columns)

        for i in range(len(self.data[self.lead_name])):
            # Getting a fixed portion of the signal for all records
            row = np.array(self.data[self.lead_name].iloc[i])[0:4900]

            indexes, avg, local, pre, post = Extract().feature_extraction(row)

            appending = table(indexes, avg, local, pre, post, row)

            lead_df = lead_df.append(appending, ignore_index=True)

        lead_df.index = self.data.index
        lead_df['bradycardia'] = self.data['426627000']
        lead_df['Age'] = self.data['age'].astype(int)
        lead_df['Gender'] = self.data['gender']

        gender = {'M': 1, 'F': 0}
        lead_df.Gender = [gender[item] for item in lead_df.Gender]

        lead_df = lead_df[lead_df['Column 2'] != 0]

        # rus = RandomUnderSampler(random_state=42)
        # ros = RandomOverSampler(random_state=42)

        y = lead_df['bradycardia']
        X = lead_df.drop(columns=['bradycardia'])

        #X, y = ros.fit_resample(X, y)

        return X, y

    def test_model(self):
        X, y = self.obtain_model_input()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=0)

        # rus = RandomUnderSampler(random_state=42)
        # ros = RandomOverSampler(random_state=42)
        # X_train, y_train = ros.fit_resample(X_train, y_train)

        model_lead = self.model
        model_lead.fit(X_train, y_train)
        pred = model_lead.predict(X_test)
        print(classification_report(y_test, pred))
        print(balanced_accuracy_score(y_test, pred))
        plot_confusion_matrix(model_lead, X_test, y_test)
