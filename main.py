import matplotlib.pyplot as plt
import numpy as np
import math
import neurokit2 as nk
from scipy import signal
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import statistics

def dif_mas (mass):
    sig_diff_mass = []
    for i in range(len(mass)):
        sig_diff_mass.append(np.diff(mass[i], n=1))
    return sig_diff_mass

def ecg_filter(input_signal, order, cutoff, fs, type = 'lowpass' #or 'highpass'
               ):
    b, a = signal.butter(order,
                  cutoff,
                  fs=fs,
                  btype=type,
                  analog=False)
    filter = signal.filtfilt
    signal_filtered = filter(b, a, input_signal)
    return signal_filtered

def find_ecg_epochs (sig,fs):
    time_offset = 0.5#sec
    sample_offser = round(time_offset*fs)
    sig_filtered = ecg_filter(sig, 5, 1, fs, type='highpass')
    sig_filtered = ecg_filter(sig_filtered, 5, 25, fs, type='lowpass')
    sig_peaks = nk.ecg_findpeaks(sig_filtered, sampling_rate=fs)["ECG_R_Peaks"]
    qrs_epochs = []
    for sig_peak in sig_peaks:
        offset_plus = sig_peak + sample_offser
        offset_minus = sig_peak - sample_offser
        if offset_plus< len(sig) and offset_minus>=0:
            cycle = np.take(sig, list(range(offset_minus, offset_plus)))
            qrs_epochs.append(cycle)
    return qrs_epochs

def distance_func(x1,y1,x2,y2):
    """
    Distance between two curves in 2d space
    :param x1: input cardio-cycle (len [0:-1] to fit with y1)
    :param y1: input cardio-cycle differential
    :param x2: input averaged cardio-cycle (len [0:-1] to fit with y1)
    :param y2: input averaged cardio-cycle differential
    :return: Distance between two curves in 2d space
    """
    assert len(x1) == len(y1) == len(x2) == len(y2)
    L1 = sum(math.sqrt((x1[i] - x1[i - 1]) ** 2 + (y1[i] - y1[i - 1]) ** 2) for i in range(1, len(x1)))
    L2 = sum(math.sqrt((x2[i] - x2[i - 1]) ** 2 + (y2[i] - y2[i - 1]) ** 2) for i in range(1, len(x2)))
    return L1-L2

def qf1_2_calc (epochs, epochs_mean, epochs_mean_diff):
    distance1_mas = []
    for epoch in epochs:
        epoch_dif = dif_mas([epoch])[0]
        distance1 = distance_func(epoch[0:-1], epoch_dif, epochs_mean[0:-1], epochs_mean_diff)
        distance1_mas.append(distance1)

    quality_factor1 = np.mean(distance1_mas)
    quality_factor2 = np.std(distance1_mas)
    return  quality_factor1, quality_factor2

def qf_first (epochs, epochs_dif, epochs_mean, epochs_mean_dif):
    # Calculate the Euclidean distance
    distance_mass = []
    for epoch, epoch_dif in zip(epochs, epochs_dif):
        x1 = epoch[0:-1]
        y1 = epoch_dif
        x2 = epochs_mean[0:-1]
        y2 = epochs_mean_dif
        distance = 0
        for i in range(len(x1)):
            distance += math.sqrt((x2[i] - x1[i]) ** 2 + (y2[i] - y1[i]) ** 2)
        distance_mass.append(distance)
    return np.mean(distance_mass), np.std(distance_mass)
def entropy_incell_count(data_frequencies, max_freq):
    """
    There about entropy calculation: https://habr.com/ru/articles/526460/
    :param data_frequencies: number of points in cells after grid splitting signal
    :param max_freq: the maximum possible numper of points in cell (input sigpart len)
    :return:entropy value
    """
    probabilities_mas = []
    probabilities_log = []
    for freq in data_frequencies:
        prob = freq/max_freq
        probabilities_mas.append(prob)
        if prob == 0: # to avoid log(0) with exeption:
            prob = 0.0000000000000000000001
        probabilities_log.append(np.log2(prob))
    total_entropy_minuse = 0
    for prob, log in zip(probabilities_mas, probabilities_log):
        total_entropy_minuse = total_entropy_minuse+(prob*log)
    return total_entropy_minuse * (-1)
def grid_splitter(epochs, epochs_diff, cels_num_param = 8, cutt_epoch = True):
    """
    Function to split 2d space with signals by grid
    :param epochs: Input QRS complexes
    :param epochs_diff: Input differentials of QRS complexes
    :param cels_num_param: The size of the grid parameter, if cels_num_param = 8, we get 8x8 grid
    :param cutt_epoch: if epochs and epochs_diff has different sizes, set True
    :return: The input 2d data devided by grid
    """
    cycle_params = {
        'x_max' : None,
        'y_max' : None,
        'x_min' : None,
        'y_min' : None
    }
    for epoch, epoch_dif in zip(epochs, epochs_diff):
        if cutt_epoch:
            epoch = epoch[0:-1]
        x_min = np.min(epoch)
        y_min = np.min(epoch_dif)
        x_max = np.max(epoch)
        y_max = np.max(epoch_dif)
        if cycle_params['x_max'] == None or cycle_params['x_max'] < x_max: cycle_params['x_max'] = x_max
        if cycle_params['x_min'] == None or cycle_params['x_min'] < x_min : cycle_params['x_min'] = x_min
        if cycle_params['y_max'] == None or cycle_params['y_max'] < y_max : cycle_params['y_max'] = y_max
        if cycle_params['y_min'] == None or cycle_params['y_min'] < y_min : cycle_params['y_min'] = y_min

    x_step = (cycle_params['x_max'] - cycle_params['x_min'])/cels_num_param
    y_step = (cycle_params['y_max'] - cycle_params['y_min'])/cels_num_param

    start_x_cell = cycle_params['x_min']
    start_y_cell = cycle_params['y_min']
    stop_y_cell= start_y_cell+y_step
    grid = []
    num_x_points = cels_num_param
    num_y_points = cels_num_param
    for i in range(num_x_points):
        x = start_x_cell + i * x_step
        for j in range(num_y_points):
            y = stop_y_cell + j * y_step
            grid.append({'cell_index':(i,j), 'cell_pos':(x, y)})
    grid_data = {'cell':[],'data':[]}
    for grid_cell_start in grid:
        cel_x_startpos = grid_cell_start['cell_pos'][0]
        cel_y_startpos = grid_cell_start['cell_pos'][1]
        cell_x_endpos = cel_x_startpos+x_step
        cell_y_endpos = cel_y_startpos+y_step
        cell_params = {'cell':{},'x_mass':[],'y_mass':[],'dycle_points_in_cell':[], 'dycle_points_in_cell_nums':[]}
        for epoch, epoch_dif in zip(epochs, epochs_diff):
            epoch = epoch[0:-1]
            xy_points = np.array(list(zip(epoch, epoch_dif)))
            points_in_cell = []
            points_in_cell_counter = 0
            for xy_point in xy_points:
                if xy_point[0]>=cel_x_startpos and xy_point[0]<=cell_x_endpos and xy_point[1]>=cel_y_startpos and xy_point[1] <=cell_y_endpos :
                    points_in_cell.append(xy_point)
                    points_in_cell_counter = points_in_cell_counter+1
            cell_params['cell'] = {'cell_code':grid_cell_start['cell_index'], 'start_x':cel_x_startpos,'stop_x':cell_x_endpos,'start_y':cel_y_startpos, 'stop_y':cell_y_endpos}
            cell_params['x_mass'].append(epoch)
            cell_params['y_mass'].append(epochs_diff)
            cell_params['dycle_points_in_cell'].append(points_in_cell)
            cell_params['dycle_points_in_cell_nums'].append(points_in_cell_counter)
        grid_data['cell'].append(grid_cell_start['cell_index'])
        grid_data['data'].append(cell_params)
    return grid_data

def entropy_param(epochs, epochs_dif, only_ears = False):
    """
    General function for calculating entropy between 2d plots of cardiocycles and differential of cardiocycles
    :param epochs: Input massive of cardiocycles
    :param epochs_dif: Input massive of differetial of cardiocycles
    :param only_ears: If we want to calculate the entropy using the phasic plots (x - cardiocycle, y - cardiocycle diff) ears (looks like ears on graph)
    :return: some features values
    """
    if only_ears:
        new_epochs = []
        new_epochs_diff = []
        for epoch, epoch_diff in zip(epochs, epochs_dif):
            ear_cluster_x, ear_cluster_y, not_ear_x, not_ear_y = ears_by_clusters_create(epoch, epoch_diff)
            new_epochs.append(ear_cluster_x)
            new_epochs_diff.append(ear_cluster_y)
        epochs = new_epochs
        epochs_dif = new_epochs_diff
        grid_data = grid_splitter(epochs, epochs_dif, cutt_epoch=False)
    else:
        grid_data = grid_splitter(epochs, epochs_dif)
    cell_statistics = {'means':[], 'medians':[], 'stds':[],'entropys':[]}
    for cl_cd, cell_dataa in zip(list(grid_data.values())[0],list(grid_data.values())[1]):
        nums_in_cell_data = cell_dataa['dycle_points_in_cell_nums']
        incell_mean = np.mean(nums_in_cell_data)
        incell_median = np.median(nums_in_cell_data)
        incell_std = np.std(nums_in_cell_data)
        incell_entropy = entropy_incell_count(nums_in_cell_data, len(epochs[0])-1)
        cell_statistics['means'].append(incell_mean)
        cell_statistics['medians'].append(incell_median)
        cell_statistics['stds'].append(incell_std)
        cell_statistics['entropys'].append(incell_entropy)

    means_mean = np.mean(cell_statistics['means'])
    means_std = np.std(cell_statistics['means'])
    count_lopop_cell = 0
    for mean_dots_in_cell in cell_statistics['means']:
        if mean_dots_in_cell <(means_mean-means_std):
            count_lopop_cell = count_lopop_cell+1
    count_empty_cell = 0
    for mean_dots_in_cell in cell_statistics['means']:
        if mean_dots_in_cell == 0:
            count_empty_cell = count_empty_cell+1

    mean_std_by_cell = np.mean(cell_statistics['stds'])
    entropy_by_cells = np.mean(cell_statistics['entropys'])
    entropy_std_by_cells = np.std(cell_statistics['entropys'])

    return count_empty_cell, count_lopop_cell, mean_std_by_cell, entropy_by_cells, entropy_std_by_cells

def ears_by_clusters_create(epochs,epochs_dif):
    """
    This function uses claster analysis to devide centers and ears on phasic plots (x - cardiocycle, y - cardiocycle diff)
    :param epochs:  cardiocycle mass
    :param epochs_dif: cardiocycle diff mass
    :return: two 2d plots 1 - ear graph x, y, 2 - center graph x, y
    """
    # Join the coordinates into an array of points
    points = np.vstack((epochs[:-1], epochs_dif)).T
    # Set the number of clusters
    num_clusters = 2
    # Create a k-mean model
    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    # Train the model on points
    kmeans.fit(points)
    # Get cluster labels for each point
    labels = kmeans.predict(points)
    # Divide points into clusters
    clusters = [points[labels == i] for i in range(num_clusters)]
    cluster_pointsx = []
    cluster_pointsy = []
    for i, cluster in enumerate(clusters):
        cluster_pointsx.append(cluster[:, 0])
        cluster_pointsy.append(cluster[:, 1])
    cluster1_x = cluster_pointsx[0]
    cluster1_y = cluster_pointsy[0]
    cluster2_x = cluster_pointsx[1]
    cluster2_y = cluster_pointsy[1]
    ear_cluster_x = []
    ear_cluster_y = []
    not_ear_x =[]
    not_ear_y = []
    #find ear cluster by number of points
    if len(cluster2_y) < len(cluster1_y):
        ear_cluster_x = cluster2_x
        ear_cluster_y = cluster2_y
        not_ear_x = cluster1_x
        not_ear_y = cluster1_y
    elif len(cluster2_y) > len(cluster1_y):
        ear_cluster_x = cluster1_x
        ear_cluster_y = cluster1_y
        not_ear_x = cluster2_x
        not_ear_y = cluster2_y
    return ear_cluster_x, ear_cluster_y, not_ear_x, not_ear_y

def qf4_calc(epochs, epochs_diff):
    """
    This function uses inside area and len of ear of phasic graph (x - cardiocycle, y - cardiocycle diff) for calculate features
    :param epochs: cardiocycle mass
    :param epochs_diff: cardiocycle diff mass
    :return: area params values
    """
    ears_lens_mass = []
    inside_area_mass = []
    for epoch, epoch_diff in zip(epochs, epochs_diff):
        ear_cluster_x, ear_cluster_y, not_ear_x, not_ear_y = ears_by_clusters_create(epoch, epoch_diff)
        #len of curve
        ear_cluster_len = sum(math.sqrt((ear_cluster_x[i] - ear_cluster_x[i - 1]) ** 2 + (ear_cluster_y[i] - ear_cluster_y[i - 1]) ** 2) for i in range(1, len(ear_cluster_x)))
        ears_lens_mass.append(ear_cluster_len)
        ear_line = list(zip(ear_cluster_x,ear_cluster_y))
        ear_line.append(ear_line[0])
        #find area inside curve
        polygon = Polygon(ear_line)
        area = polygon.area
        inside_area_mass.append(area)
    mean_inside_area = np.mean(inside_area_mass)
    std_inside_area = np.std(inside_area_mass)
    mean_ear_len = np.mean(ears_lens_mass)
    std_ear_len = np.std(ears_lens_mass)
    return mean_inside_area, std_inside_area, mean_ear_len, std_ear_len

def calc_features(input_ecg_signal, fs):
    epochs = find_ecg_epochs(input_ecg_signal,fs)
    epochs_mean = np.mean(epochs,axis=0)
    epochs_mean_diff = dif_mas([epochs_mean])[0]
    epochs_dif = dif_mas(epochs)

    quality_factor1, quality_factor2 = qf1_2_calc(epochs, epochs_mean, epochs_mean_diff)
    qf_firstt1, qf_firstt2 = qf_first(epochs, epochs_dif, epochs_mean, epochs_mean_diff)
    mean_inside_area, std_inside_area, mean_ear_len, std_ear_len = qf4_calc(epochs, epochs_dif)
    count_empty_cell, count_lopop_cell, mean_std_by_cell, entropy_by_cells, entropy_std_by_cells = entropy_param(epochs,
                                                                                                                 epochs_dif,
                                                                                                                 only_ears=False)
    count_empty_cell2, count_lopop_cell2, mean_std_by_cell2, entropy_by_cells2, entropy_std_by_cells2 = entropy_param(
        epochs,
        epochs_dif,
        only_ears=True)

    quality_params = {
        'qf1': quality_factor1,
        'qf2': quality_factor2,
        'qf_firstt1': qf_firstt1,
        'qf4_std_ear_len': std_ear_len,
        'entropy_by_cells': entropy_by_cells,
        'entropy_std_by_cells': entropy_std_by_cells,
        'entropy_std_by_cells2': entropy_std_by_cells2
    }

    return quality_params


best_params = {
    'qf1':'min',
    'qf2':'min',
    'qf_firstt1':'min',
    'qf4_std_ear_len':'min',
    'entropy_by_cells':'max',
    'entropy_std_by_cells':'max',
    'entropy_std_by_cells2':'max',
}

def prepare_signal_part(signal_part, fs):
    # filtering
    sig_part_filtered = ecg_filter(signal_part,5,1,fs, type='highpass')
    # normalisation
    values = sig_part_filtered.reshape((len(sig_part_filtered), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    sig_part_filtered_scaled = scaler.fit_transform(values)
    sig_part_filtered_scaled = sig_part_filtered_scaled.reshape((1, len(sig_part_filtered)))[0]
    sig_part_filtered_scaled = sig_part_filtered_scaled - np.mean(sig_part_filtered_scaled)
    return sig_part_filtered_scaled


def signals_cut(input_signals: list[np.array],
               partlen: int  # sec
               , fs) -> list[np.array]:
    """
    This function cut massive of leads signals to the equal parts
    :param input_signals:
    :param partlen: len of cutted part in seconds
    :param fs:
    :return: array of signal parts and gruped by leads
    """
    split_samples = int(partlen * fs)
    num_splits = len(input_signals[0]) // split_samples
    signals_splits = []
    for input_signal in input_signals:
        splits = np.split(input_signal[:num_splits * split_samples], num_splits)
        signals_splits.append(splits)
    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.plot(signals_splits[0][0])
    # plt.subplot(3, 1, 2)
    # plt.plot(signals_splits[1][0])
    # plt.subplot(3, 1, 3)
    # plt.plot(signals_splits[2][0])
    # plt.show()
    return signals_splits

def find_best_lead(input_signals:list[np.array],
                   signals_names:list[str],
                   partlen:int #sec
                   ,fs) -> str:
    '''
    The starter function of best lead selection.
    This function cut signals on parts, calculate features for each part and each lead,
    after that detect the best lead by voting between features
    proposed partlen 10-30 secs
    :param input_signals: the array if leads of input signal
    :param signals_names: names of input leads
    :param partlen: The len of divided part
    :param fs:
    :return: the best lead name
    '''
    signals_parts = signals_cut(input_signals, partlen, fs)
    signals_parts_features = {}
    counter = 0 #optional
    for signal_parts, signal_name in zip(signals_parts, signals_names):
        signal_parts_features = []
        for signal_part in signal_parts:
            counter += 1#optional
            print(str(counter) + '/' + str(len(signal_parts)*len(signals_parts)))#optional
            signal_part_prepared = prepare_signal_part(signal_part, fs)
            features = calc_features(signal_part_prepared, fs)
            signal_parts_features.append(features)
        signals_parts_features[signal_name] = signal_parts_features

    parts_bestlead_mass = []
    for partnum in range(len(signals_parts[0])):
        best_lead_by_params = []
        for param_name,param_decision_target in zip(list(best_params.keys()), list(best_params.values())):
            leads_features_by_part = {}
            for lead_name in list(signals_parts_features.keys()):
                features_line = signals_parts_features[lead_name][partnum]
                leads_features_by_part[lead_name] = features_line
            best_lead_by_param = None
            best_lead_param_value = None
            for lead in list(leads_features_by_part.keys()):
                param_value = leads_features_by_part[lead][param_name]
                if best_lead_by_param is None:
                    best_lead_by_param = lead
                if best_lead_param_value is None:
                    best_lead_param_value = param_value
                if param_decision_target == 'min':
                    if param_value < best_lead_param_value:
                        best_lead_by_param = lead
                        best_lead_param_value = param_value
                if param_decision_target == 'max':
                    if param_value > best_lead_param_value:
                        best_lead_by_param = lead
                        best_lead_param_value = param_value
            best_lead_by_params.append(best_lead_by_param)
        part_bestlead = statistics.mode(best_lead_by_params)
        parts_bestlead_mass.append(part_bestlead)
    signal_bestlead = statistics.mode(parts_bestlead_mass)
    return signal_bestlead


