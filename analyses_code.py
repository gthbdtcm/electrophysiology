######################################### Itinerary #########################################
# [0] Background functions
# [1] [1] Tables 1 + 2 (neuron counts + response properties)
# [2] Figure 2 (screening raster plot + histogram)
# [3] Figure 3 (clip raster plot + histogram)
# [4] Figure 4 (rumination neurons)
# [5] Figure 5 (clip-gorilla neuron during screening)
# [6] Figure 6 (clip-gorilla neuron firing similarity to gorilla neuron)
# [7] Figure 8 (spike-field coherence)
# [8] Supplementary table 1
# [9] Supplementary figure 2 (clip-gorilla neuron - all during-experiment screening)
# [10] Text analyses - gorilla neuron
# [11] Text analyses - rumination neurons
# [12] Text analyses - clip-gorilla neuron (contextual learning)
# [13] Create video + figure 8c (still from video)
# [External files] "requirements.txt", lfp files, 'aggregate_statistics_by_revision1.csv' (this is built by this file)
######################################### End of itinerary #########################################

######################################### [0] Background functions #########################################
def readCSV(csvFileName):
    import csv, sys
    csvdataRows = []
    with open(csvFileName) as csvfile:
        spamreader = csv.reader(csvfile)
        #for line in data:
        for row in spamreader:
            csvdataRows.append(row)
    ## Return rows #
    return csvdataRows

def writeToCSV(csvFileName, csvdataRows):
    import csv, sys
    if sys.version_info >= (3,0,0):
        csvfile = open(csvFileName, 'w', newline='')
    else:
        csvfile = open(csvFileName, 'wb')
    spamwriter = csv.writer(csvfile)
    for row in csvdataRows:
        try:
            spamwriter.writerow(row)
        except:
            print("Failed at " + str(row))
    return
"""
def build_timestamps_sixvideomap():
    from mat4py import loadmat as loadmat_mat4py
    mat_file_name = '434e53gr_20.02.2012_13h05/434e53gr_20.02.2012_13h05/data/index_Recognition5001.mat'
    data = loadmat_mat4py(mat_file_name)
    timestamps = [el[0] for el in data['pepNEV']['index']['sync']['timestamps']]
    timestamps_difference_real_seconds = [(timestamps[i+1]-timestamps[i])/30000 if i < (len(timestamps)-1) else 0 for i in range(len(timestamps))]
    timestamps_difference_real_seconds_rounded = [round(el,2) for el in timestamps_difference_real_seconds]
    video_event_index_sequences = [[2749, 2750], [2771, 2772], [3251, 3252], [3273, 3274], [3295, 3296]]
    six_video_map = {i+1:{'events':video_event_index_sequences[i], 'timestamp_range_seconds':[min([timestamps[j]/30000 for j in video_event_index_sequences[i]]), max([timestamps[j]/30000 for j in video_event_index_sequences[i]])]} for i in range(len(video_event_index_sequences))}
    return timestamps, six_video_map
"""

def build_timestamps_sixvideomap():
    import mat73
    mat_file_name = 'data/events.mat'
    data = mat73.loadmat(mat_file_name)
    timestamps = list(data['events'])
    timestamps_difference_real_seconds = [(timestamps[i+1]-timestamps[i])/30000 if i < (len(timestamps)-1) else 0 for i in range(len(timestamps))]
    timestamps_difference_real_seconds_rounded = [round(el,2) for el in timestamps_difference_real_seconds]
    #recurring_sequences_images = [[i,i+1,i+2,i+3] for i in range(len(timestamps_difference_real_seconds_rounded)) if (i >= 2790) and (i <= 3224) and (timestamps_difference_real_seconds_rounded[i] in [0.84, 0.85, 0.86]) and (timestamps_difference_real_seconds_rounded[i-1] not in [0.84, 0.85, 0.86])]#2917 is a mistake
    video_event_index_sequences = [[2749, 2750], [2771, 2772], [3251, 3252], [3273, 3274], [3295, 3296]]
    six_video_map = {i+1:{'events':video_event_index_sequences[i], 'timestamp_range_seconds':[min([timestamps[j]/30000 for j in video_event_index_sequences[i]]), max([timestamps[j]/30000 for j in video_event_index_sequences[i]])]} for i in range(len(video_event_index_sequences))}
    return timestamps, six_video_map

def read_mat_file(mat_file_name):
    from scipy.io import loadmat
    mat_file_name = 'data/spikes/'+mat_file_name
    mat_data = loadmat(mat_file_name)
    return mat_data

"""
def read_mat_file_revision1(mat_file_name, revision=None):
    #e.g., revision=1
    if revision==None:
        from scipy.io import loadmat
        mat_file_name = '434e53gr_20.02.2012_13h05/434e53gr_20.02.2012_13h05/data/'+mat_file_name
        mat_data = loadmat(mat_file_name)
        return mat_data
    else:
        from scipy.io import loadmat
        mat_file_name = '434e53gr_20.02.2012_13h05/434e53gr_20.02.2012_13h05/revision'+str(revision)+'-data/'+mat_file_name
        mat_data = loadmat(mat_file_name)
        return mat_data
"""

def read_mat_file_revision1(mat_file_name='timesNSX_1.mat', revision=None):
    import mat73
    mat_file_name = 'data/spikes/'+mat_file_name
    mat_data = mat73.loadmat(mat_file_name)
    #cluster_class = data['cluster_class']
    #neurons = list(set([el[0] for el in cluster_class]))
    #neuron_spikes = {neuron:[el[1]/1000 for el in cluster_class if el[0]==neuron] for neuron in neurons}
    return mat_data

def map_electrode_to_videos(mat_file_name, six_video_map, revision=None):
    # [1] Read file
    mata_data = read_mat_file_revision1(mat_file_name, revision)
    cluster_class = mata_data['cluster_class']
    neurons = list(set([el[0] for el in cluster_class]))
    neuron_spikes = {neuron:[el[1]/1000 for el in cluster_class if el[0]==neuron] for neuron in neurons}
    neuron_video_trials = {i:{j:[] for j in list(six_video_map.keys())} for i in neurons}
    # For each neuron ("cluster")
    for neuron in list(neuron_video_trials.keys()):
        # For each video_trial
        for trial in list(neuron_video_trials[neuron].keys()):
            trial_start_time = six_video_map[trial]['timestamp_range_seconds'][0]
            trial_end_time = six_video_map[trial]['timestamp_range_seconds'][1]
            for neuron_spike in neuron_spikes[neuron]:
                # -3, +3 seconds (longer range around video interval)
                if (neuron_spike >= trial_start_time-3.0) and (neuron_spike <= trial_end_time+3.0):
                    neuron_spike_from_trial_start = round(neuron_spike - trial_start_time,2)
                    neuron_video_trials[neuron][trial].append(neuron_spike_from_trial_start)
    return neuron_video_trials

def build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision=None):
    from statistics import mean, stdev, median
    import math
    mata_data = read_mat_file_revision1(mat_file_name, revision)
    cluster_class = mata_data['cluster_class']
    neurons = list(set([el[0] for el in cluster_class]))
    data_start_time = timestamps[0]/30000#this could just be 0?
    data_end_time = timestamps[len(timestamps)-1]/30000
    neuron_spikes = {neuron:[el[1]/1000 for el in cluster_class if el[0]==neuron] for neuron in neurons}
    neuron_spikes_cluster = neuron_spikes[neuron_cluster_number]
    neuron_spikes_cluster_within_start_end = [el for el in neuron_spikes_cluster if (el>=data_start_time) and (el<=data_end_time)]
    # Do we need to put in bins, or should it be the same if we sum/divide by bin count? average of averages? Let's create bins to be safe...
    bin_length = 1
    data_start_time_full_second = math.floor(data_start_time)
    data_end_time_full_second = math.ceil(data_end_time)
    bins = [[i, i+1] for i in list(range(data_start_time_full_second,data_end_time_full_second))]
    # bins = [[12, 13], [13, 14], [14, 15], [15, 16], ..., [3204, 3205]]
    histogram_index = {i:[neuron_spike for neuron_spike in neuron_spikes_cluster_within_start_end if (neuron_spike >= bins[i][0]) and (neuron_spike < bins[i][1])] for i in range(len(bins))}
    histogram = [len(el) for el in list(histogram_index.values())]
    average = mean(histogram)
    med = median(histogram)
    average_plus_two_std = average + 2*stdev(histogram)
    average_rounded = round(average,2)
    average_plus_two_std_rounded = round(average_plus_two_std,2)
    standard_deviation = stdev(histogram)
    return average_rounded, average_plus_two_std_rounded, standard_deviation

def build_average_firing_rate_dictionary(statistically_significant_neurons_within_1s_onset, six_video_map, revision=None):
    average_firing_rate_dictionary = {channel_neuron:{} for channel_neuron in statistically_significant_neurons_within_1s_onset}
    for channel_neuron in statistically_significant_neurons_within_1s_onset:
        print(str(statistically_significant_neurons_within_1s_onset.index(channel_neuron)) + ' of ' + str(len(statistically_significant_neurons_within_1s_onset)))
        cluster = int(channel_neuron[len('cluster_'):channel_neuron.index('_neuron_')])
        neuron_cluster_number = int(channel_neuron[channel_neuron.index('_neuron_')+len('_neuron_'):])
        mat_file_name = 'timesNSX_'+str(cluster)+'.mat'
        mat_file_name = convert_mat_filename_to_revision(mat_file_name, revision)
        neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
        neural_data = list(neuron_video_trials[neuron_cluster_number].values())
        neural_data_histogram = [item for sublist in neural_data for item in sublist]
        average_rounded, average_plus_two_std_rounded, standard_deviation = build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision)
        average_firing_rate_dictionary[channel_neuron]['average_firing_rate'] = average_rounded
        average_firing_rate_dictionary[channel_neuron]['standard_deviation'] = standard_deviation
    return average_firing_rate_dictionary

def build_full_average_firing_rate_dictionary(revision=None):
    all_channel_neurons = build_all_channel_neurons(revision)
    average_firing_rate_dictionary = build_average_firing_rate_dictionary(all_channel_neurons, six_video_map, revision)
    return average_firing_rate_dictionary

def build_video_histogram(mat_file_name, neuron_cluster_number, six_video_map, timestamps, revision=None):
    from statistics import mean, median
    #from scipy.signal import find_peaks
    # [0] Build raw data - waveform and average, stdev
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    neural_data = list(neuron_video_trials[neuron_cluster_number].values())
    neural_data_histogram = [item for sublist in neural_data for item in sublist]
    # [1] Build waveform (1s interval counts, same start_end as earlier plots)
    data_start_time = 0
    data_end_time = 36
    bins = [[i, i+1] for i in list(range(data_start_time,data_end_time))]
    histogram_index = {i:[neuron_spike for neuron_spike in neural_data_histogram if (neuron_spike >= bins[i][0]) and (neuron_spike < bins[i][1])] for i in range(len(bins))}
    histogram_waveform = [len(el) for el in list(histogram_index.values())]
    #[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 2, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0]
    video_histogram = histogram_waveform
    return video_histogram

def build_high_level_statistics_by_region(six_video_map, timestamps, average_firing_rate_dictionary, revision=None):
    from statistics import mean, stdev
    import json
    # [0] Read in data
    #csv_file_name = 'manual_statistics.csv'
    #data = readCSV(csv_file_name)
    #data[0][0] = 'cluster'
    all_channel_neurons = build_all_channel_neurons(revision)
    # [1] Build regions
    #regions = list(set([row[data[0].index('area of the brain')] for row in data[1:]]))
    regions = ['Right Orbitfronal', 'Right Entorhinal Cortex', 'Right Medial Hippocampus', 'Right Anterior F...', 'Left Amygdala', 'Left Orbitfronal', 'Left Entorhinal Cortex', 'Left Anterior F...', 'Left Anterior Hippocampus', 'Left Anterior Cingulate', 'Right Amygdala', 'Right Anterior Cingulate']
    # [2] For each neuron in region, determine if responds to any of 3-4 epochs (beginning, gorilla, end)
    epochs = {'beginning':[0,2], 'gorilla':[19,29], 'end':[33,36]}
    all_channel_neurons_dictionary = {channel_neuron:{'region':'tbd', 'epochs':{epoch:{'response':0} for epoch in list(epochs.keys())}} for channel_neuron in all_channel_neurons}
    cluster_map = build_cluster_map(all_channel_neurons)
    for channel_neuron in all_channel_neurons:
        print(str(all_channel_neurons.index(channel_neuron)) + ' of ' + str(len(all_channel_neurons)))
        # Region
        #all_channel_neurons_dictionary[channel_neuron]['region'] = data[1:][all_channel_neurons.index(channel_neuron)][data[0].index('area of the brain')]
        all_channel_neurons_dictionary[channel_neuron]['region'] = cluster_map[channel_neuron]
        #[[all_channel_neurons_dictionary[channel_neuron]['region'] for channel_neuron in list(all_channel_neurons_dictionary.keys())].count(region) for region in regions]
        cluster = int(channel_neuron[len('cluster_'):channel_neuron.index('_neuron_')])
        neuron_cluster_number = int(channel_neuron[channel_neuron.index('_neuron_')+len('_neuron_'):])
        mat_file_name = 'timesNSX_'+str(cluster)+'.mat'
        mat_file_name = convert_mat_filename_to_revision(mat_file_name, revision)
        video_histogram = build_video_histogram(mat_file_name, neuron_cluster_number, six_video_map, timestamps, revision)
        # Events
        for epoch in list(epochs.keys()):
            stimuli_interval = epochs[epoch]
            if any([(el > average_firing_rate_dictionary[channel_neuron]['average_firing_rate']+3*average_firing_rate_dictionary[channel_neuron]['standard_deviation']) for el in video_histogram[stimuli_interval[0]:stimuli_interval[1]]]):
                # Remove neurons where >= 10 bins >= (mean + 3std) [not responding to epoch but rather entire video]
                if ((len([el for el in video_histogram if (el > average_firing_rate_dictionary[channel_neuron]['average_firing_rate']+3*average_firing_rate_dictionary[channel_neuron]['standard_deviation'])]) < 10) or (epoch=='gorilla' and channel_neuron in ['cluster_22_neuron_2', 'cluster_17_neuron_1'])) and not (epoch=='gorilla' and channel_neuron not in ['cluster_22_neuron_2', 'cluster_17_neuron_1']):
                    all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch]['response'] = 1
                    # [3] For each epoch responded to, build reference statistics (average latency, duration and density)
                    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
                    neural_data = list(neuron_video_trials[neuron_cluster_number].values())
                    neural_data_histogram = [item for sublist in neural_data for item in sublist]
                    #### [i] Onset (relevance: any within 1000ms=1s of stimuli_interval start)
                    relevant_trials_latency = [[el for el in trial if (el >= stimuli_interval[0]) and (el <= stimuli_interval[1])] for trial in neural_data if (len(trial) > 0) and any([(el-stimuli_interval[0] < 1) and (el-stimuli_interval[0] >= 0) for el in trial])]
                    if len(relevant_trials_latency) > 0:
                        all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch]['average_latency'] = mean([el[0] for el in relevant_trials_latency])
                    #### [ii] Duration (relevance: any within 1000ms=1s of stimuli_interval start that have at least 2 spikes)
                    relevant_trials_duration_density = [trial for trial in [[el for el in trial if (el >= stimuli_interval[0]) and (el <= stimuli_interval[1])] for trial in neural_data if (len(trial) > 0) and any([(el-stimuli_interval[0] < 1) and (el-stimuli_interval[0] >= 0) for el in trial])] if len(trial) > 1]
                    if len(relevant_trials_duration_density) > 0:
                        all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch]['average_duration'] = mean([trial[len(trial)-1]-trial[0] for trial in relevant_trials_duration_density])
                        #### [iii] Density (relevance: any within 1000ms=1s of stimuli_interval start that have at least 2 spikes)
                        bin_length_in_seconds = 1
                        all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch]['average_density'] = mean([len(trial)/(stimuli_interval[1]-stimuli_interval[0]) for trial in relevant_trials_duration_density])
    # [4] Aggregate individual statistics by region (neuron count + ...)
    aggregate_statistics = {region:{epoch:{'neuron_count':0, 'neurons':[], 'average_latency':'tbd', 'average_duration':'tbd', 'average_density':'tbd'} for epoch in list(epochs.keys())} for region in regions}
    for region in regions:
        for epoch in list(epochs.keys()):
            # All neurons that [a] are part of the region and [b] have a response to the epoch
            relevant_neurons = [channel_neuron for channel_neuron in (all_channel_neurons_dictionary.keys()) if (all_channel_neurons_dictionary[channel_neuron]['region']==region) and (all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch]['response']==1)]
            aggregate_statistics[region][epoch]['neuron_count'] = len(relevant_neurons)
            aggregate_statistics[region][epoch]['neurons'] = relevant_neurons
            descriptive_statistics = ['average_latency', 'average_density', 'average_duration']
            for descriptive_statistic in descriptive_statistics:
                aggregate_statistics[region][epoch][descriptive_statistic] = mean([all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch][descriptive_statistic] for channel_neuron in relevant_neurons if (descriptive_statistic in list(all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch].keys()))]) if len([all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch][descriptive_statistic] for channel_neuron in relevant_neurons if (descriptive_statistic in list(all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch].keys()))]) > 0 else "N/A"
                aggregate_statistics[region][epoch][descriptive_statistic+'_stdev'] = stdev([all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch][descriptive_statistic] for channel_neuron in relevant_neurons if (descriptive_statistic in list(all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch].keys()))]) if len([all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch][descriptive_statistic] for channel_neuron in relevant_neurons if (descriptive_statistic in list(all_channel_neurons_dictionary[channel_neuron]['epochs'][epoch].keys()))]) > 1 else "N/A"
                # sum([aggregate_statistics[region]['end']['neuron_count'] for region in aggregate_statistics.keys()])#62-->69
                # {region:aggregate_statistics[region]['end']['neuron_count'] for region in aggregate_statistics.keys()}#62-->69
    # [5] Save
    csv_file_name = 'aggregate_statistics_by_region5.csv' if revision==None else 'aggregate_statistics_by_revision1.csv'
    csv_data_rows = [['region', 'epoch', 'neuron_count', 'neurons', 'average_latency', 'average_latency_stdev', 'average_duration', 'average_duration_stdev', 'average_density', 'average_density_stdev']]
    for region in regions:
        for epoch in list(epochs.keys()):
            csv_data_rows.append([region, epoch, aggregate_statistics[region][epoch]['neuron_count'], json.dumps(aggregate_statistics[region][epoch]['neurons']), aggregate_statistics[region][epoch]['average_latency'], aggregate_statistics[region][epoch]['average_latency_stdev'], aggregate_statistics[region][epoch]['average_duration'], aggregate_statistics[region][epoch]['average_duration_stdev'], aggregate_statistics[region][epoch]['average_density'], aggregate_statistics[region][epoch]['average_density_stdev']])
    writeToCSV(csv_file_name, csv_data_rows)
    return aggregate_statistics

def map_electrode(mat_file_name, image_trials_map, revision=None):
    # [1] Read file
    mata_data = read_mat_file_revision1(mat_file_name, revision)
    cluster_class = mata_data['cluster_class']
    # [2] Split into neurons (non-zero only --> 0 == noise?)
    #neurons = list(set([el[0] for el in cluster_class if el[0] != 0.0]))
    neurons = list(set([el[0] for el in cluster_class]))
    neuron_spikes = {neuron:[el[1]/1000 for el in cluster_class if el[0]==neuron] for neuron in neurons}
    # [3] Map each neuron to time image_108: {image_index1:{'neuron1_spikes':[], 'neuron2_spikes':[]}, }
    #one_hundred_eight_images_map_with_neurons = {i:{'neurons':} for i in list(one_hundred_eight_images_map.keys())}
    # [4] Map all images together (neurons[0:2] -> images[1:9] -> trials[1:12] -> spikes[?])
    neuron_trials = {i:{j:{k:[] for k in range(len(image_trials_map[j]))} for j in list(image_trials_map.keys())} for i in neurons}
    # For each neuron
    for neuron in list(neuron_trials.keys()):
        # For each image
        for image in list(neuron_trials[neuron].keys()):
            # For each trial
            for trial in list(neuron_trials[neuron][image].keys()):
                # Add, remove 0.5 seconds for comparison in Raster plot?
                trial_start_time = image_trials_map[image][trial][0]
                trial_end_time = image_trials_map[image][trial][1]
                # Append all spike timestamps within period, from start time of trial?
                for neuron_spike in neuron_spikes[neuron]:
                    # If neuron spike is within interval
                    #if (neuron_spike >= trial_start_time) and (neuron_spike <= trial_end_time):
                    if (neuron_spike >= trial_start_time-0.5) and (neuron_spike <= trial_end_time):
                        # Append as difference between time and interval start time
                        neuron_spike_from_trial_start = round(neuron_spike - trial_start_time,2)
                        neuron_trials[neuron][image][trial].append(neuron_spike_from_trial_start)
    return neuron_trials

"""
def build_image_trials_map_and_image_map():
    from mat4py import loadmat as loadmat_mat4py
    mat_file_name = '434e53gr_20.02.2012_13h05/434e53gr_20.02.2012_13h05/data/index_Recognition5001.mat'
    data = loadmat_mat4py(mat_file_name)
    timestamps = [el[0] for el in data['pepNEV']['index']['sync']['timestamps']]
    timestamps_difference_real_seconds = [(timestamps[i+1]-timestamps[i])/30000 if i < (len(timestamps)-1) else 0 for i in range(len(timestamps))]
    timestamps_difference_real_seconds_rounded = [round(el,2) for el in timestamps_difference_real_seconds]
    recurring_sequences_images = [[i,i+1,i+2,i+3] for i in range(len(timestamps_difference_real_seconds_rounded)) if (i >= 2790) and (i <= 3224) and (timestamps_difference_real_seconds_rounded[i] in [0.84, 0.85, 0.86]) and (timestamps_difference_real_seconds_rounded[i-1] not in [0.84, 0.85, 0.86])]#2917 is a mistake
    from scipy.io import loadmat as loadmat_scipy
    mat_file_name = '434e53gr_20.02.2012_13h05/434e53gr_20.02.2012_13h05/psychophysics/Prepare.mat'
    prepare_mat_data = loadmat_scipy(mat_file_name)
    trial_order = [el[0] for el in prepare_mat_data['TrialOrder']]
    image_map = {i+1: prepare_mat_data['ImageNames'][0][i][0] for i in range(len(prepare_mat_data['ImageNames'][0]))}
    one_hundred_eight_images_map = {i:{'image_id':trial_order[i], 'image_name':image_map[trial_order[i]], 'events':recurring_sequences_images[i], 'timestamp_range_seconds':[min([timestamps[j]/30000 for j in recurring_sequences_images[i]]), max([timestamps[j]/30000 for j in recurring_sequences_images[i]])]} for i in range(len(trial_order))}
    image_trials = {i:[j for j, x in enumerate(trial_order) if x == i] for i in list(set(trial_order))}
    image_trials_map = {i:[one_hundred_eight_images_map[j]['timestamp_range_seconds'] for j in image_trials[i]] for i in list(image_trials.keys())}
    return image_map, image_trials_map
"""
def build_image_trials_map_and_image_map():
    import mat73
    mat_file_name = 'data/events.mat'
    data = mat73.loadmat(mat_file_name)
    timestamps = list(data['events'])
    timestamps_difference_real_seconds = [(timestamps[i+1]-timestamps[i])/30000 if i < (len(timestamps)-1) else 0 for i in range(len(timestamps))]
    timestamps_difference_real_seconds_rounded = [round(el,2) for el in timestamps_difference_real_seconds]
    recurring_sequences_images = [[i,i+1,i+2,i+3] for i in range(len(timestamps_difference_real_seconds_rounded)) if (i >= 2790) and (i <= 3224) and (timestamps_difference_real_seconds_rounded[i] in [0.84, 0.85, 0.86]) and (timestamps_difference_real_seconds_rounded[i-1] not in [0.84, 0.85, 0.86])]#2917 is a mistake
    mat_file_name = 'data/screening.mat'
    data = mat73.loadmat(mat_file_name)
    trial_order = data['order']
    image_map = {i+1: data['images']['name'][i] for i in range(len(data['images']['name']))}
    one_hundred_eight_images_map = {i:{'image_id':trial_order[i], 'image_name':image_map[trial_order[i]], 'events':recurring_sequences_images[i], 'timestamp_range_seconds':[min([timestamps[j]/30000 for j in recurring_sequences_images[i]]), max([timestamps[j]/30000 for j in recurring_sequences_images[i]])]} for i in range(len(trial_order))}
    image_trials = {i:[j for j, x in enumerate(trial_order) if x == i] for i in list(set(trial_order))}
    image_trials_map = {i:[one_hundred_eight_images_map[j]['timestamp_range_seconds'] for j in image_trials[i]] for i in list(image_trials.keys())}
    return image_map, image_trials_map

def build_reference_statistics_images(mat_file_name, neuron_cluster_number, images_of_interest, revision=None):
    #### mat_file_name, neuron_cluster_number, images_of_interest = 'timesNSX_22.mat', 2, {i:image_map[i] for i in [3,5,6,7]}####
    neuron_trials = map_electrode(mat_file_name, image_trials_map, revision)# {cluster:{image#1-9:{trial#0-11}}}
    from statistics import mean, median, stdev
    reference_statistics = {i:{'image':images_of_interest[i]} for i in list(images_of_interest.keys())}
    for image in list(images_of_interest.keys()):
        spikes_included_in_trials = {i:[el for el in neuron_trials[neuron_cluster_number][image][i] if (el >= 0) and (el < 1)] for i in list(neuron_trials[neuron_cluster_number][image].keys())}
        trial_statistics = {'average_latency':'', 'average_duration':'', 'average_density':''}
        # [1] Latency = start time
        trial_statistics['average_latency'] = mean([el[0] for el in list(spikes_included_in_trials.values()) if len(el) > 0])
        trial_statistics['latency_standard_deviation'] = stdev([el[0] for el in list(spikes_included_in_trials.values()) if len(el) > 0])
        trial_statistics['median_latency'] = median([el[0] for el in list(spikes_included_in_trials.values()) if len(el) > 0])
        # [2] Duration = start time - end time (if at least 2 data points exist)
        trial_statistics['average_duration'] = mean([el[len(el)-1]-el[0] for el in list(spikes_included_in_trials.values()) if len(el) > 1])
        trial_statistics['duration_standard_deviation'] = stdev([el[len(el)-1]-el[0] for el in list(spikes_included_in_trials.values()) if len(el) > 1])
        trial_statistics['median_duration'] = median([el[len(el)-1]-el[0] for el in list(spikes_included_in_trials.values()) if len(el) > 1])
        # [3] Density = Average number per 10ms period, within the duration
        trial_intervals_relevant_trials = {i: [spikes_included_in_trials[i][0], spikes_included_in_trials[i][len(spikes_included_in_trials[i])-1]] for i in list(spikes_included_in_trials.keys()) if len(spikes_included_in_trials[i]) > 1}
        bin_length_in_seconds = .1#10ms
        #trial_statistics['average_density'] = {i:len(spikes_included_in_trials[i])/((trial_intervals_relevant_trials[i][1]-trial_intervals_relevant_trials[i][0])/bin_length_in_seconds) for i in list(trial_intervals_relevant_trials.keys())}
        trial_statistics['average_density'] = mean([len(spikes_included_in_trials[i])/((trial_intervals_relevant_trials[i][1]-trial_intervals_relevant_trials[i][0])/bin_length_in_seconds) for i in list(trial_intervals_relevant_trials.keys())])
        trial_statistics['density_standard_deviation'] = stdev([len(spikes_included_in_trials[i])/((trial_intervals_relevant_trials[i][1]-trial_intervals_relevant_trials[i][0])/bin_length_in_seconds) for i in list(trial_intervals_relevant_trials.keys())])
        trial_statistics['median_density'] = median([len(spikes_included_in_trials[i])/((trial_intervals_relevant_trials[i][1]-trial_intervals_relevant_trials[i][0])/bin_length_in_seconds) for i in list(trial_intervals_relevant_trials.keys())])
        # Add to reference_statistics
        reference_statistics[image]['trial_statistics'] = trial_statistics
    return reference_statistics

def build_raster_plot_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision=None, file_name_to_save=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle(mat_file_name + ', cluster: ' + str(int(neuron_cluster_number)) + ', image: ' + image_map[image][:image_map[image].index('.')], fontsize=10, fontweight='bold')# electrode file name, neuron cluster
    neural_data = list(neuron_trials_specific_neuron[image].values())
    axs.eventplot(neural_data)
    axs.set_xlabel('Time(s)', fontsize=8)
    axs.set_xlim([-0.5,1])
    axs.tick_params(labelsize=6)
    axs.set_ylabel('Trial number', fontsize=8)
    #axs.set_ylim([0,len(neuron_trials_specific_neuron[image])])
    axs.set_ylim([-0.5,11.5])#clip-gorilla supplementary-materials
    file_name = 'tests/screening_raster_plot.svg' if (revision==None) else 'revision'+str(revision)+'/screening_raster_plot.svg'
    if file_name_to_save!=None:
        file_name=file_name_to_save
    fig.savefig(file_name, format='svg')
    plt.close()
    print('Saved ' + file_name)
    return

def build_histogram_as_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision=None, file_name_to_save=None):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(1, 1)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle(mat_file_name + ', cluster: ' + str(int(neuron_cluster_number)) + ', image: ' + image_map[image][:image_map[image].index('.')], fontsize=10, fontweight='bold')# electrode file name, neuron cluster
    neural_data = list(neuron_trials_specific_neuron[image].values())
    neural_data_histogram = [item for sublist in neural_data for item in sublist]
    axs.hist(neural_data_histogram, np.linspace(-0.5,1.0,int((1.0-(-0.5))/.1)+1))
    axs.set_xlim((-0.5,1.0))
    axs.set_xlabel('Time(s) [bin size = .1s]', fontsize=8)
    axs.set_ylim((0,12))# to set each on the same scale --> see the comparison more easily between gorilla response and nonresponse
    #axs.set_ylim((0,10))# clip gorilla supplementary-materials
    axs.set_ylabel('Count', fontsize=8)
    file_name = 'tests/screening_histogram.svg' if (revision==None) else 'revision'+str(revision)+'/screening_histogram.svg'
    if file_name_to_save!=None:
        file_name=file_name_to_save
    fig.savefig(file_name, format='svg')
    plt.close()
    print('Saved ' + file_name)
    return

def build_aggregate_neuron_trials(image_trials_map, revision=None):
    relevant_files = ['times_NSX87.mat', 'times_NSX50.mat', 'times_NSX79.mat', 'times_NSX45.mat', 'times_NSX51.mat', 'times_NSX47.mat', 'times_NSX85.mat', 'times_NSX42.mat', 'times_NSX57.mat', 'times_NSX80.mat', 'times_NSX69.mat', 'times_NSX41.mat', 'times_NSX68.mat', 'times_NSX27.mat', 'times_NSX26.mat', 'times_NSX7.mat', 'times_NSX5.mat', 'times_NSX24.mat', 'times_NSX30.mat', 'times_NSX18.mat', 'times_NSX19.mat', 'times_NSX25.mat', 'times_NSX21.mat', 'times_NSX20.mat', 'times_NSX3.mat', 'times_NSX22.mat', 'times_NSX23.mat', 'times_NSX2.mat', 'times_NSX12.mat', 'times_NSX13.mat', 'times_NSX11.mat', 'times_NSX10.mat', 'times_NSX14.mat', 'times_NSX15.mat', 'times_NSX8.mat', 'times_NSX17.mat', 'times_NSX71.mat', 'times_NSX65.mat', 'times_NSX58.mat', 'times_NSX70.mat', 'times_NSX72.mat', 'times_NSX67.mat', 'times_NSX63.mat', 'times_NSX76.mat', 'times_NSX62.mat', 'times_NSX48.mat', 'times_NSX74.mat', 'times_NSX49.mat']
    aggregate_neuron_trials = {file_name:map_electrode(file_name, image_trials_map, revision) for file_name in relevant_files}
    return aggregate_neuron_trials

def build_raster_plot_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision=None, file_name_to_save=None):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(1, 1)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Gorilla <> Basketball video trials: '+str(mat_file_name)+' cluster '+str(int(neuron_cluster_number)), fontsize=10, fontweight='bold')
    neural_data = list(neuron_video_trials[neuron_cluster_number].values())
    neural_data_histogram = [item for sublist in neural_data for item in sublist]
    axs.eventplot(neural_data)
    axs.set_xlabel('Time(s)', fontsize=8)
    axs.set_xlim((0,40))
    axs.set_ylabel('Trials', fontsize=8)
    axs.set_ylim((-.5,4.5))
    axs.set_yticks(list(range(0,5)))
    axs.axvline(19, color='g', linestyle='dashed', linewidth=1)
    axs.axvline(29, color='g', linestyle='dashed', linewidth=1)
    axs.axvline(0, color='r', linestyle='solid', linewidth=1)
    axs.axvline(35.65, color='r', linestyle='solid', linewidth=1)
    file_name = 'tests/clip_raster_plot.svg'
    if (file_name_to_save!=None):
        file_name = file_name_to_save
    fig.savefig(file_name, format='svg')
    plt.close()
    print('Saved ' + file_name)
    return

def build_histogram_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision=None, file_name_to_save=None):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(1, 1)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Gorilla <> Basketball video trials: '+str(mat_file_name)+' cluster '+str(int(neuron_cluster_number)), fontsize=10, fontweight='bold')
    neural_data = list(neuron_video_trials[neuron_cluster_number].values())
    neural_data_histogram = [item for sublist in neural_data for item in sublist]
    bins = list(range(-3, 41))
    axs.hist(neural_data_histogram, bins, alpha=0.5)
    #axs.set_xlim((-3,40))
    axs.set_xlim((0,40))
    axs.set_xlabel('Time(s) [bin size = 1s]', fontsize=8)
    axs.set_ylabel('Count', fontsize=8)
    axs.set_ylim((0,30))
    axs.axvline(19, color='g', linestyle='dashed', linewidth=1, label='Gorilla on-screen')
    axs.axvline(29, color='g', linestyle='dashed', linewidth=1)
    axs.axvline(0, color='r', linestyle='solid', linewidth=1, label='Movie on-screen')
    axs.axvline(35.65, color='r', linestyle='solid', linewidth=1)
    average_rounded, average_plus_two_std_rounded, standard_deviation = build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision)
    #axs.axhline(average_plus_two_std_rounded, color='black', linestyle='dashed', linewidth=1, label='Mean + 2*std')
    axs.axhline(average_rounded+(3*standard_deviation), color='black', linestyle='dashed', linewidth=1, label='Mean + 3*std')
    axs.axhline(average_rounded, color='black', linestyle='solid', linewidth=1, label='Mean')
    file_name = 'tests/clip_histogram.svg'
    if (file_name_to_save!=None):
        file_name = file_name_to_save
    fig.savefig(file_name, format='svg')
    plt.close()
    print('Saved ' + file_name)
    return

def build_statistically_significant_response_end_of_video_programmatic(average_firing_rate_dictionary, stimuli_interval=[33,36], revision=None):
    statistically_significant_neurons = []
    # [1] Read in all neurons
    all_channel_neurons = build_all_channel_neurons(revision)
    # [2] Reduce to neurons where any 1s bin within stimuli window >= (mean + 2td)
    for channel_neuron in all_channel_neurons:
        print(str(all_channel_neurons.index(channel_neuron)) + ' of ' + str(len(all_channel_neurons)))
        cluster = int(channel_neuron[len('cluster_'):channel_neuron.index('_neuron_')])
        neuron_cluster_number = int(channel_neuron[channel_neuron.index('_neuron_')+len('_neuron_'):])
        mat_file_name = 'timesNSX_'+str(cluster)+'.mat'
        mat_file_name = convert_mat_filename_to_revision(mat_file_name, revision)
        # Analyze
        # Video histogram - seconds 0 to 36 (index 0 = second from 0 to 1 - [0,1])
        video_histogram = build_video_histogram(mat_file_name, neuron_cluster_number, six_video_map, timestamps, revision)
        if any([(el > average_firing_rate_dictionary[channel_neuron]['average_firing_rate']+3*average_firing_rate_dictionary[channel_neuron]['standard_deviation']) for el in video_histogram[stimuli_interval[0]:stimuli_interval[1]]]):
            # [3] Remove neurons where >= 10 bins >= (mean + 2std) [not responding to epoch but rather entire video]
            #if len([el for el in video_histogram if (el > average_firing_rate_dictionary[channel_neuron]['average_firing_rate']+4*average_firing_rate_dictionary[channel_neuron]['standard_deviation'])]) < 10:
            if (len([el for el in video_histogram if (el > average_firing_rate_dictionary[channel_neuron]['average_firing_rate']+3*average_firing_rate_dictionary[channel_neuron]['standard_deviation'])]) < 10):
                statistically_significant_neurons.append(channel_neuron)
    return statistically_significant_neurons

def build_epoch_analysis_single_neuron(mat_file_name, neuron_cluster_number, six_video_map, timestamps, stimuli_interval=[32,35.6], unaware_trials=[1,2], aware_trials=[3,4,5,6], revision=None):
    from statistics import mean, median
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    unaware_spike_count = len([el for el in [item for sublist in [neuron_video_trials[neuron_cluster_number][i] for i in unaware_trials] for item in sublist] if (el >= stimuli_interval[0]) and (el < stimuli_interval[1])])
    aware_spike_count = len([el for el in [item for sublist in [neuron_video_trials[neuron_cluster_number][i] for i in aware_trials] for item in sublist] if (el >= stimuli_interval[0]) and (el < stimuli_interval[1])])
    epoch_analysis_single_neuron = {'unaware_spike_count':unaware_spike_count, 'aware_spike_count':aware_spike_count}
    return epoch_analysis_single_neuron

def build_epoch_analysis_multiple_neurons(statistically_significant_neurons, six_video_map, timestamps, stimuli_interval=[33,34], unaware_trials=[1,2], aware_trials=[3,4,5,6], revision=None):
    epoch_analysis_multiple_neurons = {}
    for el in statistically_significant_neurons:
        cluster = int(el[len('cluster_'):el.index('_neuron_')])
        neuron_cluster_number = int(el[el.index('_neuron_')+len('_neuron_'):])
        mat_file_name = 'timesNSX_'+str(cluster)+'.mat'
        mat_file_name = convert_mat_filename_to_revision(mat_file_name, revision)
        epoch_analysis_single_neuron = build_epoch_analysis_single_neuron(mat_file_name, neuron_cluster_number, six_video_map, timestamps, stimuli_interval, unaware_trials, aware_trials, revision)
        epoch_analysis_multiple_neurons['cluster_'+str(cluster)+'_neuron_'+str(neuron_cluster_number)] = epoch_analysis_single_neuron
    return epoch_analysis_multiple_neurons

def build_statistical_significance(statistically_significant_neurons, average_firing_rate_dictionary, epoch_interval_1=[33,36], epoch_interval_2=[8,31], unaware_trials=[1,2], aware_trials=[3,4,5], file_name_to_save='tests/rumination.svg', revision=None):
    # [0] Build data
    ## [0a] Epoch analysis 1
    epoch_analysis_multiple_neurons_epoch_1 = build_epoch_analysis_multiple_neurons(statistically_significant_neurons, six_video_map, timestamps, epoch_interval_1, unaware_trials, aware_trials, revision)
    epoch_analysis_multiple_neurons_epoch_2 = build_epoch_analysis_multiple_neurons(statistically_significant_neurons, six_video_map, timestamps, epoch_interval_2, unaware_trials, aware_trials, revision)
    unaware_aware_delta_epoch_1 = {}
    for channel_neuron in list(epoch_analysis_multiple_neurons_epoch_1.keys()):
        # (average spikes per aware trial) distance from mean (std) - (average spikes per unaware trial) distance from mean (std)
        unaware_aware_delta_epoch_1[channel_neuron] = (((epoch_analysis_multiple_neurons_epoch_1[channel_neuron]['aware_spike_count']/len(aware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation']) - (((epoch_analysis_multiple_neurons_epoch_1[channel_neuron]['unaware_spike_count']/len(unaware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation'])
    ## [0b] Epoch analysis 2
    unaware_aware_delta_epoch_2 = {}
    for channel_neuron in list(epoch_analysis_multiple_neurons_epoch_2.keys()):
        unaware_aware_delta_epoch_2[channel_neuron] = (((epoch_analysis_multiple_neurons_epoch_2[channel_neuron]['aware_spike_count']/len(aware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation']) - (((epoch_analysis_multiple_neurons_epoch_2[channel_neuron]['unaware_spike_count']/len(unaware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation'])
    # [1] Plot
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(8,6))
    #bins=np.linspace(-10,40,50)
    bins=np.linspace(-15,15,30)
    plt.hist(list(unaware_aware_delta_epoch_1.values()), bins=bins, alpha=0.5, label="Epoch 1")
    plt.hist(list(unaware_aware_delta_epoch_2.values()), bins=bins, alpha=0.5, label="Epoch 2")
    plt.legend()
    plt.ylabel('Count (# of neurons)')
    plt.xlabel('spike activity % change from unaware to aware trials')
    plt.title("Spike activity is significantly inhibited during trials where the patient is unaware")
    #plt.savefig("chi_squared_average_histograms.png")
    format = file_name_to_save[file_name_to_save.index('.')+len('.'):]
    plt.savefig(file_name_to_save, format=format)
    plt.close()
    # [2] Statistical (t-)tests
    from scipy import stats
    ## [2a] Decrease in spike activity between aware and unaware trials] (assumes mean 0)
    print(stats.ttest_1samp(list(unaware_aware_delta_epoch_1.values())+list(unaware_aware_delta_epoch_2.values()),0.0))
    # degrees_of_freedom = len(list(unaware_aware_delta_epoch_1.values())+list(unaware_aware_delta_epoch_2.values()))-1
    ## [2b] Comparison between end of video event and random part of video]
    print(stats.ttest_ind(list(unaware_aware_delta_epoch_1.values()),list(unaware_aware_delta_epoch_2.values())))
    # [3] By brain region (amalygda vs. x vs...)
    # degrees_of_freedom = len(list(unaware_aware_delta_epoch_1.values())) + len(list(unaware_aware_delta_epoch_2.values())) - 2
    # F(211)
    return

def build_data_points(aggregate_neuron_trials, mat_file_name='timesNSX_17.mat', neuron_cluster_number=1, image=8):
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    neural_data = list(neuron_trials_specific_neuron[image].values())
    data_points = [item for sublist in [[(j,i) for j in neural_data[i] if (j>=-0.5) and (j<=1.0)] for i in list(range(len(neural_data)))] for item in sublist]
    relevant_data_points = [el for el in data_points if el[0]>=0.25]
    relevant_onsets = [(min([j[0] for j in relevant_data_points if j[1]==trial]),trial) for trial in list(set([el[1] for el in relevant_data_points]))]# [GOOD]
    return data_points, relevant_onsets

def build_bar_charts_from_onsets(relevant_onsets, channel, neuron, image, file_name_to_save=None):
    import numpy as np
    import matplotlib.pyplot as plt
    heights, bins = np.histogram([i[0] for i in relevant_onsets], bins=np.linspace(-0.5,1,16))
    heights_percentage = [i/sum(heights) for i in heights]
    print(sum(heights), sum(heights_percentage))
    plt.figure()
    plt.bar([str(round(i,1)) for i in list(bins[:-1])], [i for i in heights_percentage], align='edge')
    plt.ylim(0,1)
    if file_name_to_save!=None:
        plt.savefig(file_name_to_save)
    else:
        plt.savefig('tests/learning_gradient_channel'+str(channel)+'neuron'+str(neuron)+'image'+str(image)+'.svg')
    plt.close()
    return

def build_average_spike_distance_normalized(main_neuron_data_eg_22=neural_data_cluster_22_neuron_2, comparison_neuron_data_eg_17=neural_data_cluster_17_neuron_1, start_time_seconds=19.0, end_time_seconds=29.0):
    #### THIS FUNCTION IS NO LONGER NORMALIZED [ONLY IN NAME] ####
    # Relevant neural data
    main_neuron_data_eg_22_relevant_time_period = {trial:[el for el in main_neuron_data_eg_22[trial] if (el>=start_time_seconds) and (el<end_time_seconds)] for trial in list(main_neuron_data_eg_22.keys())}
    comparison_neuron_data_eg_17_relevant_time_period = {trial:[el for el in comparison_neuron_data_eg_17[trial] if (el>=start_time_seconds) and (el<end_time_seconds)] for trial in list(comparison_neuron_data_eg_17.keys())}
    from statistics import mean
    import numpy as np
    min_spike_distances = []
    for trial in list(main_neuron_data_eg_22_relevant_time_period.keys()):
        for spike in main_neuron_data_eg_22_relevant_time_period[trial]:
            if len(comparison_neuron_data_eg_17_relevant_time_period[trial])>0:
                min_distance = min([abs(spike-el) for el in comparison_neuron_data_eg_17_relevant_time_period[trial]])
                min_spike_distances.append(min_distance)
    average_spike_distance = mean(min_spike_distances)#0.0945454545454548
    return average_spike_distance

def build_all_average_spike_distance_normalized(main_neuron_data_eg_22, six_video_map, start_time_seconds=19.0, end_time_seconds=29.0, revision=None):
    from statistics import stdev
    all_channel_neurons = build_all_channel_neurons(revision)
    def findnth(haystack, needle, n):
        parts = haystack.split(needle, n+1)
        if len(parts)<=n+1:
            return -1
        return len(haystack)-len(parts[-1])-len(needle)
    all_channel_neuron_pairs = [{'cluster':str(row[findnth(row,'_',0)+len('_'):findnth(row,'_',1)]),'neuron':str(row[findnth(row,'_',2)+len('_'):])} for row in all_channel_neurons]
    all_average_spike_distance_normalized = []
    for channel_neuron_pair in all_channel_neuron_pairs:
        if channel_neuron_pair != {'cluster': '22', 'neuron': '2'}:#Exclude original neuron
            print(channel_neuron_pair['cluster'], channel_neuron_pair['neuron'])
            comparison_neuron_data = map_electrode_to_videos(convert_mat_filename_to_revision('timesNSX_'+str(channel_neuron_pair['cluster'])+'.mat', revision), six_video_map, revision)[int(channel_neuron_pair['neuron'])]
            main_neuron_data_eg_22_relevant_time_period = {trial:[el for el in main_neuron_data_eg_22[trial] if (el>=start_time_seconds) and (el<end_time_seconds)] for trial in list(main_neuron_data_eg_22.keys())}
            comparison_neuron_data_relevant_time_period = {trial:[el for el in comparison_neuron_data[trial] if (el>=start_time_seconds) and (el<end_time_seconds)] for trial in list(comparison_neuron_data.keys())}
            # Both have at least 1 point within the range of the same trial (at least 1 trial)
            if any([main_neuron_data_eg_22_relevant_time_period[trial]!=[] and comparison_neuron_data_relevant_time_period[trial]!=[] for trial in list(main_neuron_data_eg_22_relevant_time_period.keys())]):
                all_average_spike_distance_normalized.append(build_average_spike_distance_normalized(main_neuron_data_eg_22=main_neuron_data_eg_22, comparison_neuron_data_eg_17=comparison_neuron_data, start_time_seconds=19.0, end_time_seconds=29.0))
    return all_average_spike_distance_normalized

def plot_average_spike_distance_normalized(all_average_spike_distance_normalized, file_name_to_save='1_to_1_cross_correlation_normalized_for_random_spikesv2.png'):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(8,8))
    #bins=np.linspace(0,2.1,50)
    bins=np.linspace(0,8.25,50)
    plt.hist(all_average_spike_distance_normalized, bins=bins, alpha=1.0)
    #plt.legend()
    #plt.ylabel('Count (# of neurons)')
    #plt.xlabel('spike activity % change from unaware to aware trials')
    plt.title("22 is significantly more correlated to 17 than any other neuron")
    #plt.ylim([0,3.5])
    plt.xlim([0,8.25])
    #plt.savefig("chi_squared_average_histograms.png")
    file_format = file_name_to_save[file_name_to_save.index('.')+len('.'):]
    plt.savefig(file_name_to_save, format=file_format)
    plt.close()
    return

def build_raster_plot_as_vector_video_17_22(neuron_video_trials_17_neuron_1, neuron_video_trials_22_neuron_2, timestamps, file_name_to_save=None):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(1, 1)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    neural_data_17 = list(neuron_video_trials_17_neuron_1.values())
    neural_data_22 = list(neuron_video_trials_22_neuron_2.values())
    axs.eventplot(neural_data_17, color='orange', alpha=0.5, label='17')
    axs.eventplot(neural_data_22, color='blue', alpha=0.5, label='22')
    axs.set_xlabel('Time(s)', fontsize=8)
    axs.set_xlim((19,29))
    axs.set_ylabel('Trials', fontsize=8)
    axs.set_ylim((0,5))
    axs.set_yticks(list(range(0,6)))
    axs.axvline(19, color='g', linestyle='dashed', linewidth=1, alpha=0.2)
    axs.axvline(29, color='g', linestyle='dashed', linewidth=1, alpha=0.2)
    axs.axvline(0, color='r', linestyle='solid', linewidth=1, alpha=0.2)
    axs.axvline(35.65, color='r', linestyle='solid', linewidth=1, alpha=0.2)
    if file_name_to_save!=None:
        fig.savefig(file_name_to_save, format='svg')
    else:
        file_name_to_save = 'tests/figure6a.svg'
        fig.savefig(file_name_to_save, format='svg')
    plt.close()
    print('Saved ' + file_name_to_save)
    return

def read_mat_file_lfp_filter(mat_file_name):
    from scipy.io import loadmat
    mat_file_name = 'data/LFP/'+mat_file_name
    mat_data = loadmat(mat_file_name)
    lfp_filtered = [el[0] for el in mat_data['lfp']]
    return lfp_filtered

def read_mat_file_lfp_filter_73(mat_file_name):
    import mat73
    mat_file_name = 'data/LFP/'+mat_file_name
    mat_data = mat73.loadmat(mat_file_name)
    lfp_filtered = [el for el in mat_data['lfp']]
    return lfp_filtered

def map_neuron_spikes_in_seconds(mat_file_name, cluster, revision=None):
    mata_data = read_mat_file_revision1(mat_file_name, revision)
    cluster_class = mata_data['cluster_class']
    neurons = list(set([el[0] for el in cluster_class]))
    neuron_spikes = {neuron:[el[1]/1000 for el in cluster_class if el[0]==neuron] for neuron in neurons}
    return neuron_spikes[cluster]

def build_circular_histogram(spike_degree_buckets, graph_title, file_name_to_save, file_format='svg', max_yaxis=None):
    import numpy as np
    from statistics import mean
    import matplotlib.pyplot as plt
    N = 20
    aggregate_bins = [[i*18,i*18+18] for i in list(range(0,N))]
    spike_degree_buckets_aggregate = [[spike for spike in spike_degree_buckets if (spike>=bin[0]) and (spike<bin[1])] for bin in aggregate_bins]
    radii = [len(el) for el in spike_degree_buckets_aggregate]
    width = (2*np.pi) / N
    bottom = 0
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)
    #### Axis scale - In case we need to set all to the same scale
    if max_yaxis!=None:
        ax.set_ylim([0,max_yaxis])
        #ax.set_ylim([0,3])
    bars = ax.bar(theta, radii, width=width, bottom=bottom)
    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.5)
    #### AVERAGE (RED) LINE ####
    # Because of 0=360, we need to compute unit vectors from the angles and take the angle of their average - https://en.wikipedia.org/wiki/Mean_of_circular_quantities, https://stackoverflow.com/questions/491738/how-do-you-calculate-the-average-of-a-set-of-circular-data
    degrees_in_radians = np.radians(spike_degree_buckets)
    x,y = 0,0
    for angle in degrees_in_radians:
        x+= np.cos(angle)
        y+= np.sin(angle)
    average_angle_radians = np.arctan2(y,x)
    average_angle_degrees = np.degrees(average_angle_radians)
    radii_average = [max(radii)]
    theta_average = [(average_angle_degrees/360)*(2*np.pi)]
    ax.plot((0,theta_average[0]),(0,radii_average[0]), color='r', linewidth=4)
    #### END AVERAGE (RED) LINE ####
    plt.title(graph_title)
    plt.savefig(file_name_to_save+'.'+file_format, format=file_format)
    plt.close()
    return

def build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds):
    wave_start_time_seconds, wave_end_time_seconds = int(lfp_wave_x[0]), int(lfp_wave_x[len(lfp_wave_x)-1])
    # [0] Identify peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(lfp_wave_y, height=None)
    # [1] Form peak intervals [e.g., peak 1 --> [0.2s,0.3s]] #ONLY COUNT THOSE BETWEEN PEAKS, NOT START/END (false positives)
    peak_intervals_indexes = [[peaks[i], peaks[i+1]] for i in list(range(len(peaks)-1))]
    # [2] Divide each peak interval into n=360 degrees - create a map for {peak_interval_start_1: degree, ..., peak_interval_start_n:map}
    import numpy as np
    wave_dictionary = {}
    for i in list(range(len(peak_intervals_indexes))):
        wave_dictionary[i] = {'lfp_index_start':peak_intervals_indexes[i][0], 'lfp_index_end':peak_intervals_indexes[i][1]}
        wave_dictionary[i]['start_time_seconds_53m_experiment'] = lfp_wave_x[peak_intervals_indexes[i][0]]
        wave_dictionary[i]['end_time_seconds_53m_experiment'] = lfp_wave_x[peak_intervals_indexes[i][1]]
        degrees_in_seconds = np.linspace(lfp_wave_x[peak_intervals_indexes[i][0]], lfp_wave_x[peak_intervals_indexes[i][1]], 361)
        degrees_in_seconds_intervals = [[degrees_in_seconds[j], degrees_in_seconds[j+1]] for j in list(range(len(degrees_in_seconds)-1))]
        wave_dictionary[i]['degrees_to_second_intervals'] = {j:degrees_in_seconds_intervals[j] for j in list(range(len(degrees_in_seconds_intervals)))}
    # [3] For each spike, determine which degree it falls into
    spike_occurrences_x = [spike for spike in neuron_spikes_in_seconds if (spike >= wave_start_time_seconds) and (spike < wave_end_time_seconds)]
    spike_degrees = []
    for spike in spike_occurrences_x:
        for wave_index in list(wave_dictionary.keys()):
            if (spike >= wave_dictionary[wave_index]['start_time_seconds_53m_experiment']) and (spike < wave_dictionary[wave_index]['end_time_seconds_53m_experiment']):
                for degree in list(wave_dictionary[wave_index]['degrees_to_second_intervals'].keys()):
                    if (spike >= wave_dictionary[wave_index]['degrees_to_second_intervals'][degree][0]) and (spike < wave_dictionary[wave_index]['degrees_to_second_intervals'][degree][1]):
                        spike_degrees.append(degree)
    return spike_degrees

def build_circular_histogram_screening_all_trials(image_trials_map, channel=22, neuron=2, relevant_image=6, relevant_trials=None, wave='alpha', file_name_to_save=None, revision=None):
    import numpy as np
    #from statistics import mean
    if relevant_trials==None:
        relevant_trials = list(range(len(image_trials_map[relevant_image])))
    lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    lfp_filtered = read_mat_file_lfp_filter(lfp_filtered_file_name)
    filtered_lfp_file_name = 'lfp'+str(channel)
    neuron_spikes_mat_file_name = 'timesNSX_'+str(int(filtered_lfp_file_name[len('lfp'):]))+'.mat'#'timesNSX_22.mat'
    neuron_spikes_mat_file_name = convert_mat_filename_to_revision(neuron_spikes_mat_file_name, revision)
    #neuron = 2
    neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron, revision)
    ####
    spike_degrees_images = {image:{} for image in list(image_trials_map.keys())}
    for image in list(image_trials_map.keys()):
        for k in list(range(len(image_trials_map[image]))):
            if k in relevant_trials:
                image_start_time_seconds, image_end_time_seconds = image_trials_map[image][k][0], image_trials_map[image][k][1]
                sr = 30000
                start_time_sr, end_time_sr = int(image_start_time_seconds*sr), int(image_end_time_seconds*sr)
                lfp_wave_y = lfp_filtered[start_time_sr:end_time_sr]
                lfp_wave_x = np.linspace(image_start_time_seconds,image_end_time_seconds,len(lfp_wave_y))
                spike_degrees = build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds)
                spike_degrees_images[image][k] = spike_degrees
    spike_degree_buckets = [item for sublist in list(spike_degrees_images[relevant_image].values()) for item in sublist]
    #### Save
    if file_name_to_save!=None:
        file_name_to_save = file_name_to_save
    else:
        file_name_to_save = 'tests/'+wave+'_cluster'+str(channel)+'_neuron2_image'+str(relevant_image)
    graph_title = '12 screening trials - '+wave +' wave ' + ' image ' + str(relevant_image)
    build_circular_histogram(spike_degree_buckets, graph_title, file_name_to_save, file_format='svg')
    return

def build_circular_histogram_video_all_trials(channel=22, neuron=2, relevant_trials=None, wave='alpha', event='gorilla_video_10s_on_screen', file_name_to_save=None, revision=None):
    print(channel,neuron,wave,event)
    import numpy as np
    lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    #if int(channel) not in [13, 16, 61, 67, 69, 70, 72, 78]
    try:
        lfp_filtered = read_mat_file_lfp_filter(lfp_filtered_file_name)
    except:
        lfp_filtered=read_mat_file_lfp_filter_73(lfp_filtered_file_name)
    video_trials = list(six_video_map.keys())
    if relevant_trials == None:
        relevant_trials = video_trials
    events = {'gorilla_video_10s_on_screen':{'graph_title':'Gorilla on-screen video', 'start_from_video_start_seconds':19, 'end_from_video_start_seconds':29}, 'end_of_video_event':{'graph_title':'end_of_video_event', 'start_from_video_start_seconds':33, 'end_from_video_start_seconds':36}, 'random_video_selection':{'graph_title':'random_video_selection', 'start_from_video_start_seconds':3, 'end_from_video_start_seconds':18}, 'beginning_of_video_event':{'graph_title':'beginning_of_video_event', 'start_from_video_start_seconds':0, 'end_from_video_start_seconds':3}}
    #event = 'gorilla_video_10s_on_screen'
    aggregate_spike_degrees = []
    for video_trial in video_trials:
        print('beginning video trial: ' + str(video_trial))
        if video_trial in relevant_trials:
            #import numpy as np
            #video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0], six_video_map[video_trial]['timestamp_range_seconds'][1]
            video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0]+events[event]['start_from_video_start_seconds'], six_video_map[video_trial]['timestamp_range_seconds'][0]+events[event]['end_from_video_start_seconds']
            sr = 30000
            start_time_sr = int(video_start_time_seconds*sr)
            end_time_sr = int(video_end_time_seconds*sr)
            lfp_wave_y = lfp_filtered[start_time_sr:end_time_sr]
            lfp_wave_x = np.linspace(video_start_time_seconds,video_end_time_seconds,len(lfp_wave_y))
            #
            filtered_lfp_file_name = 'lfp'+str(channel)
            neuron_spikes_mat_file_name = 'timesNSX_'+str(int(filtered_lfp_file_name[len('lfp'):]))+'.mat'#'timesNSX_22.mat'
            neuron_spikes_mat_file_name = convert_mat_filename_to_revision(neuron_spikes_mat_file_name, revision)
            neuron = int(neuron)
            neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron, revision)
            #
            spike_degrees = build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds)
            print(spike_degrees)
            #spike_degrees_movies[video_trial] = spike_degrees
            aggregate_spike_degrees += spike_degrees
    spike_degree_buckets = aggregate_spike_degrees
    if file_name_to_save!=None:
        file_name_to_save = file_name_to_save
    else:
        file_name_to_save = 'tests/'+wave+'_cluster'+str(channel)+'_neuron'+str(neuron) +'_'+events[event]['graph_title'].replace(' ','_')+ '_trial_'+str(min(relevant_trials))+'_to_'+str(max(relevant_trials))
    #graph_title = '6 video trials - '+wave +' wave'
    graph_title = events[event]['graph_title']
    print(file_name_to_save)
    #build_circular_histogram(spike_degree_buckets, graph_title, file_name_to_save, file_format='svg', max_yaxis=3)
    build_circular_histogram(spike_degree_buckets, graph_title, file_name_to_save, file_format='svg')
    return

def convert_mat_filename_to_revision(mat_file_name='timesNSX_17.mat', revision=None):
    if revision==None:
        revision_mat_file_name = mat_file_name
    else:
        revision_mat_file_name = 'times_'+mat_file_name.replace('_','')[len('times'):]
    return revision_mat_file_name

def build_all_channel_neurons(revision=None):
    all_channel_neurons = ['cluster_87_neuron_1', 'cluster_50_neuron_1', 'cluster_50_neuron_2', 'cluster_79_neuron_1', 'cluster_45_neuron_1', 'cluster_45_neuron_2', 'cluster_51_neuron_1', 'cluster_51_neuron_2', 'cluster_47_neuron_1', 'cluster_47_neuron_2', 'cluster_47_neuron_3', 'cluster_85_neuron_1', 'cluster_85_neuron_2', 'cluster_85_neuron_3', 'cluster_42_neuron_1', 'cluster_57_neuron_1', 'cluster_80_neuron_1', 'cluster_80_neuron_2', 'cluster_69_neuron_1', 'cluster_69_neuron_2', 'cluster_41_neuron_1', 'cluster_68_neuron_1', 'cluster_68_neuron_2', 'cluster_27_neuron_1', 'cluster_27_neuron_2', 'cluster_26_neuron_1', 'cluster_7_neuron_1', 'cluster_5_neuron_1', 'cluster_24_neuron_1', 'cluster_24_neuron_2', 'cluster_30_neuron_1', 'cluster_30_neuron_2', 'cluster_18_neuron_1', 'cluster_18_neuron_2', 'cluster_18_neuron_3', 'cluster_19_neuron_1', 'cluster_19_neuron_2', 'cluster_19_neuron_3', 'cluster_25_neuron_1', 'cluster_25_neuron_2', 'cluster_21_neuron_1', 'cluster_20_neuron_1', 'cluster_3_neuron_1', 'cluster_3_neuron_2', 'cluster_22_neuron_1', 'cluster_22_neuron_2', 'cluster_23_neuron_1', 'cluster_23_neuron_2', 'cluster_2_neuron_1', 'cluster_2_neuron_2', 'cluster_2_neuron_3', 'cluster_12_neuron_1', 'cluster_13_neuron_1', 'cluster_11_neuron_1', 'cluster_11_neuron_2', 'cluster_11_neuron_3', 'cluster_10_neuron_1', 'cluster_10_neuron_2', 'cluster_14_neuron_1', 'cluster_14_neuron_2', 'cluster_15_neuron_1', 'cluster_8_neuron_1', 'cluster_17_neuron_1', 'cluster_17_neuron_2', 'cluster_71_neuron_1', 'cluster_65_neuron_1', 'cluster_65_neuron_2', 'cluster_58_neuron_1', 'cluster_58_neuron_2', 'cluster_70_neuron_1', 'cluster_70_neuron_2', 'cluster_72_neuron_1', 'cluster_67_neuron_1', 'cluster_67_neuron_2', 'cluster_67_neuron_3', 'cluster_63_neuron_1', 'cluster_76_neuron_1', 'cluster_76_neuron_2', 'cluster_62_neuron_1', 'cluster_48_neuron_1', 'cluster_48_neuron_2', 'cluster_74_neuron_1', 'cluster_74_neuron_2', 'cluster_49_neuron_1', 'cluster_49_neuron_2']
    return all_channel_neurons

def append_baseline_firing_rate_by_region_to_aggregate_statistics(average_firing_rate_dictionary, aggregate_statistics_file_name='aggregate_statistics_by_revision1.csv'):
    import json
    from statistics import mean, stdev
    # [0] Read in aggregate_statistics
    csvdataRows = readCSV(aggregate_statistics_file_name)
    csvdataRows[0] += ['baseline_firing_rate', 'baseline_firing_rate_stdev']
    for row in csvdataRows[1:]:
        if int(row[csvdataRows[0].index('neuron_count')])!=0:
            neurons = json.loads(row[csvdataRows[0].index('neurons')])
            neuron_region_statistics = {'baseline_firing_rate':[], 'baseline_firing_rate_stdev':[]}
            for neuron in neurons:
                baseline_firing_rate, baseline_firing_rate_stdev = average_firing_rate_dictionary[neuron]['average_firing_rate'], average_firing_rate_dictionary[neuron]['standard_deviation']
                neuron_region_statistics['baseline_firing_rate'].append(baseline_firing_rate)
                neuron_region_statistics['baseline_firing_rate_stdev'].append(baseline_firing_rate_stdev)
                # STDEV OF THE MEAN, NOT MEAN OF THE STDEV
            row += [mean(neuron_region_statistics['baseline_firing_rate']), stdev(neuron_region_statistics['baseline_firing_rate_stdev']) if len(neuron_region_statistics['baseline_firing_rate_stdev'])>1 else "N/A"]
        else:
            row += ['N/A', 'N/A']
    # Save
    writeToCSV(aggregate_statistics_file_name, csvdataRows)
    return

def build_total_number_of_responsive_neurons(aggregate_statistics_file_name='aggregate_statistics_by_revision1.csv'):
    import json
    # [0] Read in aggregate_statistics
    csvdataRows = readCSV(aggregate_statistics_file_name)
    all_responsive_neurons = []
    for row in csvdataRows[1:]:
        if int(row[csvdataRows[0].index('neuron_count')])!=0:
            all_responsive_neurons += json.loads(row[csvdataRows[0].index('neurons')])
    all_responsive_neurons = list(set(all_responsive_neurons))
    return len(all_responsive_neurons), all_responsive_neurons

def build_raster_plot(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, directory_to_save_in='revision1/spike_raster_plots_images/'):
    import matplotlib.pyplot as plt
    # [0] Set super title etc across saved figure
    fig, axs = plt.subplots(3, 3)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle(mat_file_name + ', cluster: ' + str(int(neuron_cluster_number)), fontsize=10, fontweight='bold')# electrode file name, neuron cluster
    image_to_plot_position_map = {1:[0,0], 2:[0,1], 3:[0,2], 4:[1,0], 5:[1,1], 6:[1,2], 7:[2,0], 8:[2,1], 9:[2,2]}
    for image in list(neuron_trials_specific_neuron.keys()):
        neural_data = list(neuron_trials_specific_neuron[image].values())
        axs[image_to_plot_position_map[image][0], image_to_plot_position_map[image][1]].eventplot(neural_data)
        axs[image_to_plot_position_map[image][0], image_to_plot_position_map[image][1]].set_title(image_map[image], fontsize=8)
        axs[image_to_plot_position_map[image][0], image_to_plot_position_map[image][1]].set_xlabel('Time(s)', fontsize=8)
        axs[image_to_plot_position_map[image][0], image_to_plot_position_map[image][1]].set_xlim([-0.5,1])
        axs[image_to_plot_position_map[image][0], image_to_plot_position_map[image][1]].tick_params(labelsize=6)
        axs[image_to_plot_position_map[image][0], image_to_plot_position_map[image][1]].set_ylabel('Trial number', fontsize=8)
        axs[image_to_plot_position_map[image][0], image_to_plot_position_map[image][1]].set_ylim([0,len(neuron_trials_specific_neuron[image])])
    #fig.show()
    file_name_png = directory_to_save_in+mat_file_name[:mat_file_name.index('.mat')]+'_neuron_'+str(int(neuron_cluster_number))+'.png'
    fig.savefig(file_name_png)
    plt.close()
    print('Saved ' + file_name_png)
    return

def build_raster_plot_with_histogram_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, directory_to_save_in='revision1/video_response_graphs/', revision=None):
    import matplotlib.pyplot as plt
    import numpy as np
    # [0] General elements
    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Gorilla <> Basketball video trials: '+str(mat_file_name)+' cluster '+str(int(neuron_cluster_number)), fontsize=10, fontweight='bold')
    #image_to_plot_position_map = {1:[0,0], 2:[0,1], 3:[0,2], 4:[1,0], 5:[1,1], 6:[1,2], 7:[2,0], 8:[2,1], 9:[2,2]}
    neural_data = list(neuron_video_trials[neuron_cluster_number].values())
    neural_data_histogram = [item for sublist in neural_data for item in sublist]
    # [1] Histogram
    axs[1].hist(neural_data_histogram, list(range(-3, 41)))
    axs[1].set_xlim((-3,40))
    axs[1].set_xlabel('Time(s) [bin size = 1s]', fontsize=8)
    axs[1].set_ylabel('Count', fontsize=8)
    axs[1].axvline(19, color='g', linestyle='dashed', linewidth=1, label='Gorilla on-screen')
    axs[1].axvline(29, color='g', linestyle='dashed', linewidth=1)
    axs[1].axvline(0, color='r', linestyle='solid', linewidth=1, label='Movie on-screen')
    axs[1].axvline(35.65, color='r', linestyle='solid', linewidth=1)
    average_rounded, average_plus_two_std_rounded, standard_deviation = build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision)
    #axs[1].axhline(average_plus_two_std_rounded, color='black', linestyle='dashed', linewidth=1, label='Mean + 2*std')
    axs[1].axhline(average_rounded+(3*standard_deviation), color='black', linestyle='dashed', linewidth=1, label='Mean + 3*std')
    axs[1].axhline(average_rounded, color='black', linestyle='solid', linewidth=1, label='Mean')
    axs[1].legend(fontsize=6)
    # [1] Raster plot
    axs[0].eventplot(neural_data)
    #axs[0].set_title('tbd', fontsize=8)
    axs[0].set_xlabel('Time(s)', fontsize=8)
    axs[0].set_xlim((-3,40))
    axs[0].set_ylabel('Trials', fontsize=8)
    axs[0].set_ylim((0,5))
    axs[0].set_yticks(list(range(0,6)))
    axs[0].axvline(19, color='g', linestyle='dashed', linewidth=1)
    axs[0].axvline(29, color='g', linestyle='dashed', linewidth=1)
    axs[0].axvline(0, color='r', linestyle='solid', linewidth=1)
    axs[0].axvline(35.65, color='r', linestyle='solid', linewidth=1)
    #
    #fig.show()
    file_name_png = directory_to_save_in+mat_file_name[:mat_file_name.index('.mat')]+'_neuron_'+str(int(neuron_cluster_number))+'.png'
    fig.savefig(file_name_png)
    plt.close()
    print('Saved ' + file_name_png)
    return

def all_raster_plots_screening(image_trials_map, image_map):
    aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map,1)
    #sum([len(aggregate_neuron_trials[key]) for key in list(aggregate_neuron_trials.keys())])#128
    for mat_file_name in list(aggregate_neuron_trials.keys()):
        for neuron_cluster_number in list(aggregate_neuron_trials[mat_file_name].keys()):
            neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
            build_raster_plot(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, directory_to_save_in='revision1/spike_raster_plots_images/')
    return

def all_raster_plots_video(image_trials_map, timestamps, six_video_map, revision=None):
    aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map,1)
    #sum([len(aggregate_neuron_trials[key]) for key in list(aggregate_neuron_trials.keys())])#128
    for mat_file_name in list(aggregate_neuron_trials.keys()):
        mat_file_name = convert_mat_filename_to_revision(mat_file_name, revision)
        neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
        for neuron_cluster_number in list(aggregate_neuron_trials[mat_file_name].keys()):
            build_raster_plot_with_histogram_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, directory_to_save_in='revision1/video_response_graphs/', revision=revision)
    return

def build_spike_degree_buckets_screening(image_trials_map, channel=22, neuron=2, relevant_image=6, relevant_trials=None, wave='alpha', revision=None):
    import numpy as np
    #from statistics import mean
    if relevant_trials==None:
        relevant_trials = list(range(len(image_trials_map[relevant_image])))
    lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    lfp_filtered = read_mat_file_lfp_filter(lfp_filtered_file_name)
    filtered_lfp_file_name = 'lfp'+str(channel)
    neuron_spikes_mat_file_name = 'timesNSX_'+str(int(filtered_lfp_file_name[len('lfp'):]))+'.mat'#'timesNSX_22.mat'
    neuron_spikes_mat_file_name = convert_mat_filename_to_revision(neuron_spikes_mat_file_name, revision)
    #neuron = 2
    neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron, revision)
    ####
    spike_degrees_images = {image:{} for image in list(image_trials_map.keys())}
    for image in list(image_trials_map.keys()):
        for k in list(range(len(image_trials_map[image]))):
            if k in relevant_trials:
                image_start_time_seconds, image_end_time_seconds = image_trials_map[image][k][0], image_trials_map[image][k][1]
                sr = 30000
                start_time_sr, end_time_sr = int(image_start_time_seconds*sr), int(image_end_time_seconds*sr)
                lfp_wave_y = lfp_filtered[start_time_sr:end_time_sr]
                lfp_wave_x = np.linspace(image_start_time_seconds,image_end_time_seconds,len(lfp_wave_y))
                spike_degrees = build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds)
                spike_degrees_images[image][k] = spike_degrees
    spike_degrees_screening = spike_degrees_images
    spike_degree_buckets_screening = [item for sublist in list(spike_degrees_images[relevant_image].values()) for item in sublist]
    #### Save
    return spike_degrees_screening, spike_degree_buckets_screening

def build_spike_degree_buckets_video(channel=22, neuron=2, relevant_trials=None, wave='alpha', event='gorilla_video_10s_on_screen', revision=None):
    print(channel,neuron,wave,event)
    import numpy as np
    lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    #if int(channel) not in [13, 16, 61, 67, 69, 70, 72, 78]
    try:
        lfp_filtered = read_mat_file_lfp_filter(lfp_filtered_file_name)
    except:
        lfp_filtered=read_mat_file_lfp_filter_73(lfp_filtered_file_name)
    video_trials = list(six_video_map.keys())
    if relevant_trials == None:
        relevant_trials = video_trials
    events = {'gorilla_video_10s_on_screen':{'graph_title':'Gorilla on-screen video', 'start_from_video_start_seconds':19, 'end_from_video_start_seconds':29}, 'end_of_video_event':{'graph_title':'end_of_video_event', 'start_from_video_start_seconds':33, 'end_from_video_start_seconds':36}, 'random_video_selection':{'graph_title':'random_video_selection', 'start_from_video_start_seconds':3, 'end_from_video_start_seconds':18}, 'beginning_of_video_event':{'graph_title':'beginning_of_video_event', 'start_from_video_start_seconds':0, 'end_from_video_start_seconds':3}}
    #event = 'gorilla_video_10s_on_screen'
    aggregate_spike_degrees = []
    for video_trial in video_trials:
        print('beginning video trial: ' + str(video_trial))
        if video_trial in relevant_trials:
            #import numpy as np
            #video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0], six_video_map[video_trial]['timestamp_range_seconds'][1]
            video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0]+events[event]['start_from_video_start_seconds'], six_video_map[video_trial]['timestamp_range_seconds'][0]+events[event]['end_from_video_start_seconds']
            sr = 30000
            start_time_sr = int(video_start_time_seconds*sr)
            end_time_sr = int(video_end_time_seconds*sr)
            lfp_wave_y = lfp_filtered[start_time_sr:end_time_sr]
            lfp_wave_x = np.linspace(video_start_time_seconds,video_end_time_seconds,len(lfp_wave_y))
            #
            filtered_lfp_file_name = 'lfp'+str(channel)
            neuron_spikes_mat_file_name = 'timesNSX_'+str(int(filtered_lfp_file_name[len('lfp'):]))+'.mat'#'timesNSX_22.mat'
            neuron_spikes_mat_file_name = convert_mat_filename_to_revision(neuron_spikes_mat_file_name, revision)
            neuron = int(neuron)
            neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron, revision)
            #
            spike_degrees = build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds)
            print(spike_degrees)
            #spike_degrees_movies[video_trial] = spike_degrees
            aggregate_spike_degrees += spike_degrees
    spike_degrees_video = aggregate_spike_degrees
    spike_degree_buckets_video = aggregate_spike_degrees
    return spike_degrees_video, spike_degree_buckets_video

def build_buckets_rayleigh(spike_degree_buckets):
    import numpy as np
    from statistics import mean
    import matplotlib.pyplot as plt
    N = 20
    aggregate_bins = [[i*18,i*18+18] for i in list(range(0,N))]
    spike_degree_buckets_aggregate = [[spike for spike in spike_degree_buckets if (spike>=bin[0]) and (spike<bin[1])] for bin in aggregate_bins]
    radii = [len(el) for el in spike_degree_buckets_aggregate]
    weights = radii
    buckets = [(360/N)*i for i in range(N)]
    return buckets, weights

def build_table1_neuron_by_region_count(average_firing_rate_dictionary):
    clusters = [key[key.index('cluster_')+len('cluster_'):key.index('_neuron')] for key in list(average_firing_rate_dictionary.keys())]
    regions = {'Right Orbitfronal':{'min':1, 'max':8}, 'Right Medial Hippocampus':{'min':9, 'max':16}, 'Right Amygdala':{'min':17, 'max':24}, 'Right Entorhinal Cortex':{'min':25, 'max':32}, 'Left Orbitfronal':{'min':33, 'max':40}, 'Left Anterior Hippocampus':{'min':41, 'max':48}, 'Left Amygdala':{'min':49, 'max':56}, 'Left Entorhinal Cortex':{'min':57, 'max':64}, 'Right Anterior Cingulate':{'min':65, 'max':72}, 'Right Anterior F...':{'min':73, 'max':80}, 'Left Anterior Cingulate':{'min':81, 'max':88}, 'Left Anterior F...':{'min':89, 'max':96}}
    region_count = {key:len([cluster for cluster in clusters if (int(cluster)>=regions[key]['min']) and (int(cluster)<=regions[key]['max'])]) for key in regions}
    table1_neuron_by_region_count = region_count
    return table1_neuron_by_region_count

def build_cluster_map(all_channel_neurons):
    clusters = [key[key.index('cluster_')+len('cluster_'):key.index('_neuron')] for key in list(average_firing_rate_dictionary.keys())]
    regions = {'Right Orbitfronal':{'min':1, 'max':8}, 'Right Medial Hippocampus':{'min':9, 'max':16}, 'Right Amygdala':{'min':17, 'max':24}, 'Right Entorhinal Cortex':{'min':25, 'max':32}, 'Left Orbitfronal':{'min':33, 'max':40}, 'Left Anterior Hippocampus':{'min':41, 'max':48}, 'Left Amygdala':{'min':49, 'max':56}, 'Left Entorhinal Cortex':{'min':57, 'max':64}, 'Right Anterior Cingulate':{'min':65, 'max':72}, 'Right Anterior F...':{'min':73, 'max':80}, 'Left Anterior Cingulate':{'min':81, 'max':88}, 'Left Anterior F...':{'min':89, 'max':96}}
    cluster_map = {channel_neuron:[region for region in list(regions.keys()) if (int(channel_neuron[channel_neuron.index('cluster_')+len('cluster_'):channel_neuron.index('_neuron')])>=regions[region]['min']) and (int(channel_neuron[channel_neuron.index('cluster_')+len('cluster_'):channel_neuron.index('_neuron')])<=regions[region]['max'])][0] for channel_neuron in all_channel_neurons}
    return cluster_map

def build_spike_count_across_entire_experiment(timestamps, revision):
    spike_count_across_entire_experiment = {}
    # [a] Build all channel_neurons
    all_channel_neurons = build_all_channel_neurons(revision)
    # [b] For each channel neuron, get spike count
    for channel_neuron in all_channel_neurons:
        cluster = channel_neuron[channel_neuron.index('cluster_')+len('cluster_'):channel_neuron.index('_neuron')]
        neuron_number = channel_neuron[channel_neuron.index('_neuron_')+len('_neuron_'):]
        mat_file_name = convert_mat_filename_to_revision('timesNSX_'+str(cluster)+'.mat',revision)
        mata_data = read_mat_file_revision1(mat_file_name, revision)
        cluster_class = mata_data['cluster_class']
        neurons = list(set([el[0] for el in cluster_class]))
        data_start_time = timestamps[0]/30000#this could just be 0?
        data_end_time = timestamps[len(timestamps)-1]/30000
        neuron_spikes = {neuron:[el[1]/1000 for el in cluster_class if el[0]==neuron] for neuron in neurons}
        neuron_spikes_cluster = neuron_spikes[int(neuron_number)]
        neuron_spikes_cluster_within_start_end = [el for el in neuron_spikes_cluster if (el>=data_start_time) and (el<=data_end_time)]
        spike_count_across_entire_experiment[channel_neuron] = len(neuron_spikes_cluster_within_start_end)
    return spike_count_across_entire_experiment

def save_supplementary_spike_count_table(all_channel_neurons, spike_count_across_entire_experiment):
    cluster_map = build_cluster_map(all_channel_neurons)
    csvFileName = 'spike_count_across_entire_experiment.csv'
    csvdataRows = [['region', 'cluster', 'neuron', 'spike_count']]
    for channel_neuron in all_channel_neurons:
        region = cluster_map[channel_neuron]
        cluster = channel_neuron[channel_neuron.index('cluster_')+len('cluster_'):channel_neuron.index('_neuron')]
        neuron = channel_neuron[channel_neuron.index('_neuron_')+len('_neuron_'):]
        spike_count = spike_count_across_entire_experiment[channel_neuron]
        csvdataRows.append([region,cluster,neuron,spike_count])
    writeToCSV(csvFileName, csvdataRows)
    return

def build_spike_range_gorilla_on_screen(main_neuron_data_eg_22, six_video_map, start_time_seconds=19.0, end_time_seconds=29.0, revision=None):
    range_data = []
    from statistics import mean, stdev
    all_channel_neurons = build_all_channel_neurons(revision)
    def findnth(haystack, needle, n):
        parts = haystack.split(needle, n+1)
        if len(parts)<=n+1:
            return -1
        return len(haystack)-len(parts[-1])-len(needle)
    all_channel_neuron_pairs = [{'cluster':str(row[findnth(row,'_',0)+len('_'):findnth(row,'_',1)]),'neuron':str(row[findnth(row,'_',2)+len('_'):])} for row in all_channel_neurons]
    all_average_spike_distance_normalized = []
    for channel_neuron_pair in all_channel_neuron_pairs:
        if channel_neuron_pair != {'cluster': '22', 'neuron': '2'}:#Exclude original neuron
            print(channel_neuron_pair['cluster'], channel_neuron_pair['neuron'])
            comparison_neuron_data = map_electrode_to_videos(convert_mat_filename_to_revision('timesNSX_'+str(channel_neuron_pair['cluster'])+'.mat', revision), six_video_map, revision)[int(channel_neuron_pair['neuron'])]
            main_neuron_data_eg_22_relevant_time_period = {trial:[el for el in main_neuron_data_eg_22[trial] if (el>=start_time_seconds) and (el<end_time_seconds)] for trial in list(main_neuron_data_eg_22.keys())}
            comparison_neuron_data_relevant_time_period = {trial:[el for el in comparison_neuron_data[trial] if (el>=start_time_seconds) and (el<end_time_seconds)] for trial in list(comparison_neuron_data.keys())}
            if any([main_neuron_data_eg_22_relevant_time_period[trial]!=[] and comparison_neuron_data_relevant_time_period[trial]!=[] for trial in list(main_neuron_data_eg_22_relevant_time_period.keys())]):
                range_data.append([item for sublist in list(comparison_neuron_data_relevant_time_period.values()) for item in sublist])
    range_data_count = [len(i) for i in range_data]
    spike_range_statistics = {'min':min(range_data_count), 'max':max(range_data_count), 'mean':mean(range_data_count), 'stdev':stdev(range_data_count)}
    #{'min': 1, 'max': 82, 'mean': 14, 'stdev': 15.126914074316833}
    return spike_range_statistics

def build_gorilla_neuron_screening_statistics_screening_response(image_trials_map, aggregate_neuron_trials, revision=None):
    aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map, revision)
    mat_file_name, neuron_cluster_number, images = 'times_NSX22.mat', 2, [5,6,7]
    gorilla_22_screening_response = {image:[] for image in images}
    for image in images:
        for trial in (aggregate_neuron_trials[mat_file_name][neuron_cluster_number][image].keys()):
            trial_spike_count_vector = []
            for spike in aggregate_neuron_trials[mat_file_name][neuron_cluster_number][image][trial]:
                if spike>=0 and spike<=1:
                    trial_spike_count_vector.append(spike)
            gorilla_22_screening_response[image].append(len(trial_spike_count_vector))
    return gorilla_22_screening_response

def build_gorilla_neuron_screening_statistics(image_trials_map, aggregate_neuron_trials, revision=None):
    # Gorilla neuron info
    aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map, revision)
    mat_file_name, neuron_cluster_number, images = 'times_NSX22.mat', 2, [5,6,7]
    gorilla_22_screening_response = {image:[] for image in images}
    for image in images:
        for trial in (aggregate_neuron_trials[mat_file_name][neuron_cluster_number][image].keys()):
            trial_spike_count_vector = []
            for spike in aggregate_neuron_trials[mat_file_name][neuron_cluster_number][image][trial]:
                if spike>=0 and spike<=1:
                    trial_spike_count_vector.append(spike)
            gorilla_22_screening_response[image].append(len(trial_spike_count_vector))
    from statistics import mean, stdev
    gorilla_22_screening_response_statistics = {image:{'mean_firing_rate':'tbd', 'stdev_firing_rate':'tbd'} for image in images}
    for image in images:
        gorilla_22_screening_response_statistics[image]['mean_firing_rate'] = mean(gorilla_22_screening_response[image])
        gorilla_22_screening_response_statistics[image]['stdev_firing_rate'] = stdev(gorilla_22_screening_response[image])
    # {5: {'mean_firing_rate': 2.3333333333333335, 'stdev_firing_rate': 1.4354811251305468}, 6: {'mean_firing_rate': 1.8333333333333333, 'stdev_firing_rate': 1.1146408580454255}, 7: {'mean_firing_rate': 1.6666666666666667, 'stdev_firing_rate': 1.556997888323046}}
    average_rounded, average_plus_two_std_rounded, standard_deviation = build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision)
    # (0.12, 1.11, 0.4957072630849471)
    # 0.12+3*0.4957072630849471 = 1.6071217892548413
    gorilla_neuron_screening_statistics = {'baseline':{'baseline_firing_rate_mean':average_rounded, 'baseline_firing_rate_mean_stdev':standard_deviation}, 'image_response':gorilla_22_screening_response_statistics}
    return gorilla_neuron_screening_statistics

def build_clipgorilla_neuron_screening_statistics(aggregate_neuron_trials):
    # Clip-gorilla neuron info
    mat_file_name, neuron_cluster_number, images = 'times_NSX17.mat', 1, [8]
    gorilla_17_screening_response = {image:[] for image in images}
    for image in images:
        for trial in (aggregate_neuron_trials[mat_file_name][neuron_cluster_number][image].keys()):
            trial_spike_count_vector = []
            for spike in aggregate_neuron_trials[mat_file_name][neuron_cluster_number][image][trial]:
                if spike>=0 and spike<=1:
                    trial_spike_count_vector.append(spike)
            gorilla_17_screening_response[image].append(len(trial_spike_count_vector))
    from statistics import mean, stdev
    gorilla_17_screening_response_statistics = {image:{'mean_firing_rate':'tbd', 'stdev_firing_rate':'tbd'} for image in images}
    for image in images:
        gorilla_17_screening_response_statistics[image]['mean_firing_rate'] = mean(gorilla_17_screening_response[image])
        gorilla_17_screening_response_statistics[image]['stdev_firing_rate'] = stdev(gorilla_17_screening_response[image])
    # {8: {'mean_firing_rate': 3.5, 'stdev_firing_rate': 1.8829377433825436}}
    average_rounded, average_plus_two_std_rounded, standard_deviation = build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision)
    # (0.83, 4.52, 1.8429359862827126)
    # 0.83 + 3*1.8429359862827126 = 6.358807958848137
    clipgorilla_neuron_screening_statistics = {'baseline':{'baseline_firing_rate_mean':average_rounded, 'baseline_firing_rate_mean_stdev':standard_deviation}, 'image_response':gorilla_17_screening_response_statistics}
    return clipgorilla_neuron_screening_statistics
######################################### End [0] Background functions #########################################

######################################### [1] Tables 1 + 2 (neuron counts + response properties) #########################################
def build_metrics_for_table1_and_table2():
    #### [1] Table 1 ####
    image_map, image_trials_map = build_image_trials_map_and_image_map()
    timestamps, six_video_map = build_timestamps_sixvideomap()
    revision = 1
    average_firing_rate_dictionary = build_full_average_firing_rate_dictionary(revision)
    ## Site + Total neuron columns ##
    table1_neuron_by_region_count = build_table1_neuron_by_region_count(average_firing_rate_dictionary)
    ## Pre-experiment screening and during-experiment recorded manually (small n) ##
    ## Clip response columns (this creates a .csv file that contains relevant information in columns "region", "epoch", and "neuron_count") ##
    aggregate_statistics = build_high_level_statistics_by_region(six_video_map, timestamps, average_firing_rate_dictionary, revision)
    append_baseline_firing_rate_by_region_to_aggregate_statistics(average_firing_rate_dictionary, aggregate_statistics_file_name='aggregate_statistics_by_revision1.csv')
    #### [2] Table 2 ####
    ## [i] Gorilla neuron during-experiment screening responses ##
    revision = 1
    aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map, revision)
    mat_file_name, neuron_cluster_number = 'times_NSX22.mat', 2
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    average_rounded, average_plus_two_std_rounded, standard_deviation = build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision)
    gorilla_neuron_screening_statistics=build_gorilla_neuron_screening_statistics(image_trials_map, aggregate_neuron_trials, revision)# {'baseline': {'baseline_firing_rate_mean': 0.12, 'baseline_firing_rate_mean_stdev': 0.4957072630849471}, 'image_response': {5: {'mean_firing_rate': 2.3333333333333335, 'stdev_firing_rate': 1.4354811251305468}, 6: {'mean_firing_rate': 1.8333333333333333, 'stdev_firing_rate': 1.1146408580454255}, 7: {'mean_firing_rate': 1.6666666666666667, 'stdev_firing_rate': 1.556997888323046}}}
    from statistics import mean, stdev
    images = [5,6,7]
    for j in images:
        durations = [max([i for i in neuron_trials_specific_neuron[j][trial] if i>=0.0 and i<1])-min([i for i in neuron_trials_specific_neuron[j][trial] if i>=0.0 and i<1]) if (len([i for i in neuron_trials_specific_neuron[j][trial] if i>=0.0 and i<1])>1) else None for trial in list(neuron_trials_specific_neuron[j].keys())]
        duration_mean, duration_stdev = mean([i for i in durations if i!=None]), stdev([i for i in durations if i!=None])
        onset_times = [min([i for i in neuron_trials_specific_neuron[j][trial] if i>=0.0 and i<1]) if (len([i for i in neuron_trials_specific_neuron[j][trial] if i>=0.0 and i<1])>0) else None for trial in list(neuron_trials_specific_neuron[j].keys())]
        onset_mean, onset_stdev = mean([i for i in onset_times if i!=None]), stdev([i for i in onset_times if i!=None])#(0.4490909090909091, 0.19227347947413578)
        print(j)
        print(duration_mean, duration_stdev)
        print(onset_mean, onset_stdev)
        # Image: 5
        # duration_mean, duration_stdev: 0.42625 0.059506902360746614
        # onset_mean, onset_stdev: 0.41600000000000004 0.11107554986484551
        # Image: 6
        # duration_mean, duration_stdev: 0.2571428571428571 0.14056010407346337
        # onset_mean, onset_stdev: 0.37 0.0917605579756357
        # Image: 7
        # duration_mean, duration_stdev: 0.504 0.08619744775803978
        # onset_mean, onset_stdev: 0.49777777777777776 0.25073780019064623
    ## [ii] Clip gorilla neuron during-experiment screening responses ##
    clipgorilla_neuron_screening_statistics = build_clipgorilla_neuron_screening_statistics(aggregate_neuron_trials)# {'baseline': {'baseline_firing_rate_mean': 0.83, 'baseline_firing_rate_mean_stdev': 1.8429359862827126}, 'image_response': {8: {'mean_firing_rate': 3.5, 'stdev_firing_rate': 1.8829377433825436}}}
    mat_file_name = 'times_NSX17.mat'
    neuron_cluster_number = 1.0
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    durations = [max([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1])-min([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1]) if (len([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1])>1) else None for trial in list(neuron_trials_specific_neuron[8].keys())]
    duration_mean, duration_stdev = mean([i for i in durations if i!=None]), stdev([i for i in durations if i!=None])#(0.25272727272727274, 0.20243966553474196)
    ## Onset ##
    onset_times = [min([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1]) if (len([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1])>0) else None for trial in list(neuron_trials_specific_neuron[8].keys())]
    onset_mean, onset_stdev = mean([i for i in onset_times if i!=None]), stdev([i for i in onset_times if i!=None])#(0.4490909090909091, 0.19227347947413578)
    ## [iii] Rumimation neuron video responses (all info in the file created above - 'aggregate_statistics_by_revision1.csv'), in the epoch column under "end" (we coded rumination as the "end" epoch); density = response firing rate ##
    return

######################################### End [1] Tables 1 + 2 (neuron counts + response properties) #########################################

######################################### [2] Figure 2 (screening raster plot + histogram) #########################################
# revision = 1
# image_map, image_trials_map = build_image_trials_map_and_image_map()
# aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map, revision)
# mat_file_name, neuron_cluster_number, neuron_trials_specific_neuron, image = 'times_NSX22.mat', 2.0, aggregate_neuron_trials[mat_file_name][neuron_cluster_number], 6

def build_raster_plots_and_histograms_for_figure2b(image_map, image_trials_map, aggregate_neuron_trials, revision):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision)
    mat_file_name = 'times_NSX22.mat'
    neuron_cluster_number = 2.0
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    for image in [5,6,2,4]:
        file_name_to_save = 'revision'+str(revision)+'/figure2_image'+str(image)+'_raster.svg'
        build_raster_plot_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save)
        file_name_to_save = 'revision'+str(revision)+'/figure2_image'+str(image)+'_histo.svg'
        build_histogram_as_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save)
    return
######################################### End [2] Figure 2 (screening raster plot + histogram) #########################################

######################################### [3] Figure 3 (clip raster plot + histogram) #########################################
# revision = 1
# timestamps, six_video_map = build_timestamps_sixvideomap()
def build_figure3_raster_plot_and_histogram(revision, timestamps, six_video_map):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision)
    #### SCALE AND COLOR CAN BE ADJUSTED INSIDE FUNCTION OR MANUALLY IN AN .SVG EDITOR ####
    mat_file_name = 'times_NSX22.mat'
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    neuron_cluster_number = 2
    file_name_to_save = 'revision'+str(revision)+'/figure3_raster.svg'
    build_raster_plot_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision, file_name_to_save)
    file_name_to_save = 'revision'+str(revision)+'/figure3_histo.svg'
    build_histogram_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision, file_name_to_save)
    return
######################################### End [3] Figure 3 (clip raster plot + histogram) #########################################

######################################### [4] Figure 4 (rumination neurons) #########################################
# revision = 1
# average_firing_rate_dictionary = build_full_average_firing_rate_dictionary(revision)

# [i] Figures 4a + 4b (same method as in [3])
def build_figures4a4b(revision, average_firing_rate_dictionary):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    # [4a]
    mat_file_name = 'times_NSX45.mat'
    neuron_cluster_number = 1
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    build_raster_plot_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision, 'revision'+str(revision)+'/figure4a_raster.svg')
    build_histogram_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision, 'revision'+str(revision)+'/figure4a_histo.svg')
    # [4b]
    mat_file_name = 'times_NSX65.mat'
    neuron_cluster_number = 1
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    build_raster_plot_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision, 'revision'+str(revision)+'/figure4b_raster.svg')
    build_histogram_as_vector_video(neuron_video_trials, mat_file_name, neuron_cluster_number, timestamps, revision, 'revision'+str(revision)+'/figure4b_histo.svg')
    return

# [ii] Figure 4e
def build_figure4e(average_firing_rate_dictionary, revision):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    #### n=69 is the length of statistically_significant_neurons ####
    #### The test statistics hard coded below are printed (e.g., in the terminal) by the function build_statistical_significance ####
    statistically_significant_neurons = build_statistically_significant_response_end_of_video_programmatic(average_firing_rate_dictionary, stimuli_interval=[33,36],revision=1)#
    build_statistical_significance(statistically_significant_neurons, average_firing_rate_dictionary, epoch_interval_1=[33,36], epoch_interval_2=[8,31], unaware_trials=[1,2], aware_trials=[3,4,5], file_name_to_save='revision'+str(revision)+'/figure4e.svg', revision=1)
    # Ttest_1sampResult(statistic=4.147273335758336, pvalue=5.869268150978946e-05)
    # Ttest_indResult(statistic=3.7824747039100273, pvalue=0.000231797154790198)
    return
######################################### End [4] Figure 4 (rumination neurons) #########################################

######################################### [5] Figure 5 (clip-gorilla neuron during screening) #########################################
# revision = 1
# image_map, image_trials_map = build_image_trials_map_and_image_map()
# aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map, revision)
# mat_file_name, neuron_cluster_number, neuron_trials_specific_neuron, image = 'times_NSX22.mat', 2.0, aggregate_neuron_trials[mat_file_name][neuron_cluster_number], 6
def build_figure5b(image_map, image_trials_map, aggregate_neuron_trials, revision):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    #### Colors were manually adjusted (i.e., the bright yellow) ####
    mat_file_name = 'times_NSX17.mat'
    neuron_cluster_number = 1.0
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    for image in [8]:
        file_name_to_save = 'revision'+str(revision)+'/figure5b_raster.svg'
        build_raster_plot_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save)
        file_name_to_save = 'revision'+str(revision)+'/figure5b_histo.svg'
        build_histogram_as_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save)
    return

def build_figure5c(image_map, image_trials_map, aggregate_neuron_trials, revision):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    #### Colors were manually adjusted (i.e., the bright yellow) ####
    #### Note: The dog image's histogram (image 3) is cut off as the function sets the y-axis to 12 for all histograms for easy comparison. This can be changed by changing "axs.set_ylim((0,12))" to "axs.set_ylim((0,40)), for example" ####
    mat_file_name = 'times_NSX22.mat'
    neuron_cluster_number = 2.0
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    for image in [3,5,6]:
        file_name_to_save = 'revision'+str(revision)+'/figure5c_image'+str(image)+'_raster.svg'
        build_raster_plot_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save)
        file_name_to_save = 'revision'+str(revision)+'/figure5c_image'+str(image)+'_histo.svg'
        build_histogram_as_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save)
    return

# revision = 1
# aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map, revision)
def build_figure_5d(revision, channel_neuron_images):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    channel_neuron_images = {'17':{'neuron':1, 'images':[8]}, '22':{'neuron':2, 'images':[3,5,6]}}
    for channel in list(channel_neuron_images.keys()):
        neuron = channel_neuron_images[channel]['neuron']
        for image in channel_neuron_images[channel]['images']:
            print(channel, neuron, image)
            data_points, relevant_onsets = build_data_points(aggregate_neuron_trials, mat_file_name='times_NSX'+str(channel)+'.mat', neuron_cluster_number=int(neuron), image=int(image))
            print(relevant_onsets)
            file_name_to_save = 'revision'+str(revision)+'/figure5_channel_'+str(channel)+'image'+str(image)+'.svg'
            build_bar_charts_from_onsets(relevant_onsets, channel, neuron, image, file_name_to_save)
    return
######################################### End [5] Figure 5 (clip-gorilla neuron during screening) #########################################

######################################### [6] Figure 6 (clip-gorilla neuron firing similarity to gorilla neuron) #########################################
# revision = 1
# timestamps, six_video_map = build_timestamps_sixvideomap()

# [i] Figure 6a
def build_figure6a():
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    #### Axis/labels were added manually ####
    mat_file_name = 'times_NSX17.mat'
    neuron_cluster_number = 1
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    neuron_video_trials_17_neuron_1 = neuron_video_trials[neuron_cluster_number]
    mat_file_name = 'times_NSX22.mat'
    neuron_cluster_number = 2
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    neuron_video_trials_22_neuron_2 = neuron_video_trials[neuron_cluster_number]
    file_name_to_save = 'revision'+str(revision)+'/figure6a.svg'
    build_raster_plot_as_vector_video_17_22(neuron_video_trials_17_neuron_1, neuron_video_trials_22_neuron_2, timestamps, file_name_to_save)
    return

# [ii] Figure 6b
def build_figure6b():
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    #### Highlighting the color of the matching dyad was done normally (one can check that it matches via the average_spike_distance below) ####
    neural_data_cluster_17_neuron_1 = map_electrode_to_videos('times_NSX17.mat', six_video_map, revision)[1]
    neural_data_cluster_22_neuron_2 = map_electrode_to_videos('times_NSX22.mat', six_video_map, revision)[2]
    main_neuron_data_eg_22 = neural_data_cluster_22_neuron_2
    # Note that the below function no longer normalizes (the name is only a relic of a previous version)
    average_spike_distance = build_average_spike_distance_normalized(main_neuron_data_eg_22=neural_data_cluster_22_neuron_2, comparison_neuron_data_eg_17=neural_data_cluster_17_neuron_1, start_time_seconds=19.0, end_time_seconds=29.0)
    all_average_spike_distance = build_all_average_spike_distance_normalized(main_neuron_data_eg_22, six_video_map, start_time_seconds=19.0, end_time_seconds=29.0, revision=revision)#52
    plot_average_spike_distance_normalized(all_average_spike_distance, file_name_to_save='revision'+str(revision)+'/figure6b.svg')
    spike_range_statistics = build_spike_range_gorilla_on_screen(main_neuron_data_eg_22, six_video_map, start_time_seconds=19.0, end_time_seconds=29.0, revision=revision)
    return
######################################### End [6] Figure 6 (clip-gorilla neuron firing similarity to gorilla neuron) #########################################

######################################### [7] Figure 8 (spike-field coherence) #########################################
# revision = 1
# image_map, image_trials_map = build_image_trials_map_and_image_map()

# [i] Figures 8a + 8b
def build_figures8a(image_trials_map, revision):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    ## a-screening
    file_name_to_save = 'revision'+str(revision)+'/figure8a_screening.svg'
    build_circular_histogram_screening_all_trials(image_trials_map, channel=22, neuron=2, relevant_image=6, relevant_trials=None, wave='alpha', file_name_to_save=file_name_to_save, revision=revision)
    ## a-clip
    file_name_to_save = 'revision'+str(revision)+'/figure8a_clip.svg'
    build_circular_histogram_video_all_trials(channel=22, neuron=2, relevant_trials=[3,4,5], wave='alpha', event='gorilla_video_10s_on_screen', file_name_to_save=file_name_to_save, revision=revision)
    return

def build_figures8b(image_trials_map, revision):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    ## b-screening
    file_name_to_save = 'revision'+str(revision)+'/figure8b_screening.svg'
    build_circular_histogram_screening_all_trials(image_trials_map, channel=22, neuron=2, relevant_image=6, relevant_trials=None, wave='theta', file_name_to_save=file_name_to_save, revision=revision)
    ## b-clip
    file_name_to_save = 'revision'+str(revision)+'/figure8b_clip.svg'
    build_circular_histogram_video_all_trials(channel=22, neuron=2, relevant_trials=[3,4,5], wave='theta', event='gorilla_video_10s_on_screen', file_name_to_save=file_name_to_save, revision=revision)
    return

# [ii] Data for Rayleigh + KW CoM analyses
def build_raw_data_for_figure8d_analyses():
    #spike_degrees_screening_alpha, spike_degree_buckets_screening_alpha = build_spike_degree_buckets_screening(image_trials_map, channel=22, neuron=2, relevant_image=6, relevant_trials=None, wave='alpha', revision=1)
    #spike_degrees_screening_theta, spike_degree_buckets_screening_theta = build_spike_degree_buckets_screening(image_trials_map, channel=22, neuron=2, relevant_image=6, relevant_trials=None, wave='theta', revision=1)
    #spike_degrees_video_alpha, spike_degrees_buckets_video_alpha = build_spike_degree_buckets_video(channel=22, neuron=2, relevant_trials=[3,4,5], wave='alpha', event='gorilla_video_10s_on_screen', revision=None)
    #spike_degrees_video_theta, spike_degrees_buckets_video_theta = build_spike_degree_buckets_video(channel=22, neuron=2, relevant_trials=[3,4,5], wave='theta', event='gorilla_video_10s_on_screen', revision=None)
    # clip-gorilla gorilla epoch alpha band
    spike_degrees_video_alpha_unaware, spike_degrees_buckets_video_alpha_unaware = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='alpha', event='gorilla_video_10s_on_screen', revision=1)
    spike_degrees_video_alpha_aware, spike_degrees_buckets_video_alpha_aware = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='alpha', event='gorilla_video_10s_on_screen', revision=1)
    # spike_degrees_buckets_video_alpha_unaware = [322, 311, 325]
    # spike_degrees_buckets_video_alpha_aware = [224, 260, 315, 318, 212, 298, 26, 51, 105, 204, 334, 10, 123, 212, 241, 18, 58, 60, 88, 112, 257, 288, 87, 135, 210, 230, 350, 225, 20, 59, 50, 82, 359, 189, 229, 338, 114, 82, 246, 203, 66, 23, 53, 341, 15, 83, 248, 134, 211, 248, 300, 6, 209, 0, 286, 310, 67, 252, 274]
    # clip-gorilla gorilla epoch theta band
    spike_degrees_video_theta_unaware, spike_degrees_buckets_video_theta_unaware = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='theta', event='gorilla_video_10s_on_screen', revision=1)
    spike_degrees_video_theta_aware, spike_degrees_buckets_video_theta_aware = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='theta', event='gorilla_video_10s_on_screen', revision=1)
    # spike_degrees_buckets_video_theta_unaware = [31, 68, 78]
    # spike_degrees_buckets_video_theta_aware = [2, 12, 164, 200, 45, 145, 287, 327, 6, 46, 34, 52, 97, 303, 337, 108, 126, 36, 164, 177, 59, 73, 320, 341, 212, 228, 203, 311, 296, 110, 0, 24, 230, 179, 204, 271, 222, 328, 51, 316, 106, 293, 308, 330, 318, 167, 287, 84, 180, 221, 212, 264, 71, 174, 211, 232, 336, 171, 290]
    return spike_degrees_buckets_video_alpha_unaware, spike_degrees_buckets_video_alpha_aware, spike_degrees_buckets_video_theta_unaware, spike_degrees_buckets_video_theta_aware

# Hard coded raw outputs for ease of replication of [ii] as it takes a long time to run (these are spike buckets that are then used to make CoM and run KW tests)
#### ALPHA
## Beginning
spike_degrees_video_alpha_unaware, spike_degrees_buckets_video_alpha_unaware_beginning = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='alpha', event='beginning_of_video_event', revision=1)
spike_degrees_video_alpha_aware, spike_degrees_buckets_video_alpha_aware_beginning = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='alpha', event='beginning_of_video_event', revision=1)
# spike_degrees_buckets_video_alpha_unaware_beginning = [273, 290, 345, 318, 94, 123]
# spike_degrees_buckets_video_alpha_aware_beginning = [256, 270, 301, 333, 233, 322, 195, 141, 284, 28, 80, 160, 252, 348, 356, 276, 306, 321]
## Random
spike_degrees_video_alpha_unaware, spike_degrees_buckets_video_alpha_unaware_random = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='alpha', event='random_video_selection', revision=1)
spike_degrees_video_alpha_aware, spike_degrees_buckets_video_alpha_aware_random = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='alpha', event='random_video_selection', revision=1)
# spike_degrees_buckets_video_alpha_unaware_random = [276, 336, 193]
# spike_degrees_buckets_video_alpha_aware_random = [71, 92, 210, 235, 276, 80, 95, 106, 12, 215]
## Gorilla
spike_degrees_video_alpha_unaware, spike_degrees_buckets_video_alpha_unaware_gorilla = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='alpha', event='gorilla_video_10s_on_screen', revision=1)
spike_degrees_video_alpha_aware, spike_degrees_buckets_video_alpha_aware_gorilla = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='alpha', event='gorilla_video_10s_on_screen', revision=1)
# spike_degrees_buckets_video_alpha_unaware_gorilla = [322, 311, 325]
# spike_degrees_buckets_video_alpha_aware_gorilla = [224, 260, 315, 318, 212, 298, 26, 51, 105, 204, 334, 10, 123, 212, 241, 18, 58, 60, 88, 112, 257, 288, 87, 135, 210, 230, 350, 225, 20, 59, 50, 82, 359, 189, 229, 338, 114, 82, 246, 203, 66, 23, 53, 341, 15, 83, 248, 134, 211, 248, 300, 6, 209, 0, 286, 310, 67, 252, 274]
## End
spike_degrees_video_alpha_unaware, spike_degrees_buckets_video_alpha_unaware_end = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='alpha', event='end_of_video_event', revision=1)
spike_degrees_video_alpha_aware, spike_degrees_buckets_video_alpha_aware_end = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='alpha', event='end_of_video_event', revision=1)
# spike_degrees_buckets_video_alpha_unaware_end = [298, 134, 213, 285, 312, 351, 142, 77, 195, 211, 37, 279, 99, 186]
# spike_degrees_buckets_video_alpha_aware_end = [359, 164, 279, 318, 320, 33, 357, 32, 164, 173, 38, 131, 139, 148, 154, 171, 199, 218, 73, 124, 162, 211, 228, 248, 288, 308, 334, 1, 34, 78, 100, 221, 267, 326, 39, 298, 10, 178, 313, 97, 287, 356, 197, 237, 112, 4, 28, 50, 223]

#### THETA
## Beginning
spike_degrees_video_theta_unaware, spike_degrees_buckets_video_theta_unaware_beginning = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='theta', event='beginning_of_video_event', revision=1)
spike_degrees_video_theta_aware, spike_degrees_buckets_video_theta_aware_beginning = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='theta', event='beginning_of_video_event', revision=1)
# spike_degrees_buckets_video_theta_unaware_beginning = [316, 326, 0, 164, 348, 358]
# spike_degrees_buckets_video_theta_aware_beginning = [259, 273, 303, 98, 83, 151, 272, 53, 116, 290, 309, 339, 330, 26, 30, 152, 282, 295]
## Random
spike_degrees_video_theta_unaware, spike_degrees_buckets_video_theta_unaware_random = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='theta', event='random_video_selection', revision=1)
spike_degrees_video_theta_aware, spike_degrees_buckets_video_theta_aware_random = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='theta', event='random_video_selection', revision=1)
# spike_degrees_buckets_video_theta_unaware_random = [9, 236, 140]
# spike_degrees_buckets_video_theta_aware_random = [148, 163, 183, 194, 212, 19, 342, 150, 235, 276]
## Gorilla
spike_degrees_video_theta_unaware, spike_degrees_buckets_video_theta_unaware_gorilla = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='theta', event='gorilla_video_10s_on_screen', revision=1)
spike_degrees_video_theta_aware, spike_degrees_buckets_video_theta_aware_gorilla = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='theta', event='gorilla_video_10s_on_screen', revision=1)
# spike_degrees_buckets_video_theta_unaware_gorilla = [31, 68, 78]
# spike_degrees_buckets_video_theta_aware_gorilla = [2, 12, 164, 200, 45, 145, 287, 327, 6, 46, 34, 52, 97, 303, 337, 108, 126, 36, 164, 177, 59, 73, 320, 341, 212, 228, 203, 311, 296, 110, 0, 24, 230, 179, 204, 271, 222, 328, 51, 316, 106, 293, 308, 330, 318, 167, 287, 84, 180, 221, 212, 264, 71, 174, 211, 232, 336, 171, 290]
## End
spike_degrees_video_theta_unaware, spike_degrees_buckets_video_theta_unaware_end = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[1,2], wave='theta', event='end_of_video_event', revision=1)
spike_degrees_video_theta_aware, spike_degrees_buckets_video_theta_aware_end = build_spike_degree_buckets_video(channel=17, neuron=1, relevant_trials=[3,4,5], wave='theta', event='end_of_video_event', revision=1)
# spike_degrees_buckets_video_theta_unaware_end = [20, 91, 193, 214, 222, 233, 292, 16, 73, 112, 150, 52, 134, 83]
# spike_degrees_buckets_video_theta_aware_end = [219, 243, 98, 350, 214, 92, 50, 33, 259, 265, 54, 104, 108, 113, 116, 125, 290, 296, 12, 29, 42, 58, 64, 71, 84, 91, 100, 109, 118, 129, 135, 166, 178, 193, 219, 327, 128, 171, 205, 255, 329, 355, 280, 327, 260, 281]

######################################### End [7] Figure 8 (spike-field coherence) #########################################

######################################### [8] Supplementary table 1 #########################################
def build_supplementary_table1(all_channel_neurons, timestamps, revision):
    #### This function builds 1) region, 2) spike count & 3) rumination (True/False); single/multi-unit from Matlab output ####
    import json
    # [1] Regions
    cluster_map = build_cluster_map(all_channel_neurons)
    # [2] Spike count
    spike_count_across_entire_experiment = build_spike_count_across_entire_experiment(timestamps, revision)
    # [3] Rumination
    csvdataRowsAggregateStatistics = readCSV('aggregate_statistics_by_revision1.csv')#aggregate_statistics_by_revision1 is the output of building tables 1 & 2
    rumination_neurons = [item for sublist in [json.loads(row[csvdataRowsAggregateStatistics[0].index('neurons')]) for row in [row for row in csvdataRowsAggregateStatistics[1:] if row[csvdataRowsAggregateStatistics[0].index('epoch')]=='end']] for item in sublist]
    # Aggregate
    supplementary_table1 = {i:{'region':cluster_map[i], 'spike_count':spike_count_across_entire_experiment[i], 'rumination':True if (i in rumination_neurons) else False} for i in list(cluster_map.keys())}
    return supplementary_table1
######################################### End [8] Supplementary table 1 #########################################

######################################### [9] Supplementary figure 2 (clip-gorilla neuron - all during-experiment screening) #########################################
# aggregate_neuron_trials = build_aggregate_neuron_trials(image_trials_map)
def build_supplementary_figure2(aggregate_neuron_trials):
    #### THIS FUNCTION WILL SAVE OUTPUTS AS .SVG IN A DIRECTORY NAMED "revision"+str(revision) ####
    revision = 1
    mat_file_name = 'times_NSX17.mat'
    neuron_cluster_number = 1
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    images = [1,2,3,4,5,6,7,8,9]
    for image in images:
        file_name_to_save_raster = 'revision'+str(revision)+'/supplementary-materials-figure2_image'+str(image)+'raster.svg'
        build_raster_plot_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save_raster)
        file_name_to_save_histo = 'revision'+str(revision)+'/supplementary-materials-figure2_image'+str(image)+'histo.svg'
        build_histogram_as_vector_screening(neuron_trials_specific_neuron, mat_file_name, neuron_cluster_number, image_map, image, revision, file_name_to_save_histo)
    return
######################################### [9] Supplementary figure 2 (clip-gorilla neuron - all during-experiment screening) #########################################

######################################### [10] Text analyses - gorilla neuron #########################################
def build_gorilla_neuron_text_analyses(timestamps, aggregate_neuron_trials, image_trials_map):
    #### [1] Baseline firing rate: Mean + standard deviation ####
    revision = 1
    mat_file_name, neuron_cluster_number = 'times_NSX22.mat', 2
    average_rounded, average_plus_two_std_rounded, standard_deviation = build_average_statistics_for_channel_cluster(mat_file_name, neuron_cluster_number, timestamps, revision)
    standard_deviation_rounded = round(standard_deviation,2)
    average_plus_three_std_rounded = round(average_rounded+(3*standard_deviation),2)
    #### [2] Total spikes ####
    spike_count_across_entire_experiment = build_spike_count_across_entire_experiment(timestamps, revision)
    total_spikes = spike_count_across_entire_experiment['cluster_22_neuron_2']#369
    #### [3] During-experiment screening change in firing rate ####
    ## [3a] Mean + baseline ##
    gorilla_neuron_screening_statistics = build_gorilla_neuron_screening_statistics(image_trials_map, aggregate_neuron_trials, revision)
    mean_during_experiment_screening, stdev_during_experiment_screening = gorilla_neuron_screening_statistics['image_response'][6]['mean_firing_rate'], gorilla_neuron_screening_statistics['image_response'][6]['stdev_firing_rate']#(1.8333333333333333, 1.1146408580454255)
    ## [3b] Wilcoxon rank-sum ##
    from statistics import mean, stdev, median
    import math
    mata_data = read_mat_file_revision1(mat_file_name, revision)
    cluster_class = mata_data['cluster_class']
    neurons = list(set([el[0] for el in cluster_class]))
    data_start_time = timestamps[0]/30000
    data_end_time = timestamps[len(timestamps)-1]/30000
    neuron_spikes = {neuron:[el[1]/1000 for el in cluster_class if el[0]==neuron] for neuron in neurons}
    neuron_spikes_cluster = neuron_spikes[neuron_cluster_number]
    neuron_spikes_cluster_within_start_end = [el for el in neuron_spikes_cluster if (el>=data_start_time) and (el<=data_end_time)]
    bin_length = 1
    data_start_time_full_second = math.floor(data_start_time)
    data_end_time_full_second = math.ceil(data_end_time)
    bins = [[i, i+1] for i in list(range(data_start_time_full_second,data_end_time_full_second))]
    histogram_index = {i:[neuron_spike for neuron_spike in neuron_spikes_cluster_within_start_end if (neuron_spike >= bins[i][0]) and (neuron_spike < bins[i][1])] for i in range(len(bins))}
    histogram = [len(el) for el in list(histogram_index.values())]
    sample1 = histogram# This is across the entire experiment
    gorilla_22_screening_response = build_gorilla_neuron_screening_statistics_screening_response(image_trials_map, aggregate_neuron_trials, revision)
    sample2 = gorilla_22_screening_response[6]# This is the response in the during-experiment screening to the same gorilla image as in the pre-experiment screening
    from scipy.stats import ranksums
    ranksums(sample1, sample2)#RanksumsResult(statistic=-5.1650364440302, pvalue=2.403920803585364e-07)
    #### [4] Clip change in firing rate (between unaware and aware) ####
    neuron_video_trials = map_electrode_to_videos(mat_file_name, six_video_map, revision)
    bins = [[i, i+1] for i in list(range(19,29))]
    neuron_spikes_unaware = neuron_video_trials[neuron_cluster_number][1]+neuron_video_trials[neuron_cluster_number][2]
    neuron_spikes_aware = neuron_video_trials[neuron_cluster_number][3]+neuron_video_trials[neuron_cluster_number][4] + neuron_video_trials[neuron_cluster_number][5]
    histogram_index_unaware = {i:[neuron_spike for neuron_spike in neuron_spikes_unaware if (neuron_spike >= bins[i][0]) and (neuron_spike < bins[i][1])] for i in range(len(bins))}
    histogram_index_aware = {i:[neuron_spike for neuron_spike in neuron_spikes_aware if (neuron_spike >= bins[i][0]) and (neuron_spike < bins[i][1])] for i in range(len(bins))}
    histogram_unaware = [len(el)/2 for el in list(histogram_index_unaware.values())]# Dividing by 2 because 2 unaware trials
    histogram_aware = [len(el)/3 for el in list(histogram_index_aware.values())]# Dividing by 3 because 3 aware trials
    ## Firing rate in response to gorilla-on-screen epoch ##
    mean_firing_rate_during_clip_gorilla_epoch, stdev_firing_rate_during_clip_gorilla_epoch = mean(histogram_aware), stdev(histogram_aware)
    ## Two tailed t-test (unaware --> aware) ##
    from scipy import stats
    stats.ttest_ind(histogram_unaware, histogram_aware)#Ttest_indResult(statistic=-2.905487990874559, pvalue=0.009432900712871317)
    degrees_of_freedom = len(histogram_unaware)+len(histogram_aware)-2#https://www.statsdirect.co.uk/help/parametric_methods/utt.htm
    return
######################################### End [10] Text analyses - gorilla neuron #########################################

######################################### [11] Text analyses - rumination neurons #########################################

def build_rumination_neurons_text_analyses():
    #### [1] Count ####
    statistically_significant_neurons = build_statistically_significant_response_end_of_video_programmatic(average_firing_rate_dictionary, stimuli_interval=[33,36],revision=1)#
    rumination_neuron_count = len(statistically_significant_neurons)#69
    #### [2] Increase in firing rate ####
    epoch_interval_1=[33,36]
    epoch_interval_2=[8,31]
    unaware_trials=[1,2]
    aware_trials=[3,4,5]
    ## [0a] Epoch analysis 1
    epoch_analysis_multiple_neurons_epoch_1 = build_epoch_analysis_multiple_neurons(statistically_significant_neurons, six_video_map, timestamps, epoch_interval_1, unaware_trials, aware_trials, revision)
    epoch_analysis_multiple_neurons_epoch_2 = build_epoch_analysis_multiple_neurons(statistically_significant_neurons, six_video_map, timestamps, epoch_interval_2, unaware_trials, aware_trials, revision)
    unaware_aware_delta_epoch_1 = {}
    for channel_neuron in list(epoch_analysis_multiple_neurons_epoch_1.keys()):
        # (average spikes per aware trial) distance from mean (std) - (average spikes per unaware trial) distance from mean (std)
        unaware_aware_delta_epoch_1[channel_neuron] = (((epoch_analysis_multiple_neurons_epoch_1[channel_neuron]['aware_spike_count']/len(aware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation']) - (((epoch_analysis_multiple_neurons_epoch_1[channel_neuron]['unaware_spike_count']/len(unaware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation'])
    ## [0b] Epoch analysis 2
    unaware_aware_delta_epoch_2 = {}
    for channel_neuron in list(epoch_analysis_multiple_neurons_epoch_2.keys()):
        unaware_aware_delta_epoch_2[channel_neuron] = (((epoch_analysis_multiple_neurons_epoch_2[channel_neuron]['aware_spike_count']/len(aware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation']) - (((epoch_analysis_multiple_neurons_epoch_2[channel_neuron]['unaware_spike_count']/len(unaware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation'])
    print(stats.ttest_1samp(list(unaware_aware_delta_epoch_1.values())+list(unaware_aware_delta_epoch_2.values()),0.0))#Ttest_1sampResult(statistic=4.147273335758336, pvalue=5.869268150978946e-05)
    degrees_of_freedom = len(list(unaware_aware_delta_epoch_1.values())+list(unaware_aware_delta_epoch_2.values()))-1#137
    #### [3] Increase in firing rate vs. control ####
    print(stats.ttest_ind(list(unaware_aware_delta_epoch_1.values()),list(unaware_aware_delta_epoch_2.values())))#Ttest_indResult(statistic=3.7824747039100273, pvalue=0.000231797154790198)
    degrees_of_freedom = len(list(unaware_aware_delta_epoch_1.values())) + len(list(unaware_aware_delta_epoch_2.values())) - 2#136
    #### [4] % that show increase in firing rate ####
    percentage_of_rumination_neurons_that_experience_increased_firing_rate = len([key for key in list(unaware_aware_delta_epoch_1.keys()) if unaware_aware_delta_epoch_1[key] > 0])/len(unaware_aware_delta_epoch_1)#0.7246376811594203
    #### [5] Mean change for those that increase ####
    from statistics import mean, stdev
    mean_percent_change, stdev_percent_change = mean([unaware_aware_delta_epoch_1[key] for key in list(unaware_aware_delta_epoch_1.keys()) if unaware_aware_delta_epoch_1[key] > 0]), stdev([unaware_aware_delta_epoch_1[key] for key in list(unaware_aware_delta_epoch_1.keys()) if unaware_aware_delta_epoch_1[key] > 0])#(2.6039981928743465, 1.8661273683546316)
    #### [6] % that exhibit beginning responses ####
    epoch_interval_1=[0,2]
    unaware_trials=[1,2]
    aware_trials=[3,4,5]
    epoch_analysis_multiple_neurons_epoch_1 = build_epoch_analysis_multiple_neurons(statistically_significant_neurons, six_video_map, timestamps, epoch_interval_1, unaware_trials, aware_trials, revision)
    unaware_aware_delta_epoch_1 = {}
    for channel_neuron in list(epoch_analysis_multiple_neurons_epoch_1.keys()):
        # (average spikes per aware trial) distance from mean (std) - (average spikes per unaware trial) distance from mean (std)
        unaware_aware_delta_epoch_1[channel_neuron] = (((epoch_analysis_multiple_neurons_epoch_1[channel_neuron]['aware_spike_count']/len(aware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation']) - (((epoch_analysis_multiple_neurons_epoch_1[channel_neuron]['unaware_spike_count']/len(unaware_trials))-(average_firing_rate_dictionary[channel_neuron]['average_firing_rate']))/average_firing_rate_dictionary[channel_neuron]['standard_deviation'])
    rumination_neurons_experiencing_above_3std_firing_rate_during_beginning = [neuron_cluster for neuron_cluster in list(unaware_aware_delta_epoch_1.keys()) if ((epoch_analysis_multiple_neurons_epoch_1[neuron_cluster]['unaware_spike_count']+epoch_analysis_multiple_neurons_epoch_1[neuron_cluster]['aware_spike_count'])/5)>=(average_firing_rate_dictionary[neuron_cluster]['average_firing_rate']+3*average_firing_rate_dictionary[neuron_cluster]['standard_deviation'])]#0
    percentage_of_rumination_neurons_that_experience_increased_firing_rate_during_beginning = len([key for key in list(unaware_aware_delta_epoch_1.keys()) if unaware_aware_delta_epoch_1[key] > 0])/len(unaware_aware_delta_epoch_1)#0.5507246376811594
    return
######################################### End [11] Text analyses - rumination neurons #########################################

######################################### [12] Text analyses - clip-gorilla neuron (contextual learning) #########################################
def build_clipgorilla_neuron_text_analyses():
    from statistics import mean, stdev
    #### [1] Baseline ####
    clipgorilla_neuron_screening_statistics = build_clipgorilla_neuron_screening_statistics(aggregate_neuron_trials)
    baseline_firing_rate_mean, baseline_firing_rate_mean_stdev = clipgorilla_neuron_screening_statistics['baseline']['baseline_firing_rate_mean'], clipgorilla_neuron_screening_statistics['baseline']['baseline_firing_rate_mean_stdev']#(0.83, 1.8429359862827126)
    #### [2] During-experiment screening ####
    ## Duration ##
    mat_file_name = 'times_NSX17.mat'
    neuron_cluster_number = 1.0
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    #reference_statistics_images = build_reference_statistics_images(mat_file_name, neuron_cluster_number, images_of_interest={i:image_map[i] for i in [1,2,3,4,5,6,7,8,9]}, revision=1)
    durations = [max([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1])-min([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1]) if (len([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1])>1) else None for trial in list(neuron_trials_specific_neuron[8].keys())]
    duration_mean, duration_stdev = mean([i for i in durations if i!=None]), stdev([i for i in durations if i!=None])#(0.25272727272727274, 0.20243966553474196)
    ## Onset ##
    onset_times = [min([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1]) if (len([i for i in neuron_trials_specific_neuron[8][trial] if i>=0.0 and i<1])>0) else None for trial in list(neuron_trials_specific_neuron[8].keys())]
    onset_mean, onset_stdev = mean([i for i in onset_times if i!=None]), stdev([i for i in onset_times if i!=None])#(0.4490909090909091, 0.19227347947413578)
    ## Increase above baseline ##
    screening_response_firing_rate_mean, screening_response_firing_rate_stdev = clipgorilla_neuron_screening_statistics['image_response'][8]['mean_firing_rate'], clipgorilla_neuron_screening_statistics['image_response'][8]['stdev_firing_rate']#(3.5, 1.8829377433825436)
    #### [3] Total spikes ####
    spike_count_across_entire_experiment = build_spike_count_across_entire_experiment(timestamps, revision)
    total_spikes = spike_count_across_entire_experiment['cluster_17_neuron_1']#2655
    #### [4] ANOVA (clip-gorilla neuron onset vs. gorilla neuron onset) ####
    ## Build clip-gorilla neuron onset times ####
    clipgorilla_onset_times = [i for i in onset_times if i!=None]
    ## Build gorilla neuron onset times ####
    mat_file_name = 'times_NSX22.mat'
    neuron_cluster_number = 2.0
    neuron_trials_specific_neuron = aggregate_neuron_trials[mat_file_name][neuron_cluster_number]
    onset_times = [min([i for i in neuron_trials_specific_neuron[5][trial] if i>=0.0 and i<1]) if (len([i for i in neuron_trials_specific_neuron[5][trial] if i>=0.0 and i<1])>0) else None for trial in list(neuron_trials_specific_neuron[5].keys())]# Compare against the same image (6) as used in both pre-experiment and during-experiment screening
    gorilla_onset_times = [i for i in onset_times if i!=None]
    ## Compare ##
    from scipy.stats import f_oneway
    f_oneway(clipgorilla_onset_times, gorilla_onset_times)#F_onewayResult(statistic=1.5159830155423808, pvalue=0.23251113781462554)
    from scipy import stats
    stats.ttest_ind(clipgorilla_onset_times, gorilla_onset_times)#Ttest_indResult(statistic=1.2312526205220364, pvalue=0.23251113781462568)
    from scipy.stats import bartlett
    bartlett(clipgorilla_onset_times, gorilla_onset_times)#BartlettResult(statistic=4.795782891398895, pvalue=0.028529487986759007)
    clipgorilla_onset_times = [0.33, 0.75, 0.65, 0.22, 0.55, 0.16, 0.36, 0.42, 0.37, 0.7, 0.43]
    gorilla_onset_times_image6 = [0.62, 0.31, 0.35, 0.31, 0.32, 0.34, 0.32, 0.34, 0.34, 0.37, 0.45]
    gorilla_onset_times_image_5 = [0.34, 0.34, 0.35, 0.39, 0.68, 0.37, 0.55, 0.36, 0.38, 0.4]
    gorilla_onset_times_image_7 = [0.39, 0.31, 0.3, 0.97, 0.3, 0.38, 0.35, 0.77, 0.71]
    bartlett(clipgorilla_onset_times, gorilla_onset_times_image6, gorilla_onset_times_image_5, gorilla_onset_times_image_7)#BartlettResult(statistic=10.7740772789502, pvalue=0.013012408709542895)
    bartlett_degrees_of_freedom = len([clipgorilla_onset_times, gorilla_onset_times_image6, gorilla_onset_times_image_5, gorilla_onset_times_image_7])-1#3, https://docs.tibco.com/pub/enterprise-runtime-for-R/4.1.0/doc/html/Language_Reference/stats/bartlett.test.html#:~:text=The%20Bartlett%20K%2Dsquared%20test,(statistic%2C%20parameter%2C%20lower.
    return

######################################### End [12] Text analyses - clip-gorilla neuron (contextual learning) #########################################

######################################### [13] Create video + figure 8c (still from video) #########################################
def build_circular_histogram_video_all_trials_mov_frames(six_video_map, path_directory_to_save, lfp_filtered, neuron_spikes_in_seconds, channel=17, neuron=1, relevant_trials=None, wave='alpha', interval_start_seconds=0, interval_stop_seconds=3):
    import numpy as np
    #lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    #if int(channel) not in [13, 16, 61, 67, 69, 70, 72, 78]
    video_trials = list(six_video_map.keys())
    if relevant_trials == None:
        relevant_trials = video_trials
    #events = {'gorilla_video_10s_on_screen':{'graph_title':'Gorilla on-screen video', 'start_from_video_start_seconds':19, 'end_from_video_start_seconds':29}, 'end_of_video_event':{'graph_title':'end_of_video_event', 'start_from_video_start_seconds':33, 'end_from_video_start_seconds':36}, 'random_video_selection':{'graph_title':'random_video_selection', 'start_from_video_start_seconds':3, 'end_from_video_start_seconds':18}, 'beginning_of_video_event':{'graph_title':'beginning_of_video_event', 'start_from_video_start_seconds':0, 'end_from_video_start_seconds':3}}
    #event = 'gorilla_video_10s_on_screen'
    aggregate_spike_degrees = []
    for video_trial in video_trials:
        #print('beginning video trial: ' + str(video_trial))
        if video_trial in relevant_trials:
            #import numpy as np
            #video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0], six_video_map[video_trial]['timestamp_range_seconds'][1]
            video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0]+interval_start_seconds, six_video_map[video_trial]['timestamp_range_seconds'][0]+interval_stop_seconds
            sr = 30000
            start_time_sr = int(video_start_time_seconds*sr)
            end_time_sr = int(video_end_time_seconds*sr)
            lfp_wave_y = lfp_filtered[start_time_sr:end_time_sr]
            lfp_wave_x = np.linspace(video_start_time_seconds,video_end_time_seconds,len(lfp_wave_y))
            #
            #filtered_lfp_file_name = 'lfp'+str(channel)
            #neuron_spikes_mat_file_name = 'timesNSX_'+str(int(filtered_lfp_file_name[len('lfp'):]))+'.mat'#'timesNSX_22.mat'
            #neuron = int(neuron)
            #neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron)
            #
            spike_degrees = build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds)
            #print(spike_degrees)
            #spike_degrees_movies[video_trial] = spike_degrees
            aggregate_spike_degrees += spike_degrees
    spike_degree_buckets = aggregate_spike_degrees
    file_name_to_save = path_directory_to_save + '/interval_start_seconds_'+str(round(float(interval_start_seconds),5))+'.png'
    #if relevant_trials == video_trials:
    #    file_name_to_save = 'figures/raw_data/spike-field-coherence/movie/'+wave+'/'+'cluster'+str(channel)+'_neuron'+str(neuron)+'_'+events[event]['graph_title'].replace(' ','_')
    #else:
    #    file_name_to_save = 'figures/raw_data/spike-field-coherence/movie/'+wave+'/'+'cluster'+str(channel)+'_neuron'+str(neuron) +'_'+events[event]['graph_title'].replace(' ','_')+ '_trial_'+str(min(relevant_trials))+'_to_'+str(max(relevant_trials))
    #graph_title = '6 video trials - '+wave +' wave'
    #graph_title = events[event]['graph_title']
    print(file_name_to_save)
    #build_circular_histogram(spike_degree_buckets, graph_title, file_name_to_save, file_format='svg', max_yaxis=3)
    graph_title = ''
    build_circular_histogram(spike_degree_buckets, graph_title, file_name_to_save, file_format='png')
    return

def build_movie_images(wave='alpha', window_size=3, fps=10, channel=17, neuron=1, revision=None):
    # Read in LFP data
    lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    try:
        lfp_filtered = read_mat_file_lfp_filter(lfp_filtered_file_name)
    except:
        lfp_filtered = read_mat_file_lfp_filter_73(lfp_filtered_file_name)
    # Read in spike data
    neuron = int(neuron)
    neuron_spikes_mat_file_name = 'timesNSX_'+str(channel)+'.mat'#'timesNSX_22.mat'
    neuron_spikes_mat_file_name = convert_mat_filename_to_revision(neuron_spikes_mat_file_name, revision)
    neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron, revision)
    # Create interval map
    interval_starts = np.linspace(0,35.6,int(35.6*fps))
    intervals = [[i,i+window_size] for i in interval_starts if (i <= interval_starts[-1]-window_size)]
    # For each interval, save images
    path_directory_to_save = 'figures/raw_data/spike-field-coherence/created-movies/channel17_alpha'
    for interval in intervals:
        build_circular_histogram_video_all_trials_mov_frames(six_video_map, path_directory_to_save, lfp_filtered, neuron_spikes_in_seconds, channel=int(channel), neuron=int(neuron), relevant_trials=None, wave=wave, interval_start_seconds=interval[0], interval_stop_seconds=interval[1])
    return

def sort_tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j]= tup[j + 1]
                tup[j + 1]= temp
    return tup

def build_video_format(image_folder='figures/raw_data/spike-field-coherence/created-movies/channel17_alpha', video_name='figures/movies/channel17_gorilla_movie.avi'):
    # pip3 install opencv-python, https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/lessons/timelapse-how-to.py
    import cv2
    import os
    #
    #image_folder = 'figures/raw_data/spike-field-coherence/created-movies/channel17_alpha'
    #video_name = 'test_video.avi'
    #
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Make sure to sort images
    #images_sorted = sort_tuple([(el[:len('interval_start_seconds_')],el[len('interval_start_seconds_'):el.index('.png')]) for el in images])
    image_tuples_sorted = sort_tuple([(el,float(el[len('interval_start_seconds_'):el.index('.png')])) for el in images])
    images_sorted = [el[0] for el in image_tuples_sorted]
    frame = cv2.imread(os.path.join(image_folder, images_sorted[0]))
    height, width, layers = frame.shape
    #
    #video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    video = cv2.VideoWriter(video_name, fourcc=0, fps=10, frameSize=(width,height))
    #
    for image in images_sorted:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    #
    cv2.destroyAllWindows()
    video.release()
    return

def build_average_angle(spike_degree_buckets):
    import numpy as np
    degrees_in_radians = np.radians(spike_degree_buckets)
    x,y = 0,0
    for angle in degrees_in_radians:
        x+= np.cos(angle)
        y+= np.sin(angle)
    average_angle_radians = np.arctan2(y,x)
    average_angle_degrees = np.degrees(average_angle_radians)
    # Can be negatives --> turn to positive (e.g., [332] --> -28.0 --> )
    average_angle = average_angle_degrees if (average_angle_degrees >= 0) else (360+average_angle_degrees)
    return average_angle

def build_circular_histogram_y_single_interval(six_video_map, path_directory_to_save, lfp_filtered, neuron_spikes_in_seconds, channel=17, neuron=1, relevant_trials=None, wave='alpha', interval_start_seconds=0, interval_stop_seconds=3):
    import numpy as np
    video_trials = list(six_video_map.keys())
    if relevant_trials == None:
        relevant_trials = video_trials
    aggregate_spike_degrees = []
    for video_trial in video_trials:
        if video_trial in relevant_trials:
            video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0]+interval_start_seconds, six_video_map[video_trial]['timestamp_range_seconds'][0]+interval_stop_seconds
            sr = 30000
            start_time_sr = int(video_start_time_seconds*sr)
            end_time_sr = int(video_end_time_seconds*sr)
            lfp_wave_y = lfp_filtered[start_time_sr:end_time_sr]
            lfp_wave_x = np.linspace(video_start_time_seconds,video_end_time_seconds,len(lfp_wave_y))
            spike_degrees = build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds)
            aggregate_spike_degrees += spike_degrees
    spike_degree_buckets = aggregate_spike_degrees
    # y = mean (but not arithmetic) - need to average the angles
    y = build_average_angle(spike_degree_buckets)
    return y

# Build a graph - x_axis = time (0 to 35.6s), y_axis = degrees (0 to 360) [need to create total plot before]
def build_scatter_plot_data(six_video_map, wave='alpha', window_size=3, fps=10, channel=17, neuron=1, revision=None):
    import numpy as np
    # Read in LFP data
    lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    try:
        lfp_filtered = read_mat_file_lfp_filter(lfp_filtered_file_name)
    except:
        lfp_filtered = read_mat_file_lfp_filter_73(lfp_filtered_file_name)
    # Read in spike data
    neuron = int(neuron)
    neuron_spikes_mat_file_name = 'timesNSX_'+str(channel)+'.mat'#'timesNSX_22.mat'
    neuron_spikes_mat_file_name = convert_mat_filename_to_revision(neuron_spikes_mat_file_name, revision)
    neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron, revision)
    # Create interval map
    interval_starts = np.linspace(0,35.6,int(35.6*fps))
    intervals = [[i,i+window_size] for i in interval_starts if (i <= interval_starts[-1]-window_size)]
    # For each interval, save images
    path_directory_to_save = 'figures/raw_data/spike-field-coherence/created-movies/channel17_alpha'
    aggregate_x, aggregate_y, aggregate_max_yaxis = [], [], 0
    for interval in intervals:
        print(str(intervals.index(interval)) + ' of ' + str(len(intervals)))
        x = interval[0]
        y = build_circular_histogram_y_single_interval(six_video_map, path_directory_to_save, lfp_filtered, neuron_spikes_in_seconds, channel=int(channel), neuron=int(neuron), relevant_trials=None, wave=wave, interval_start_seconds=interval[0], interval_stop_seconds=interval[1])
        print(x,y)
        aggregate_x.append(x)
        aggregate_y.append(y)
        spike_degree_buckets = build_spike_degree_buckets(six_video_map, lfp_filtered, neuron_spikes_in_seconds, channel=int(channel), neuron=int(neuron), relevant_trials=None, wave=wave, interval_start_seconds=interval[0], interval_stop_seconds=interval[1])
        max_yaxis = build_max_yaxis(spike_degree_buckets)
        aggregate_max_yaxis = aggregate_max_yaxis if (aggregate_max_yaxis>=max_yaxis) else max_yaxis
    return aggregate_x, aggregate_y, aggregate_max_yaxis

# (six_video_map, path_directory_to_save, lfp_filtered, neuron_spikes_in_seconds, channel=17, neuron=1, relevant_trials=None, wave='alpha', interval_start_seconds=0, interval_stop_seconds=3)
def build_scatter_plot_video_stills(part_of_video_title, current_x, aggregate_x, aggregate_y, path_directory_to_save, lfp_filtered, spike_degree_buckets, file_format='png', max_yaxis=None, interval_start_seconds=0):
    import matplotlib.pyplot as plt
    import numpy as np
    from statistics import mean
    #fig, axs = plt.subplots(1, 2)
    #fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    fig = plt.figure()
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122, projection='polar')
    fig.set_figwidth(15,15)# Not sure what this is doing exactly but looks ok today... ("good enough" --> stability)
    fig.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle(part_of_video_title, fontsize=30)
    #### SCATTER PLOT ####
    ax0.scatter(aggregate_x, aggregate_y)
    # current x
    #current_x = 19.3
    ax0.axvline(current_x, color='black', linestyle='dashed', linewidth=1)
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Center of mass (degrees)')
    #### END OF SCATTER PLOT ####
    #### POLAR COORDINATES ####
    N = 20
    aggregate_bins = [[i*18,i*18+18] for i in list(range(0,N))]
    spike_degree_buckets_aggregate = [[spike for spike in spike_degree_buckets if (spike>=bin[0]) and (spike<bin[1])] for bin in aggregate_bins]
    radii = [len(el) for el in spike_degree_buckets_aggregate]
    width = (2*np.pi) / N
    bottom = 0
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    #ax1 = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)# UNSURE IF THIS WILL WORK
    if max_yaxis!=None:
        ax1.set_ylim([0,max_yaxis])
    bars = ax1.bar(theta, radii, width=width, bottom=bottom)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.5)
    # Average red line
    degrees_in_radians = np.radians(spike_degree_buckets)
    x_radians,y_radians = 0,0
    for angle in degrees_in_radians:
        x_radians+= np.cos(angle)
        y_radians+= np.sin(angle)
    average_angle_radians = np.arctan2(y_radians,x_radians)
    average_angle_degrees = np.degrees(average_angle_radians)
    radii_average = [max(radii)]
    theta_average = [(average_angle_degrees/360)*(2*np.pi)]
    ax1.plot((0,theta_average[0]),(0,radii_average[0]), color='r', linewidth=4)
    #### END OF POLAR COORDINATES ####
    file_name_to_save = path_directory_to_save + '/interval_start_seconds_'+str(round(float(interval_start_seconds),5))+'.'+file_format
    #file_name_to_save = path_directory_to_save + '/interval_start_seconds_'+str(round(float(interval_start_seconds),5))+'.png'
    #file_name_to_save = 'figure_8_still.svg'
    #file_name_to_save = 'test_video_scatter2.' + file_format
    fig.savefig(file_name_to_save)
    plt.close()
    return

#build_circular_histogram_video_all_trials_mov_frames(six_video_map, path_directory_to_save, lfp_filtered, neuron_spikes_in_seconds, channel=17, neuron=1, relevant_trials=None, wave='alpha', interval_start_seconds=0, interval_stop_seconds=3)
def build_spike_degree_buckets(six_video_map, lfp_filtered, neuron_spikes_in_seconds, channel=17, neuron=1, relevant_trials=None, wave='alpha', interval_start_seconds=0, interval_stop_seconds=3):
    import numpy as np
    video_trials = list(six_video_map.keys())
    if relevant_trials == None:
        relevant_trials = video_trials
    aggregate_spike_degrees = []
    for video_trial in video_trials:
        if video_trial in relevant_trials:
            video_start_time_seconds, video_end_time_seconds = six_video_map[video_trial]['timestamp_range_seconds'][0]+interval_start_seconds, six_video_map[video_trial]['timestamp_range_seconds'][0]+interval_stop_seconds
            sr = 30000
            start_time_sr = int(video_start_time_seconds*sr)
            end_time_sr = int(video_end_time_seconds*sr)
            lfp_wave_y = lfp_filtered[start_time_sr:end_time_sr]
            lfp_wave_x = np.linspace(video_start_time_seconds,video_end_time_seconds,len(lfp_wave_y))
            spike_degrees = build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds)
            aggregate_spike_degrees += spike_degrees
    spike_degree_buckets = aggregate_spike_degrees
    return spike_degree_buckets

def build_wave_360_degree_buckets(lfp_wave_x, lfp_wave_y, neuron_spikes_in_seconds):
    wave_start_time_seconds, wave_end_time_seconds = int(lfp_wave_x[0]), int(lfp_wave_x[len(lfp_wave_x)-1])
    # [0] Identify peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(lfp_wave_y, height=None)
    # [1] Form peak intervals [e.g., peak 1 --> [0.2s,0.3s]] #ONLY COUNT THOSE BETWEEN PEAKS, NOT START/END (false positives)
    peak_intervals_indexes = [[peaks[i], peaks[i+1]] for i in list(range(len(peaks)-1))]
    # [2] Divide each peak interval into n=360 degrees - create a map for {peak_interval_start_1: degree, ..., peak_interval_start_n:map}
    # Want: {wave#: {'start_time_seconds':, 'end_time_seconds':, degrees:{i:[degree_start, degree_end]}}}
    import numpy as np
    #peak_interval_degree_map_indexes = {peak_interval[0]:[[np.linspace(peak_interval[0], peak_interval[1], 361)[i], np.linspace(peak_interval[0], peak_interval[1], 361)[i+1]] for i in list(range(360))] for peak_interval in peak_intervals_indexes}
    #peak_interval_degree_map_seconds = {peak_interval_start: for peak_interval_start in list(peak_interval_degree_map_indexes.keys())}
    wave_dictionary = {}
    for i in list(range(len(peak_intervals_indexes))):
        wave_dictionary[i] = {'lfp_index_start':peak_intervals_indexes[i][0], 'lfp_index_end':peak_intervals_indexes[i][1]}
        wave_dictionary[i]['start_time_seconds_53m_experiment'] = lfp_wave_x[peak_intervals_indexes[i][0]]
        wave_dictionary[i]['end_time_seconds_53m_experiment'] = lfp_wave_x[peak_intervals_indexes[i][1]]
        degrees_in_seconds = np.linspace(lfp_wave_x[peak_intervals_indexes[i][0]], lfp_wave_x[peak_intervals_indexes[i][1]], 361)
        degrees_in_seconds_intervals = [[degrees_in_seconds[j], degrees_in_seconds[j+1]] for j in list(range(len(degrees_in_seconds)-1))]
        wave_dictionary[i]['degrees_to_second_intervals'] = {j:degrees_in_seconds_intervals[j] for j in list(range(len(degrees_in_seconds_intervals)))}
        # [[np.linspace(peak_interval[0], peak_interval[1], 361)[i], np.linspace(peak_interval[0], peak_interval[1], 361)[i+1]] for i in list(range(360))]
    # [3] For each spike, determine which degree it falls into
    spike_occurrences_x = [spike for spike in neuron_spikes_in_seconds if (spike >= wave_start_time_seconds) and (spike < wave_end_time_seconds)]
    spike_degrees = []
    for spike in spike_occurrences_x:
        for wave_index in list(wave_dictionary.keys()):
            if (spike >= wave_dictionary[wave_index]['start_time_seconds_53m_experiment']) and (spike < wave_dictionary[wave_index]['end_time_seconds_53m_experiment']):
                for degree in list(wave_dictionary[wave_index]['degrees_to_second_intervals'].keys()):
                    if (spike >= wave_dictionary[wave_index]['degrees_to_second_intervals'][degree][0]) and (spike < wave_dictionary[wave_index]['degrees_to_second_intervals'][degree][1]):
                        spike_degrees.append(degree)
    return spike_degrees

def build_max_yaxis(spike_degree_buckets):
    N = 20
    aggregate_bins = [[i*18,i*18+18] for i in list(range(0,N))]
    spike_degree_buckets_aggregate = [[spike for spike in spike_degree_buckets if (spike>=bin[0]) and (spike<bin[1])] for bin in aggregate_bins]
    max_yaxis = max([len(el) for el in spike_degree_buckets_aggregate]) if len(spike_degree_buckets_aggregate)>0 else 0
    return max_yaxis

def build_video_stills_cartesian_and_polar(six_video_map, wave='alpha', window_size=3, fps=10, channel=17, neuron=1, revision=None):
    import os
    import numpy as np
    # Read in LFP data
    lfp_filtered_file_name = 'lfp'+str(channel)+'_'+wave+'.mat'
    try:
        lfp_filtered = read_mat_file_lfp_filter(lfp_filtered_file_name)
    except:
        lfp_filtered = read_mat_file_lfp_filter_73(lfp_filtered_file_name)
    # Read in spike data
    neuron = int(neuron)
    neuron_spikes_mat_file_name = 'timesNSX_'+str(channel)+'.mat'#'timesNSX_22.mat'
    neuron_spikes_mat_file_name = convert_mat_filename_to_revision(neuron_spikes_mat_file_name, revision)
    neuron_spikes_in_seconds = map_neuron_spikes_in_seconds(neuron_spikes_mat_file_name, neuron, revision)
    # Build aggregate_x, aggregate_y to graph cartesian coordinates
    aggregate_x, aggregate_y, aggregate_max_yaxis = build_scatter_plot_data(six_video_map, wave=wave, window_size=window_size, fps=fps, channel=int(channel), neuron=int(neuron), revision=revision)
    # Create interval map
    interval_starts = np.linspace(0,35.6,int(35.6*fps))
    intervals = [[i,i+window_size] for i in interval_starts if (i <= interval_starts[-1]-window_size)]
    #path_directory_to_save = 'figures/raw_data/spike-field-coherence/created-movies/channel'+str(channel)+'_'+wave+'_cartesian_and_polar'
    path_directory_to_save = 'figures/raw_data/spike-field-coherence-2021/created-movies/channel'+str(channel)+'_'+wave+'_cartesian_and_polar'
    if not os.path.exists(path_directory_to_save):
        os.makedirs(path_directory_to_save)
    parts_of_video_seconds = {'Count how many times the players wearing white pass the basketball.':[0,7], 'Players play basketball...':[7,18], 'Gorilla enters':[18,20], 'Gorilla on screen':[20,28], 'Gorilla exits':[28,30], 'More basketball...':[30,32], 'How many passes did you count?':[32,36]}
    # For each interval, save images
    for interval in intervals:
        # Build spike data for polar coordinates
        spike_degree_buckets = build_spike_degree_buckets(six_video_map, lfp_filtered, neuron_spikes_in_seconds, channel=int(channel), neuron=int(neuron), relevant_trials=None, wave=wave, interval_start_seconds=interval[0], interval_stop_seconds=interval[1])
        # Actually graph
        #current_x = interval[0]+(0.5*window_size)#Today current_x is the midway point - so if window_size=3 --> starts at 1.5s
        current_x = interval[0]
        part_of_video_title = [key for key in list(parts_of_video_seconds.keys()) if (current_x >= parts_of_video_seconds[key][0]) and (current_x < parts_of_video_seconds[key][1])][0]
        build_scatter_plot_video_stills(part_of_video_title, current_x, aggregate_x, aggregate_y, path_directory_to_save, lfp_filtered, spike_degree_buckets, file_format='png', max_yaxis=aggregate_max_yaxis, interval_start_seconds=interval[0])
        #build_circular_histogram_video_all_trials_mov_frames(six_video_map, path_directory_to_save, lfp_filtered, neuron_spikes_in_seconds, channel=int(channel), neuron=int(neuron), relevant_trials=None, wave=wave, interval_start_seconds=interval[0], interval_stop_seconds=interval[1])
    return

def build_movie_all_together(six_video_map, wave='alpha', window_size=3, fps=10, channel=17, neuron=1, revision=None):
    # [1] Build still images
    build_video_stills_cartesian_and_polar(six_video_map, wave=wave, window_size=window_size, fps=fps, channel=int(channel), neuron=int(neuron), revision)
    #path_directory_to_save = 'figures/raw_data/spike-field-coherence/created-movies/channel'+str(channel)+'_'+wave+'_cartesian_and_polar'
    path_directory_to_save = 'figures/raw_data/spike-field-coherence-2021/created-movies/channel'+str(channel)+'_'+wave+'_cartesian_and_polar'
    # [2] Build video from still images
    image_folder = path_directory_to_save
    #build_video_format(image_folder='figures/raw_data/spike-field-coherence/created-movies/channel17_alpha_cartesian_and_polar', video_name='figures/movies/channel17_gorilla_movie.avi')
    build_video_format(image_folder=image_folder, video_name='figures/2021/movies/channel'+str(channel)+'_gorilla_movie.avi')
    return

######################################### End [13] Create video + figure 8c (still from video) #########################################
