from main import find_best_lead
import mne.io

def load_edf2(edfpath="./edf_file.edf"):
    data = mne.io.read_raw_edf(edfpath)
    raw_data = data.get_data()
    info = dict(data.info)
    channels = data.ch_names
    return raw_data, info, channels

part_duration = 30 #sec
sig = 'input_sigs/tmp/testfile.edf'
raw_data, info, channels = load_edf2(sig)
# cut signal for faster calculation
lead_1 = raw_data[0][30000:40000]
lead_2 = raw_data[1][30000:40000]
lead_3 = raw_data[2][30000:40000]

fs = int(info['sfreq'])
best_lead =  find_best_lead([lead_1,lead_2,lead_3],['lead_1','lead_2','lead_3'], part_duration, fs)
print(best_lead)

