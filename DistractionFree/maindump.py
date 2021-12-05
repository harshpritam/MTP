import sys
import re
import os
from math import sqrt, atan2
import pandas as pd


def CleanFile(FILENAME):
    NewFile = FILENAME.replace('.csv', 'Refined.csv')
    df = pd.read_csv(FILENAME, delimiter=',', error_bad_lines=False)
    df = df.dropna()
    for index, row in df.iterrows():
        csi_string = re.findall(r"\[(.*)\]", row.to_csv())[0]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']    
        if len(csi_raw) != 128:
            df.drop(index, inplace=True)
            print(index)
    df.to_csv(NewFile, index=False)
    return NewFile

def Dump(FILENAME):
    df = pd.read_csv(FILENAME, delimiter=',')
    real_timestamp = pd.DataFrame({"real_timestamp": df['real_timestamp']})
    amp_list = []
    phase_list = []
    file_util = FILENAME.replace('Refined.csv', '')
    AMPDUMP_PATH = './Amplitude' + file_util + 'Dump/'
    PHASEDUMP_PATH = './Phase' + file_util + 'Dump/'
    try: 
        os.mkdir(AMPDUMP_PATH) 
    except OSError as error: 
        print(error)
    try: 
        os.mkdir(PHASEDUMP_PATH) 
    except OSError as error: 
        print(error)
    
    f = open(FILENAME)
    flag = 0

    for j, l in enumerate(f.readlines()):
        if flag == 0:
            flag = 1
            continue
        imaginary = []
        real = []
        amplitudes = []
        phases = []
        # Parse string to create integer list
        csi_string = re.findall(r"\[(.*)\]", l)[0]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
        # Create list of imaginary and real numbers from CSI
        for i in range(len(csi_raw)):
            if i % 2 == 0:
                imaginary.append(csi_raw[i])
            else:
                real.append(csi_raw[i])
        # Transform imaginary and real into amplitude and phase
        for i in range(int(len(csi_raw) / 2)):
            amplitudes.append(sqrt(imaginary[i] ** 2 + real[i] ** 2))
            phases.append(atan2(imaginary[i], real[i]))
        amp_list.append(amplitudes)
        phase_list.append(phases)
    # print(type(amp_list))
    # print(type(phase_list))
    # print(type(real_timestamp))
    for k in range(64):
        amp_filename = 'Amplitude' + str(k) + '.csv'
        phase_filename = 'Phase' + str(k) + '.csv'
        amp_path = AMPDUMP_PATH + amp_filename
        phase_path = PHASEDUMP_PATH + phase_filename
        amp_temp = []
        phase_temp = []
        for item in amp_list:
            amp_temp.append(item[k])
        for item in phase_list:
            phase_temp.append(item[k])
        ampDF = pd.DataFrame({"Amplitude": amp_temp})
        phaseDF = pd.DataFrame({"Phase": phase_temp})
        ampDF = ampDF.join(real_timestamp)
        phaseDF = phaseDF.join(real_timestamp)
        ampDF = ampDF[["real_timestamp", "Amplitude"]]
        phaseDF = phaseDF[["real_timestamp", "Phase"]]
        ampDF.to_csv(amp_path, index=False)
        phaseDF.to_csv(phase_path, index=False)


if __name__ == "__main__":
    FILENAME = sys.argv[1]
    print(FILENAME)
    NewFile = CleanFile(FILENAME)
    Dump(NewFile)
