#__AUTHOR__ : qqueing
import numpy as np
import cPickle

import os.path


data_per_wav = 200

folder_location='/data1/code2/kaldi-trunk/egs/test/v4_test/mfcc'

def find_all(name, path):
    import os
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file and 'scp' in file:
                result.append([os.path.join(root, file.replace('raw_mfcc','vad')), os.path.join(root, file)])
    return result

def load_dataset(filename ):


    #This function is very heuristics. If you need, modify.

    data = {}
    file_list = find_all(filename,folder_location)
    import kaldi_io
    from itertools import izip
    raw_data = {}
    for vad_file, mfcc_file in file_list:
        for (key1, vec1) in kaldi_io.read_vec_scp(vad_file):
            if key1 in raw_data:
                raw_data[key1].append(vec1)
            else:
                raw_data[key1] = [vec1]

    for vad_file, mfcc_file in file_list:
        for (key1, vec1) in kaldi_io.read_vec_scp(mfcc_file):
            if key1 in raw_data:
                raw_data[key1].append(vec1)
            else:
                print("error!")
                exit

    for datum in raw_data:
        vec1, vec2 = raw_data[datum]


        idx_list = []
        for idx, i in enumerate(vec1[:-data_per_wav]):
            if np.sum(vec1[idx:idx + data_per_wav]) == data_per_wav:
                idx_list.append(idx)

        if idx_list != []:
            max_idx = np.argmax(vec2[:,0])
            max_idx = max_idx - data_per_wav/2
            idx_list2 = [ x for x in idx_list if np.abs(max_idx -x ) <= 100 and  np.abs(max_idx -x )% 15  == 0]

            if len(idx_list2) == 0 :
                idx_list2 = [max(min(max_idx,len(vec2)-data_per_wav),0),max(min(idx_list[len(idx_list) / 2],len(vec2)-data_per_wav),0) ]

            if datum.split('-')[0] in data:
                data[datum.split('-')[0]].append((idx_list2, vec2))
            else:
                data[datum.split('-')[0]] = [(idx_list2, vec2)]

        else:

            if datum.split('-')[0] in data:
                data[datum.split('-')[0]].append(([0], vec2))
            else:
                data[datum.split('-')[0]] = [([0], vec2)]

    return data





def process_data(file_name):
    dump_file_name = 'data/processed/%s.p' % file_name
    if os.path.isfile(dump_file_name):
        print "file {} already exists".format(dump_file_name)
        return


    print "building data...",

    data = []
    data_set = load_dataset(file_name=file_name)
    num_classes = len(data_set)


    for p_idx, (p_name,p_data) in enumerate(data_set.iteritems()):
        for speech_idx, speech_data in p_data:
            datum = {"y": p_idx,
                    "speech": speech_data,
                     "idx": np.random.permutation(speech_idx)[0:min(data_per_wav, len(speech_idx))],
            }
            data.append(datum)

    print "data loaded!"

    cPickle.dump([data, num_classes], open(file_name, "wb"))

    print "dataset created!"

def load_dataset_test(filename ):

    data = {}
    file_list = find_all(filename,folder_location)
    import kaldi_io
    raw_data = {}

    for vad_file, mfcc_file in file_list:
        for (key1, vec1) in kaldi_io.read_vec_scp(mfcc_file):
            raw_data[key1] = vec1


    for datum in raw_data:
        vec2 = raw_data[datum]

        start_idx = np.argmax(vec2[:,0])
        start_idx = start_idx - data_per_wav/2

        start_idx  = max(start_idx , 0)
        last_idx = start_idx + data_per_wav

        if last_idx<len(vec2):

            data[datum] = vec2[start_idx:last_idx][:]
        else:

            if len(vec2) < data_per_wav:
                data[datum] = np.zeros((data_per_wav, 20))
                data[datum][start_idx:len(vec2)][:] = vec2[start_idx:last_idx][:]
            else:
                data[datum] = vec2[-200:][:]

    return data



def process_data_test(file_name ):

    dump_file_name = 'data/processed/%s.p'%file_name
    if os.path.isfile(dump_file_name):
        print "file {} already exists".format(dump_file_name)
        return

    # load data
    print "loading data...",
    data = load_dataset_test(filename=file_name)

    print "data loaded!"
    cPickle.dump(data , open(dump_file_name, "wb"))

    print "dataset created!"


def write_outputs(filename,data):

    scp_file = filename + '.scp'
    ark_file = filename + '.ark'

    scp_fd = open(scp_file, mode='w')
    ark_fd = open(ark_file, mode='w')

    for idx,datum in enumerate(data['key']):
        ark_fd.write("%s " % (datum))
        scp_fd.write("%s %s:%d\n" % (datum, ark_file,ark_fd.tell()))
        ark_fd.write(" %s\n" % (np.array2string(data['embed'][idx],max_line_width=999999999, formatter={'float_kind':lambda x: "%.7f" % x})).replace(']',' ]').replace('[','[ '))



    scp_fd.close()
    ark_fd.close()
