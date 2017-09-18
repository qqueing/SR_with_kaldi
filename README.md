# Speaker embeddings for Text-independent speaker verification using TensorFlow, with Kaldi
This is a slightly modified TensorFlow implementation of the model presented by David Snyder in [Deep Neural Network Embeddings for Text-Independent Speaker Verification](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf).

In the paper, this algorithm is a little worse than i-vector. My test show similar output. Also, in my test, shallow network was a very little worse than deep network (This is dependency of DB). <br />

In this code, there are many hard cording such folder location and some parameter related database. If I have database well-known SR database, I try to it. but I only have private database.<br />

I hope this code helps researcher.


## Credits
If you're using this code please make sure you cite following papers:
- Snyder's paper:
```
@unknown{unknown,
author = {Snyder, David and Garcia-Romero, Daniel and Povey, Daniel and Khudanpur, Sanjeev},
title = {Deep Neural Network Embeddings for Text-Independent Speaker Verification},
year = {2017}
}
```

Also, use the part of code:
- [mangate's git repository](https://github.com/mangate/ConvNetSent)
   - tensorflow classification baseline code
- [Karel Vesely's git repository](https://github.com/vesis84/kaldi-io-for-python)
   - kaldi io for python

## Features
- Supports kaldi input&output style(input : mfcc scp-ark pair, output : embedding scp-ark pair)
    - This code can replace i-vector train - extraction part in kaldi egs/SRE10/v1.
- Instead of concatenate VAD frame, I use orginal frame contain non-speech frame.
    - Training case, Many frame was used to train. Test case, max power frame to test. Detail is in the process_data_kaldi.py load_dataset function
    - This part depend on your opinion.
- Adding input layer mean normalization instead of exptional block.
- Adding some layer dropout and Batch normalized.
- Adding L2 loss in last layer.


## Requirements
- Python (2.7)
- NumPy
- TensorFlow (I tried only 1.3 version)
- Database

## Usage
### Preperation:
1) Clone the repository recursively to get all folder and subfolders
2) Prepare Database(I use private DB. If you need, the script needs to be modified)
3) Use Kaldi-recipe extracing MFCC and VAD in SRE10/v1/run.sh


### Running:
1) run Training_kaldi function in make_dvec.py.<br />
   after, run embedding_kaldi function.(Some function was written hard cording. Change you file location)
2) use kaldi-recipe calculating mean vector and PLDA scoring.<br />
   Maybe, you only run after /local/extract_ivectors.sh --stage 2 each folder.


## Authors
qqueing@gmail.com( or kindsinu@naver.com)

