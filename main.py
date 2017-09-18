#__AUTHOR__ : qqueing

from model import Model
import process_data_kaldi
import cPickle
import warnings
warnings.filterwarnings("ignore")



filter_sizes = [512, 512 ,512,512*3,512*3]
kernel_sizes = [5, 5, 7,1,1]
embeded_sizes = [500, 300]

input_vector_length = 200
input_dim = 20

num_classes = # This parameter is written by your training DB people
learning_rate = 1e-3

def Training(x,num_classes):

    cnn_model = Model()
    cnn_model.build_model(input_vector_length,  filter_sizes, kernel_sizes, num_classes,input_dim,embeded_sizes,learning_rate )
    cnn_model.run(x)

def evaluate(x,num_classes):

    cnn_model = Model()
    cnn_model.build_model(input_vector_length,  filter_sizes, kernel_sizes, num_classes,input_dim,embeded_sizes,learning_rate )
    cnn_model.eval(x)


def make_embedding(x,file_name):

    cnn_model = Model()
    cnn_model.build_model(input_vector_length,  filter_sizes, kernel_sizes, num_classes,input_dim,embeded_sizes ,learning_rate)
    outputs_data = cnn_model.make_embedding(x)
    process_data_kaldi.write_outputs("exp/ivectors_%s/ivector"%file_name.replace('raw_mfcc_',''),outputs_data)


def Training_kaldi(filename):
    process_data_kaldi.process_data(file_name = filename)
    x, num_classes= cPickle.load(open('data/processed/%s.p' % filename, "rb"))
    Training(x, num_classes)

def evaluate_kaldi():
    process_data_kaldi.process_data("data/processed/kaldi.p")
    x, num_classes= cPickle.load(open("data/processed/kaldi.p", "rb"))
    evaluate(x, num_classes)

def embedding_kaldi(filename):

    process_data_kaldi.process_data_test(file_name = filename)
    x= cPickle.load(open('data/processed/%s.p' % filename, "rb"))
    make_embedding(x,file_name = filename)



if __name__=="__main__":
    Training_kaldi(filename='raw_mfcc_train_subset_15')
    embedding_kaldi(filename='raw_mfcc_test_data')
    embedding_kaldi(filename='raw_mfcc_enroll_data')
    embedding_kaldi(filename='raw_mfcc_train')

    #maybe evaluate_kaldi() isn't need for you



