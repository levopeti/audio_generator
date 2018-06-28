from keras.models import load_model
from utils import files_for_train, generate_train_file, make_sample

epochs = 3
input_length = 128
batch_size = 256
input_path = "/home/biot/projects/Audio_generator/data/"
output_path = "/home/biot/projects/Audio_generator/generated_music/"
# input_path = "/root/inspiron/Audio_generator/data/"
# output_path = "/root/inspiron/Audio_generator/generated_music/"
sample_length = 120000

model = load_model('music_gen_model_5.h5')

make_sample(model=model, input_path=input_path, output_path=output_path, sample_length=sample_length, epoch=epochs,
           input_length=input_length, extension='mp3')

