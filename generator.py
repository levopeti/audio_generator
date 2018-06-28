from keras.utils import multi_gpu_model
from models import generate_model
from utils import files_for_train, generate_train_file, make_sample
import os
from keras.models import load_model


epochs = 3
input_length = 200
batch_size = 256
# input_path = "/home/biot/projects/Audio_generator/data/"
# output_path = "/home/biot/projects/Audio_generator/generated_music/"
input_path = "/root/inspiron/Audio_generator/data/"
output_path = "/root/inspiron/Audio_generator/generated_music/"
sample_length = 500000

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

training_files = files_for_train(_file_path=input_path, extension='mp3')
print(training_files)
#  The length of the Ludmilla mix is 156187055
#  The  length  of  the  Maldova  is  17493167

model = generate_model(input_length)

#model = load_model('music_gen_model_4.h5')

#model = multi_gpu_model(model, gpus=[0, 1, 2, 3])
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.fit_generator(generator=generate_train_file(training_files, batch_size, input_length),
                    steps_per_epoch=68000,  # 610000, 68000
                    epochs=7)

model.save('music_gen_model_7.h5')

make_sample(model=model, input_path=input_path, output_path=output_path, sample_length=sample_length, epoch=epochs,
           input_length=input_length, extension='mp3')



