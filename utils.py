import numpy as np
import random
import glob
import smtplib
from pydub import AudioSegment
import scipy.io.wavfile


def generate_train_file(training_files, batch_size, input_length):
    input_data = []
    target_data = []
    while True:
        for file in training_files:
            sound = AudioSegment.from_mp3(file)
            raw_data = sound.get_array_of_samples()
            sound_array = np.array(raw_data)
            sound_array = sound_array.reshape(sound.channels, -1, order='F')
            assert len(sound_array[0]) == len(sound_array[1])
            sound_array = sound_array[0, :]

            _mean = np.mean(sound_array)
            _std = np.std(sound_array)
            sound_array = (sound_array - _mean) / _std

            for delta in range(len(sound_array) - input_length - 1):   # len(sound_array)[0]
                start = delta
                end = delta + input_length
                # _input = sound_array[:, start:end]
                # _target = sound_array[:, end + 1]
                _input = sound_array[start:end]
                _target = sound_array[end + 1]

                input_data.append(np.array(_input))
                target_data.append(np.array(_target))

                if len(input_data) == batch_size:
                    input_data = np.array(input_data)
                    target_data = np.array(target_data)
                    input_data = input_data.reshape(input_data.shape + (1,))
                    target_data = target_data.reshape(target_data.shape + (1,))

                    yield ({'input_1': input_data}, {'output': target_data})
                    input_data = []
                    target_data = []


def files_for_train(_file_path, extension='mp3'):

    training_files = glob.glob(_file_path + '*.' + extension)

    return training_files


def make_sample(model, input_path, output_path, sample_length, epoch, input_length, extension='mp3'):
    files = glob.glob(input_path + '*.' + extension)
    print(files)
    random.shuffle(files)

    sound = AudioSegment.from_mp3(files[0])
    rate = sound.frame_rate
    raw_data = sound.get_array_of_samples()
    sound_array = np.array(raw_data)
    sound_array = sound_array.reshape(sound.channels, -1, order='F')

    _mean = np.mean(sound_array)
    _std = np.std(sound_array)
    sound_array = (sound_array - _mean) / _std

    # _rand = np.random.randint(len(sound_array[0])//2 + 1000)
    # seed_part = sound_array[:, _rand:(_rand + input_length)]
    # generated_array = np.array(seed_part)
    # generated_array = np.transpose(generated_array, (1, 0))
    # seed_part = seed_part.reshape((1,) + seed_part.shape)
    #
    # for i in range(sample_length):
    #     sample = model.predict(seed_part, batch_size=1)
    #     generated_array = np.concatenate((generated_array, sample), axis=0)
    #     seed_part = np.transpose(generated_array[-input_length:], (1, 0))
    #     seed_part = seed_part.reshape((1,) + seed_part.shape)
    #
    #     if i % (sample_length // 20) == 0:
    #         print(str((i * 100) / sample_length) + '%')
    #
    # generated_array = np.transpose(generated_array, (1, 0))

    sound_array = sound_array[0]
    _rand = np.random.randint(len(sound_array)//2 + 1000)
    seed_part = sound_array[_rand:(_rand + input_length)]
    generated_array = np.array(seed_part)
    seed_part = seed_part.reshape((1,) + seed_part.shape)
    seed_part = seed_part.reshape(seed_part.shape + (1,))

    for i in range(sample_length):
        sample = model.predict(seed_part, batch_size=1)
        generated_array = np.append(generated_array, sample)
        seed_part = generated_array[-input_length:]
        seed_part = seed_part.reshape((1,) + seed_part.shape)
        seed_part = seed_part.reshape(seed_part.shape + (1,))

        if i % (sample_length // 20) == 0:
            print(str((i * 100) / sample_length) + '%')

    generated_array = (generated_array * _std) + _mean

    scipy.io.wavfile.write(output_path + str(epoch) + "_gen.wav", rate, generated_array.T)







