import os
import random

import cv2
import pandas as pd
import shutil
from PIL import Image
from shutil import copyfile
import numpy as np
__all__ = [cv2]


def main():
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # file_path = ''.join([dir_path, '\\NazwyKlasy.xlsx'])
    #
    # df = pd.read_excel(file_path)
    #
    # for abbrev in df.columns:
    #     print(abbrev)
    #     copy_picture(df, abbrev, dir_path)

    # ===== Random choose test set =====
    random_choose(12.4)

    # ===== Transformating pictures =====
    # modify_pictures('VASC', 'Flip_Vertical')


def copy_picture(data_frame, name, directory_path):
    path = ''.join([directory_path, '\\', name])

    pair = zip(data_frame['Nazwa'], data_frame[name])
    for n, val in pair:
        if val == 1:
            file = directory_path + '\\ISIC_2019_Training_Input\\' + n + '.jpg'
            new_path = shutil.copy(file, path)
            print(new_path)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], None, cv2.INTER_LINEAR)
    return result


def modify_pictures(dir_name, how_modified):
    """
    :param dir_name: All files from this directory will be copied (String)
    :param how_modified: Prefix for new file names (String)
    :return:
    """
    main_dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = ''.join([main_dir_path, 'TrainingData', '\\', dir_name])
    dir_path_new = ''.join([main_dir_path, 'TrainingData', '\\', dir_name, '_', how_modified])

    # Make new directory if does not exist
    if not os.path.exists(dir_path_new):
        print("Provided path does not exist.")
        os.mkdir(dir_path_new)
        print("New directory was created.")

        # Copying and modifying pictures
        for picture in os.listdir(dir_path):
            src_path = ''.join([dir_path, '\\', picture])
            dst_path = ''.join([dir_path_new, '\\', how_modified, '_', picture])
            random_angle = random.randint(5, 10)  # Random angle <5, 10>
            pic = cv2.imread(src_path, 1)
            # pic = rotateImage(pic, 180)  # Rotate
            pic = cv2.flip(pic, 0)       # Flip Vertical
            # pic = cv2.flip(pic, 1)       # Flip Horizontal
            cv2.imwrite(dst_path, pic)


def random_choose(percentages):
    main_path = os.path.dirname(os.path.realpath(__file__))
    train_path = ''.join([main_path, '\\', 'TrainingData'])
    test_path = ''.join([main_path, '\\', 'ValidationData'])

    for directory in os.listdir(train_path):
        print(directory)
        dir_path = ''.join([train_path, '\\', directory])

        populacja = os.listdir(dir_path)
        k = int(percentages / 100 * len(populacja))
        lista_obrazow_testowych = random.sample(population=populacja, k=k)

        print('Liczba obrazow ogolnie: {}\nLiczba obrazow testowych: {}'.format(len(populacja), len(lista_obrazow_testowych)))
        for picture in lista_obrazow_testowych:
            file_path = dir_path + '\\' + picture
            dest_path = test_path + '\\' + directory + '\\' + picture
            new_path = shutil.move(file_path, dest_path)

        pop = os.listdir(dir_path)
        print('Liczba obrazow ogolnie: {}'.format(len(pop)))


if __name__ == '__main__':
    main()
