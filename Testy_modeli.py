import getopt
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import *

from Siec import amend_resnet_classifier, train_model, test_model

# ===== Globalne dane =====
# Słownik klas
class_name = {0: 'AK', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'SCC', 7: 'VASC'}  # Słownik class_name[identyfikator_klasy]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
main_dir_path = os.path.dirname(os.path.realpath(__file__))

# # ===== Windows =====
# test_data_path = ''.join([main_dir_path, '\\', 'TestData'])
# param_resnet_18_path = ''.join([main_dir_path, '\\', 'Sieci wyniki\\ResNet18_fine_normal_not_balanced\\', '14_parametry_sieci_resnet18.pth'])
# # =========================

# ===== Linux =====
test_data_path = ''.join([main_dir_path, '/', 'TestData'])
# param_resnet_18_path = ''.join([main_dir_path, '/', 'Sieci wyniki/', 'ResNet18_fine_normal_not_balanced', '/', '14_parametry_sieci_resnet18.pth'])
# =========================

# =================== Definicja transformat ==================
crop_size = 900             #
image_size = 224            #
random_crop_size = 224      # Wyjściowy rozmiar obrazu

test_transform = Compose([
        CenterCrop(crop_size),
        Resize(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])
# ============================================================


def main(argv):
    # ========== Obsługa flag ==========
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:p:", ["--help", "--save_dir_path=", "--net_param="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        # usage()
        sys.exit(2)

    save_dir_path = ''.join([main_dir_path, '/', 'Sieci wyniki/', 'ResNet18_fine_normal_not_balanced', '/'])
    param = ''.join([save_dir_path, '49_parametry_sieci_resnet18.pth'])

    for o, a in opts:
        if o in ("-h", "--help"):
            # usage()
            print('''\n-h, --help:          Help
            \n-s, --save_dir_path: Path to directory with net parameters which will be also a path to save files.
            \n-p, --net_param:     Name of file with net parameters to load.\n''')
            sys.exit()
        elif o in ("-s", "--save_dir_path"):
            save_dir_path = ''.join([main_dir_path, '/', a, '/'])
        elif o in ("-p", "--net_param"):
            param = ''.join([save_dir_path, a])
        else:
            assert False, "unhandled option"
    # ==================================
    print(save_dir_path)
    print(param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # ================ Podzial zbioru na podzbiory ===============
    # ***** Ustalenie wielkości zbiorów *****
    test_dataset = ImageFolder(test_data_path, transform=test_transform)
    # ***************************************

    # Sprawdzenie poprawności zbiorów danych
    # assert len(test_dataset) == 5062, 'Liczba obrazów w zbiorze testowym musi być równa 5062 a wynosi: {}'.format(
    #     len(test_dataset))
    print('Liczba elementów w zbiorze testowym: {}'.format(len(test_dataset)))

    # =========== Zdefiniowanie batchy do uczenia sieci ==========
    batch_size = 10

    test_gen = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print('Ilość batchy testowych:     {}'.format(len(test_gen)))
    # ============================================================

    # ****************** Test modelu ******************
    print('Test modelu: \n')
    net = torch.load(param)
    net.eval()
    test_model(net, test_gen, save_dir_path)
    # *************************************************
    # ============================================================


if __name__ == '__main__':
    main(sys.argv[1:])
