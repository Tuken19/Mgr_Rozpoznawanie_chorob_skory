import os
import sys
import getopt
import shutil
import time
import threading as th

import torch
from torch.backends import cudnn
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import *
import PIL
import numpy as np
from IPython.display import display
import random
from torch.utils.data import DataLoader

import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import csv

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn import metrics
from torchsampler import ImbalancedDatasetSampler

# ===== Globalne dane =====
# Slownik klas
class_name = {0: 'AK', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'SCC', 7: 'VASC'}  # Slownik class_name[identyfikator_klasy]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
main_dir_path = os.path.dirname(os.path.realpath(__file__))

# ===== Windows =====
# train_data_path = ''.join([main_dir_path, '\\', 'TrainingData'])
# validation_data_path = ''.join([main_dir_path, '\\', 'ValidationData'])
# test_data_path = ''.join([main_dir_path, '\\', 'TestData'])
# param_resnet_18_path = ''.join([main_dir_path, '\\', 'parametry_sieci_resnet18.pth'])

# ===== Linux =====
train_data_path = ''.join([main_dir_path, '/', 'TrainingData'])
validation_data_path = ''.join([main_dir_path, '/', 'ValidationData'])
test_data_path = ''.join([main_dir_path, '/', 'TestData'])
# param_resnet_18_path = ''.join([main_dir_path, '/', 'parametry_sieci_resnet18.pth'])

np.set_printoptions(precision=3)
# =========================


def main(argv):
    # ========== Obsługa flag ==========
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:d:hs:e:l:w:", ["--decay=", "--dropout=", "--help", "--save_dir_path=", "--epoch=", "--learning_rate=", "--weights="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        # usage()
        sys.exit(2)

    decay = 0
    dropout_p = 0
    save_dir_path = ''.join([main_dir_path, '/'])
    epoch = 1
    lr = 1e-5
    weights = False

    for o, a in opts:
        if o in ("-c", "--decay"):
            decay = float(a)
            print("Decay: {}".format(decay))
        elif o in ("-d", "--dropout"):
            dropout_p = float(a)
        elif o in ("-e", "--epoch"):
            epoch = int(a)
        elif o in ("-h", "--help"):
            # usage()
            print('''
                \n-c, --decay         : Weight decay
                \n-d, --dropout       : Dropout probability <0, 1>
                \n-e, --epoch         : Number of epochs
                \n-s, --save_dir_path : Path to save files
                \n-l, --learning_rate : Learning rate value
                \n-w, --weights       : Balancing with weights (True) or using sampler (False)
                \n''')
            sys.exit()
        elif o in ("-s", "--save_dir_path"):
            save_dir_path = ''.join([save_dir_path, a, '/'])
        elif o in ("-l", "--learning_rate"):
            lr = float(a)
            print("Lr: {}".format(lr))
        elif o in ("-w", "--weights"):
            weights = bool(a)
            print("Weights: {}".format(weights))
        else:
            assert False, "unhandled option"
    # ==================================

    assert isinstance(device, torch.device)
    print('Running on: {}'.format(device))

    # ========== Odnosnik do treningowego zbioru danych ==========

    if not os.path.exists(train_data_path):
        print("Provided path does not exist.")
        os.mkdir(train_data_path)
        print("New directory was created.")

    # Sprawdzenie czy zbior danych treningowych istnieje.
    assert os.path.exists(train_data_path)
    # ============================================================

    # =================== Definicja transformat ==================
    crop_size = 900  #
    image_size = 224  #
    random_crop_size = 224  # Wyjściowy rozmiar obrazu
    train_transform = Compose([
        CenterCrop(crop_size),
        Resize(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])
    test_transform = Compose([
        CenterCrop(crop_size),
        Resize(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])

    aug_transform_1 = Compose([
        CenterCrop(crop_size),
        RandomRotation(10),
        Resize(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])

    aug_transform_2 = Compose([
        Resize(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(30),
        Resize(image_size + 4),
        RandomCrop(random_crop_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])

    aug_transform_3 = Compose([
        Resize(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(180),
        Resize(image_size + 4),
        RandomCrop(random_crop_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])

    aug_transform_4 = Compose([
        CenterCrop(crop_size),
        RandomRotation(5),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        Resize(image_size + 30),
        RandomCrop(random_crop_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])
    # ============================================================

    # ================ Podzial zbioru na podzbiory ===============
    # dataset_do_testow = ImageFolder(train_data_path, transform=train_transform)
    train_dataset = ImageFolder(train_data_path, transform=aug_transform_3)
    validation_dataset = ImageFolder(validation_data_path, transform=test_transform)
    test_dataset = ImageFolder(test_data_path, transform=test_transform)

    # ************** Do testów **************
    # ***** Ustalenie wielkości zbiorów *****
    # train_dataset_size = int(len(dataset_do_testow) * 0.8)                                # 80%
    # val_dataset_size = int(len(dataset_do_testow) * 0.1)                                  # 10%
    # test_dataset_size = len(dataset_do_testow) - train_dataset_size - val_dataset_size    # 10%
    # train_dataset, validation_dataset, test_dataset = random_split(dataset_do_testow, [train_dataset_size, val_dataset_size, test_dataset_size])
    # ***************************************
    # ***************************************

    # ========== Sprawdzenie poprawności zbiorów danych ==========

    assert len(train_dataset) + len(
        validation_dataset) == 20269, 'Łączna liczba obrazów w zbiorze treningowym i walidacyjnym musi być równa 20269 a wynosi: {}'.format(
        len(train_dataset) + len(validation_dataset))
    assert len(test_dataset) == 5062, 'Liczba obrazów w zbiorze testowym musi być równa 5062 a wynosi: {}'.format(
        len(test_dataset))

    print('Liczba elementów w zbiorze treningowym: {}'.format(len(train_dataset)))
    print('Liczba elementów w zbiorze walidacyjnym: {}'.format(len(validation_dataset)))
    print('Liczba elementów w zbiorze testowym: {}'.format(len(test_dataset)))
    # ============================================================

    # =========== Zdefiniowanie batchy do uczenia sieci ==========
    batch_size = 10

    # ***** Balanced dataset with weights *****
    if weights:
        train_gen = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
        val_gen = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
        test_gen = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    else:
        # ***** Balanced dataset with sampler *****
        train_sampler = ImbalancedDatasetSampler(train_dataset)
        validation_sampler = ImbalancedDatasetSampler(validation_dataset)
        train_gen = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=10)
        val_gen = DataLoader(dataset=validation_dataset, sampler=validation_sampler, batch_size=batch_size, shuffle=False, num_workers=10)
        test_gen = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    # ***** Not balanced dataset *****
    # train_gen = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    # val_gen = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=7)
    # test_gen = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=7)

    print('Ilosc batchy treningowych:  {}'.format(len(train_gen)))
    print('Ilosc batchy walidacyjnych: {}'.format(len(val_gen)))
    print('Ilosc batchy testowych:     {}'.format(len(test_gen)))
    # ============================================================

    # ============================================================
    # ================ Przygotowanie modeli sieci ================
    # ============================================================
    # Wykorzystanie modelu sieci Resnet-101, Resnet-152 oraz Inception_v3 z biblioteki torchvision.models
    # pre-trenowany na duzym zbiorze danych obrazowych (IMAGENET)
    net_resnet18 = models.resnet18(pretrained=True)
    # net_resnet34 = models.resnet34(pretrained=True)
    # net_resnet101 = models.resnet101(pretrained=True)
    # net_resnet152 = models.resnet152(pretrained=False)
    # net_inception_v3 = models.inception_v3(pretrained=True)
    
    net = net_resnet18   # net_resnet34   # net_resnet101   net_resnet152   net_inception_v3

    # Zamrozenie parametrow sieci jest potrzebne w przypadku gdy model jest pretrenowany.
    # Jesli trening jest od poczatku nie zamrazamy parametrow.
    # Zamrozenie parametrow sieci
    # for param in net.parameters():
    #     param.requires_grad = False

    # Zmodyfikacja sieci by zwaracala tensor o liczbie cech rownej liczbie rozpoznawanych klas
    # Parametry nowoskonstruowanej warstwy beda mialy domyslne ustawienie requires_grad=True

    amend_resnet_classifier(net, num_classes=8, dropout_probability=dropout_p)

    # ***************** Trening sieci ****************
    cudnn.fastet = True
    print('Trening modelu: \n')
    print('Saving folder: ', save_dir_path)
    net.to(device)
    train_model(net, train_gen, val_gen, num_epochs=epoch, lr=lr, save_dir_path=save_dir_path, weight_decay=decay, weights=weights)
    # *************************************************

    # ****************** Test modelu ******************
    # print('Test modelu: \n')
    # test_model(net, test_gen, save_dir_path=save_dir_path)
    # *************************************************
    # ============================================================


# ========== Funkcje pomocnicze ==========
def per_channel_stats(image_tensor):
    mean = np.zeros((3,))
    std = np.zeros((3,))

    # Tablica mean - zawiera średnią wartość każdego kanału barwnego,
    # Tablica std  - zawiera odchylenie standardowe wartości każdego kanału barwnego.
    # *****************
    for c in range(3):
        s, m = torch.std_mean(image_tensor[c])
        mean[c] = m.item()
        std[c] = s.item()
    # ******************

    return mean, std


def tensor_to_image(tensor_image):
    # Przekształcenie odwrotne do normalizacji tesnora
    inv_normalize = Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                              std=[1. / 0.229, 1. / 0.224, 1. / 0.225])
    to_pil_image = Compose([inv_normalize, ToPILImage()])
    image = to_pil_image(tensor_image)
    return image


def show_examples(dataset, n):
    for i in range(n):
        tensor_image, target = random.choice(dataset)
        image = tensor_to_image(tensor_image)
        display(image)
        print('Etykieta klasy: {}    Nazwa klasy: {}'.format(target, class_name[target]))
        print('')


def count_classes(dataset):
    count = {}
    for image_tensor, target in dataset:
        if target not in count:
            count[target] = 0
        count[target] += 1

    return count


def calculate_weights(dataset):
    c = count_classes(dataset)
    weights = [0]*len(c)
    for label in c:
        weights[label] = 1/c[label]

    return torch.tensor(weights, device=device)


# Pomocnicza funkcja obliczająca liczbę poprawnych klasyfikacji
def count_correct_predictions(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return correct


def train_model(net, train_gen, val_gen, num_epochs, lr, save_dir_path, weight_decay, weights):
    # Definicja funkcji straty i optymalizatora
    if weights:
        weight = calculate_weights(train_gen.dataset)
        loss_function = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    mean_train_loss = [0] * num_epochs
    mean_val_loss = [0] * num_epochs
    epochs = list(range(num_epochs))

    correct_train_perc = [0] * num_epochs
    correct_val_perc = [0] * num_epochs

    try:
        for epoch in epochs:
            start_time_date = time.ctime()
            start_time = time.time()
            # Etap trenowania
            net.train()  # Przelaczenie modelu w tryb trenowania
            loss_l = []
            correct_train = 0
            total_train = 0
            for i, (images, labels) in enumerate(train_gen):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                optimizer.zero_grad()
                loss = loss_function(outputs, labels)
                correct_train += count_correct_predictions(outputs, labels)
                total_train += len(labels)
                loss_l.append(loss.item())
                loss.backward()
                optimizer.step()

            mean_train_loss[epoch] = np.mean(loss_l)

            # Etap walidacji
            net.eval().cuda()  # Przelaczenie modelu w tryb ewaluacji
            loss_l = []
            correct_val = 0
            total_val = 0
            best_val = 0
            with torch.no_grad():
                for images, labels in val_gen:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    correct_val += count_correct_predictions(outputs, labels)
                    total_val += len(labels)
                    loss = loss_function(outputs, labels)
                    loss_l.append(loss.item())

            mean_val_loss[epoch] = np.mean(loss_l)

            correct_train_perc[epoch] = (100. * correct_train) / total_train
            correct_val_perc[epoch] = (100. * correct_val) / total_val

            # Zapis najlepszej sieci
            if correct_val_perc[epoch] > best_val:
                best_val = correct_val_perc[epoch]
                path = ''.join([save_dir_path, '{}'.format(epoch), '_parametry_sieci_resnet34.pth'])
                torch.save(net, path)

            # Wyswietlanie komunikatu o postepach nauki
            elapsed_time = time.time() - start_time
            s = 'Epoka: {}/{}   sredni blad (trening/walidacja): {:.4f} / {:.4f}   Stopa sukcesu (trening/walidacja): {:.2f}% / {:.2f}%     Czas startu: {}     Czas obliczen: {:.2f} min'
            print(s.format(epoch + 1, num_epochs, mean_train_loss[epoch], mean_val_loss[epoch], correct_train_perc[epoch],
                           correct_val_perc[epoch], start_time_date, elapsed_time / 60))
    except KeyboardInterrupt:
        pass

    # ========== Wykresy ==========
    plt.figure(1, (15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mean_train_loss, 'b-', epochs, mean_val_loss, 'r-')
    plt.grid(True)
    plt.ylabel('Blad')
    plt.xlabel('Numer epoki')
    plt.legend(('Mean train loss', 'Mean val loss'))
    plt.autoscale(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, correct_train_perc, 'b-', epochs, correct_val_perc, 'r-')
    plt.grid(True)
    plt.ylabel('Ilosc poprawnie sklasyfikowanych [%]')
    plt.xlabel('Numer epoki')
    plt.legend(('Correct train', 'Correct val'))
    plt.autoscale(True)

    plt.savefig(''.join([save_dir_path, 'ResNet18-uczenie.png']), format='png')
    plt.show()
    print('Mean train loss = {}\nMean val loss   = {}'.format(mean_train_loss, mean_val_loss))
    print('Correct train = {}\nCorrect val   = {}'.format(correct_train_perc, correct_val_perc))
    # =============================


def amend_resnet_classifier(net, num_classes, dropout_probability=0):
    # Pomocnicza funkcja, zastepujaca ostatnia warstwe sieci typu Resnet nowa warstwa liniowa, zwracajaca tensor o liczbie cech rownej liczbie rozpoznawanych klas
    num_in_features = net.fc.in_features
    num_out_features = net.fc.out_features

    net.fc = nn.Sequential(nn.Dropout(dropout_probability),
                           nn.Linear(num_in_features, num_classes))
    # Zastapienie ostatniej warstwy sieci typu Resnet (warstwa 'fc') nowa warstwa zwaracajaca wymagana liczbe cech
    print('Zmodyfikowano siec. Dodano dropout. Liczba zwracanych cech zmieniona z {} na {}'.format(num_out_features, num_classes))


# Funkcja pomocnicza rysująca wykres słupkowy dla zadanego rozkładu prawdopodobieństwa podanego jako tablica prawdopodobieństw.
def print_bar_probabilities(probabilities, labels=[-1]):
    # Dla pojedynczego rekordu (1 obrazka)
    if len(probabilities) == 2:
        plt.bar(class_name.values(), probabilities)
        if labels != [-1]:
            plt.title('Actual label: {}'.format(class_name.get(labels[0])))
        else:
            plt.title('Probe')
        plt.ylim(0, 1)
        plt.text(0, 0.1, probabilities)
        plt.xlabel('Label')
        plt.ylabel('Probability')
        plt.show()

    # Dla sekwencji rekordów (kilku obrazków)
    else:
        for line in range(len(probabilities)):
            plt.bar(class_name.values(), probabilities[line])
            plt.title('Actual label: {}'.format(class_name.get(labels[line])))
            plt.ylim(0, 1)
            plt.text(0, 0.1, probabilities[line])
            plt.xlabel('Label')
            plt.ylabel('Probability')
            plt.show()


def print_2d_plot(array, title='', save_dir_path=''):
    if isinstance(array, np.ndarray):
        fig, ax = plt.subplots()
        ax = sns.heatmap(array, annot=True, linewidths=0.5, xticklabels=1, yticklabels=1, annot_kws={"size": 10}, fmt='.2f')
        ax.set_xticklabels(class_name.values())
        ax.set_yticklabels(class_name.values())
        ax.xaxis.tick_top()

        plt.title(title)
        plt.xlabel('Etykiety przewidziane')
        plt.ylabel('Etykiety rzeczywiste')

        plt.savefig(''.join([save_dir_path, 'MP.png']), format='png')
        plt.show()


def test_model(net, test_gen, save_dir_path):
    net.eval()  # Przełącz model w tryb ewaluacji
    correct = 0
    total = 0
    prob = np.zeros((1, 8))
    p_lab = np.zeros(1)
    lab = np.zeros(1)
    img_names = ['']
    img_names_batch = []

    for i, (images, labels) in enumerate(test_gen, 0):
        images = images.to(device)
        labels = labels.to(device)
        img_names_batch = test_gen.dataset.samples
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        total += len(labels)

        # Przepuszczenie wyjść przez warstwę SoftMax - konwersja na prawdopodobieństwo
        sm = torch.nn.Softmax(1)
        probabilities = sm(outputs)

        # Połączenie wyjść ze wszystkich batchy w jeden tensor.
        prob = np.concatenate((prob, probabilities.cpu().detach().tolist()))
        p_lab = np.concatenate((p_lab, predicted.cpu().detach().numpy()))
        lab = np.concatenate((lab, labels.cpu().detach().numpy()))

    for img_name, _ in img_names_batch:
        poczatek_nazwy = img_name.rfind("ISIC")
        img_names.append(img_name[poczatek_nazwy:])

    _, predicted = torch.max(torch.tensor(prob), 1)

    lab = lab[1:]
    p_lab = p_lab[1:]
    img_names = img_names[1:]
    prob = prob[1:]

    print('Labels: ', lab, len(lab))
    print('Predic: ', p_lab, len(p_lab))
    print('Probab: ', prob)
    print('IMG_count:', len(img_names))

    # Stworzenie plików 'GroundTruth.csv' i 'Predicted.csv'
    save_to_csv(''.join([save_dir_path, 'GroundTruth.csv']), img_names, lab)
    save_to_csv(''.join([save_dir_path, 'Predicted.csv']), img_names, prob.tolist())

    pred = list(predicted.cpu().detach().numpy())
    pred = pred[1:]

    # print_bar_probabilities(prob, lab)

    # Wyciągnięcie największych prawdopodobieństw
    # top_2_prob, top_2_labels = prob.topk(2)
    # print("Predic: {}".format(predicted.cpu().detach().numpy()))
    # print("Labels: {}".format(top_2_labels.cpu().detach().numpy()))
    # print("Probab: {}".format(top_2_prob.cpu().detach().numpy()))

    print('Stopa sukcesu: {:.2f}%'.format((100. * correct) / total))
    # print('Labels:    {}\nPredicted: {}'.format(lab, pred))

    with open(''.join([save_dir_path, 'Wyniki_Testu.txt']), 'w') as f:
        # Print the confusion matrix
        print("Confusion matrix:", file=f)
        print(metrics.confusion_matrix(lab, pred, normalize='true'), file=f)

        # Print the precision and recall, among other metrics
        print("Metrics:", file=f)
        print(metrics.classification_report(lab, pred, digits=3), file=f)
        print('\n', file=f)

        print('Recall micro:', file=f)
        print(metrics.recall_score(lab, pred, average='micro'), file=f)
        print('\n', file=f)

        print('Precision micro:', file=f)
        print(metrics.precision_score(lab, pred, average='micro'), file=f)
        print('\n', file=f)

        # Plot confusion matrix
        macierz_pomylek = metrics.confusion_matrix(lab, pred, normalize='true')
        print_2d_plot(macierz_pomylek, title='Macierz pomylek', save_dir_path=save_dir_path)


def save_to_csv(file_name, names, labels):
    # Stworzenie pliku csv

    with torch.no_grad(), open(file_name, "w") as f:
        headings = class_name.values()
        headings = list(headings)
        headings = ['image'] + headings
        filewriter = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(headings)

        for i, img_name in zip(labels, names):
            lab_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if type(i) == list:
                lab_list = i
            else:
                lab_list[int(i)] = 1.0

            lab_list = [img_name] + lab_list
            filewriter.writerow(lab_list)
            f.flush()
    f.close()


def confusion_matrix(actual_labels, predicted_labels):
    #                       predicted labels
    #                AK  BCC BKL DL  MEL NV  SCC VASC
    #                ___ ___ ___ ___ ___ ___ ___ ___
    #               |___|___|___|___|___|___|___|___|
    #               |___|___|___|___|___|___|___|___|
    #               |___|___|___|___|___|___|___|___|
    # actual labels |___|___|___|___|___|___|___|___|
    #               |___|___|___|___|___|___|___|___|
    #               |___|___|___|___|___|___|___|___|
    #               |___|___|___|___|___|___|___|___|
    #               |___|___|___|___|___|___|___|___|
    # values in %
    number_of_classes = len(class_name.keys())
    conf_mat = np.zeros((number_of_classes, number_of_classes))
    for i in range(len(actual_labels)):
        conf_mat[int(actual_labels[i])][int(predicted_labels[i])] += 1

    for i in range(number_of_classes):
        for j in range(number_of_classes):
            conf_mat[i][j] /= (conf_mat[i]).sum()
            conf_mat[i][j] *= 100  # Wynik w %
    print('=============== Confusion matrix ===============')
    print(conf_mat)


def classify(net, dataset):
    image, label = dataset
    test_gen = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    net.eval()  # Przełącz model w tryb ewaluacji
    correct = 0
    total = 0
    with torch.no_grad():
        images, labels = test_gen
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        total += len(labels)

        # print("Outputs:      {}".format(outputs)) # print output from crossentropy score
        print("Predicted:    {}".format(predicted.squeeze()))
        print("Actual label: {}".format(label))

    # Przepuszczenie wyjść przez warstwę SoftMax - konwersja na prawdopodobieństwo
    sm = torch.nn.Softmax(1)
    probabilities = sm(outputs)
    prob = probabilities.cpu().detach().numpy()[0]
    print_bar_probabilities(prob, [label])
# ========================================


if __name__ == '__main__':
    main(sys.argv[1:])
