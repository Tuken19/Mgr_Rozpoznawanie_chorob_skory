#!/bin/bash
main_dir=`pwd`

# ===== Test 1 =====
# Nazwa nowego folderu
directory="Sieci_wyniki/ResNet34_fine_augumented_3_balanced_all_weight_dropout_0-2_decay_e-5"

# Znalezienie pliku .pth w folderze
cd $directory
plik=*.pth
param=`echo $plik`

cd $main_dir
# Wywołanie testu
#python3 Testy_modeli.py -s $directory -p $param

gt="${directory}/GroundTruth.csv"
pred="${directory}/Predicted.csv"
save="${directory}/Score.txt"

# Obliczenie i zapis metryk
isic-challenge-scoring classification $gt $pred>$save


# # ===== Test 2 =====
# # Nazwa nowego folderu
directory="Sieci_wyniki/ResNet34_fine_augumented_3_balanced_all_weight_dropout_0-5_decay_e-5"

# Znalezienie pliku .pth w folderze
cd $directory
plik=*.pth
param=`echo $plik`

cd $main_dir
# Wywołanie testu
#python3 Testy_modeli.py -s $directory -p $param

gt="${directory}/GroundTruth.csv"
pred="${directory}/Predicted.csv"
save="${directory}/Score.txt"

# Obliczenie i zapis metryk
isic-challenge-scoring classification $gt $pred>$save

# # ===== Test 3 =====
directory="Sieci_wyniki/ResNet34_fine_augumented_3_balanced_all_weight_dropout_0-7_decay_e-5"

# Znalezienie pliku .pth w folderze
cd $directory
plik=*.pth
param=`echo $plik`

cd $main_dir
# Wywołanie testu
#python3 Testy_modeli.py -s $directory -p $param

gt="${directory}/GroundTruth.csv"
pred="${directory}/Predicted.csv"
save="${directory}/Score.txt"

# Obliczenie i zapis metryk
isic-challenge-scoring classification $gt $pred>$save


