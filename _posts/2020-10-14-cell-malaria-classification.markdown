---
layout: post
title:  "ML Project 1: Cell Malaria"
date:   2020-10-14
categories: classification
---

# KLASIFIKASI CELL MALARIA DENGAN DEEP LEARNING

Pada project machine learning kali ini akan membahas mengenai klasifikasi gambar sel malaria berdasarkan penampakannya di dalam sel tubuh manusia yang diambil dari hasil gambar melalui mikroskop. Total data keseluruhan yaitu 27.560 gambar yang terbagi atas 13780 data gambar *Parasitized* (Terinfeksi) dan 13780 data gambar *Uninfected* (Tidak terinfeeksi). Dari keseluruhan data gambar tersebut ternya ada beberapa gambar yang mislabel (tidak sesuai kategori), hal ini dipaparkan melalui [jurnal berikut](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7277980/). Sehingga jumlah data yang benar-banar bersih menjadi 26498 data gambar dengan jumlah 13.303 data gambar *Parasitized* dan 13.195 data gambar *Uninfected*. Berarti ada 1060 data gambar yang misslabel, 447 data dari kategori *Parasitized* dan 585 data dari kategori *Uninfected*.  Selanjutnya untuk proses pembuatan model deep learning untuk klasi gambar akan dijelaskan melalui koding berikut:

## Persiapan Library

```python
# import library yang dibutuhkan
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import cv2 as cv
from PIL import Image
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from google.colab import files
from keras.preprocessing import image
import matplotlib.image as mpimg

import tensorflow as tf
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
```

## Persiapan Dataset
Dataset : [cell malaria](https://lhncbc.nlm.nih.gov/publication/pub9932)

```python
# mengekstrak file dataset terdapat mislabel zip ke colab
zip_path = '/content/drive/My\ Drive/Datasets/malaria.zip'
!cp {zip_path} /content/
!cd /content/
!unzip -q /content/malaria.zip -d /content
!rm /content/malaria.zip
```
```python
# mengekstrak file dataset tidak terdapat mislabel zip ke colab
zip_path = '/content/drive/My\ Drive/abc/malaria_cell_images.zip'
!cp {zip_path} /content/
!cd /content/
!unzip -q /content/malaria_cell_images.zip -d /content
!rm /content/malaria_cell_images.zip
```
### Tampilan dataset sebelum di-*cleaning*
```python
# menampilkan beberapa sampel dataset sebelum clreaning

# data terinfeksi
mis_para_data = 'cell_images/cell_images/Parasitized'
mis_para_data_list = list(paths.list_images(mis_para_data))

# data tidak teriinfeksi
mis_unin_data = 'cell_images/cell_images/Uninfected'
mis_unin_data_list = list(paths.list_images(mis_unin_data))

fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(11, 6))

for i in range(4):
  img_para = cv.imread(mis_para_data_list[i])
  output_para = cv.resize(img_para, (128,128))
  ax[0, i].imshow(output_para)
  ax[0, i].set_title('Terinfeksi')
  ax[0, i].axis(False)

  img_unin = cv.imread(mis_unin_data_list[i])
  output_unin = cv.resize(img_unin, (50,50))
  ax[1, i].imshow(output_unin)
  ax[1, i].set_title('Tidak Terinfeksi')
  ax[1, i].axis(False)
```
![datasetbeforecleaning](/images/setbefore.jpg)
```python
# memisahkan data dan label
img_dir = 'cell_images/cell_images'

bf_name = []
bf_label = []
bf_img_dir = []

for path, subdirs, files in os.walk(img_dir):
  for name in files:
    bf_img_dir.append(os.path.join(path, name))
    bf_label.append(path.split('/')[-1])
    bf_name.append(name)

# melihat jumlah data per label(kategori)
df_bf = pd.DataFrame({"bf_path":bf_img_dir,'bf_file_name':bf_name,"bf_label":bf_label})
df_bf.groupby(['bf_label']).size()
```
![databefore](/images/jumlahdata_before.jpg)

### Tampilan dataset setelah di-*cleaning*
```python
# menampilkan beberapa sampel dataset setelah cleaning

# data terinfeksi
para_data = 'malaria_cell_images/True Labeled Images/Parasitized'
para_data_list = list(paths.list_images(para_data))

# data tidak teriinfeksi
unin_data = 'malaria_cell_images/True Labeled Images/Uninfected'
unin_data_list = list(paths.list_images(unin_data))

fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(11, 6))

for i in range(4):
  img_para = cv.imread(para_data_list[i])
  output_para = cv.resize(img_para, (128,128))
  ax[0, i].imshow(output_para)
  ax[0, i].set_title('Terinfeksi')
  ax[0, i].axis(False)

  img_unin = cv.imread(unin_data_list[i])
  output_unin = cv.resize(img_unin, (50,50))
  ax[1, i].imshow(output_unin)
  ax[1, i].set_title('Tidak Terinfeksi')
  ax[1, i].axis(False)
```
![pictureafte](/images/setafter.jpg)
```python
# memisahkan data dan label
img_dir = 'malaria_cell_images/True Labeled Images'

file_name = []
label = []
full_img_dir = []

for path, subdirs, files in os.walk(img_dir):
  for name in files:
    full_img_dir.append(os.path.join(path, name))
    label.append(path.split('/')[-1])
    file_name.append(name)

# melihat jumlah data per label(kategori)
df = pd.DataFrame({"path":full_img_dir,'file_name':file_name,"label":label})
df.groupby(['label']).size()
```
![jumlahafter](/images/jumlahdata.jpg)
> Dari kedua dataset cell images tersebut, dapat dilihat bahwa pada data belum dilakukan *cleaning* terdapat misslabel yang terdapat pada gambar baris pertama kolom pertama, gambar tersebut seharusnya masuk ke label *Uninfected* karena tidak memiliki ciri-ciri *Parasitized* yang ditandakan tidak ditemukannya bulir parasit pada sel, kemudian pada gambar baris kedua kolom ke ketiga, seharusnya gambar tersebut masuk ke label *Parasitized*  karena tidak memiliki ciri *Uninfected* yang ditandakan dengan ditemukannya bulir parasit pada sel.

### Memisahkan Data Training, Validation dan Testing
```python
from sklearn.model_selection import train_test_split

# variabel yang digunakan pada pemisahan dataset
X= df['path']
y= df['label']

# membagi dataset menjadi data train dan data test
X_train, X_tesv, y_train, y_tesv = train_test_split(X, y, test_size=0.10, random_state=42)

# membagi data test menjadi data test dan data validation
X_test, X_val, y_test, y_val = train_test_split(X_tesv, y_tesv, test_size=0.5, random_state=42)

# menyatukan dataset kedalam masing-masing dataframe
df_train = pd.DataFrame({'path':X_train, 'label':y_train, 'set':'train'})
df_test = pd.DataFrame({'path':X_test, 'label':y_test, 'set':'test'})
df_val = pd.DataFrame({'path':X_val, 'label':y_val, 'set':'validation'})

# melihat jumlah masing masing dataset
print('Jumlah Data Training, Validation dan Testing:')
print('- train size \t:', len(df_train))
print('- val size \t:', len(df_val))
print('- test size \t:', len(df_test))
```
![proporsidata](/images/n_trainvaltest.jpg)
```python
# melihat proporsi pada masing masing dataset
data_frame_all = df_train.append([df_test, df_val]).reset_index(drop=1)\

print('===================================================== \n')
print(data_frame_all.groupby(['set','label']).size(),'\n')
print('===================================================== \n')
```
![proporsicategori](/images/n_trainvaltest_perclass.jpg)


> Dari *coding* diatas kita telah membuat proporsi masing-masing dataset yaitu 90% untuk data train, 5% untuk data validation dan 5% untuk data test. Proporsi ini sudah cukup untuk melakukan training pada dataset karena jumlah data gambar *cell malaria* ini cukup lumayan banyak serta nantinya agar menghasilkan prediksi yang lebih akurat.

Selanjutnya memisahkan masing-masing dataset tersebut ke dalam folder di-OS dalam 3 folder yaitu train, validation dan test, dengan menjalankan *code* berikut:

```python
# membuat directory baru untuk data train, val dan test pada colab
import shutil
from tqdm.notebook import tqdm as tq

datasource_path = 'malaria_cell_images/True Labeled Images'
dataset_path = '/content/'

for index, row in tq(data_frame_all.iterrows()):
  # mendeteksi filepath
  file_path = row['path']
  if os.path.exists(file_path) == False:
    file_path = os.path.join(datasource_path, row['label'], row['image'].split('.')[0])
  # membuat folder destination dirs
  if os.path.exists(os.path.join(dataset_path, row['set'], row['label'])) == False:
    os.makedirs(os.path.join(dataset_path, row['set'], row['label']))
  # mendefinisikan file dest
  destination_file_name = file_path.split('/')[-1]
  file_dest = os.path.join(dataset_path, row['set'], row['label'], destination_file_name)
  # mengcopy file dari source ke dest
  if os.path.exists(file_dest) == False:
    shutil.copy2(file_path, file_dest)
```

## Preprocessing Dataset
### Image Augmentation
Pada tahapan ini data gambar dilakukan augmentasi agar gambar tersebut mudah diolah dan diidentifiksi oleh komputer karena pada proses augmentasi ini akan memberikan pola-pola baru pada gambar sehingga membentuk pola pola yang sesuai pada masing-masing kategorinya yang akhirnya akan mempermudah komputer untuk mengklasifikasi gambar tersebut.
```python
# membuat parameter datagenerator pada masing-masing dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   brightness_range=[0.2,1.0],
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 brightness_range=[0.2,1.0],
                                 horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 brightness_range=[0.2,1.0],
                                 horizontal_flip=True)

# membuat variabel iterator pada masing-masing dataset
train_iterator = train_datagen.flow_from_directory('train',
                                                   target_size=(50, 50),
                                                   batch_size=32,
                                                   class_mode='binary')

val_iterator = val_datagen.flow_from_directory('validation',
                                               target_size=(50, 50),
                                               batch_size=32,
                                               class_mode='binary')

test_iterator = test_datagen.flow_from_directory('test',
                                               target_size=(50, 50),
                                               batch_size=32,
                                               class_mode='binary')
```
![jumlahperkategori](/images/kategori.jpg)

> Nilai tuning hyperparameter image augmentation yang digunakan tidak begitu mencolok atau bernilai tinggi karena data gambar cell malaria ini memiliki ragam warna dan bentuk objeknya tidak begitu komplek karena perbedaanya mencoloknya hanya pada ada bulir parasit atau tidak sedangkan warna dan pola lainnya hampir sama.

### Visualisasi Augmentasi Gambar
Pada tahap ini akan menampilkan hasil augmentasi gambar dari dataset, sehingga memberitahu kita seperti apa proses yang terjadi saat augmentasi gambar.
```python
# fungsi menampilkan hasil augmentasi gambar
def img_aug(datagen_param):
  img = load_img(para_data_list[3])
  # konversi gambar ke numpy array
  data = img_to_array(img)
  # memperlluas dimensi dalam satu sampel
  samples = np.expand_dims(data, 0)
  # membuat generator augmentasi gambar
  datagen = datagen_param
  # mempersiapkan iterator
  it = datagen.flow(samples, batch_size=1)
  # menentukan ukuran gambar yang akn ditampilkan
  plt.figure(figsize=(10,10))
  # memproses dan memplot sample
  for i in range(9):
	  # mendefinisikan subplot
	  plt.subplot(330 + 1 + i)
	  # menghasilkan batch gambar
	  batch = it.next()
	  # konversikan ke bilangan bulat tak bertanda untuk dilihat
	  image = batch[0].astype('uint8')
	  # memplot data pixel gambar
	  plt.imshow(image)
  # menampilkan gambar
  plt.show()
```
```python
# augmentasi gambar 'bergeser kearah lebar'
datagen_param = ImageDataGenerator(width_shift_range=0.1)
img_aug(datagen_param)
```
![aug_width](/images/aug_width.jpg)
```python
# augmentasi gambar 'bergeser kearah tinggi'
datagen_param = ImageDataGenerator(height_shift_range=0.1)
img_aug(datagen_param)
```
![aug_height](/images/aug_heigth.jpg)
```python
# augmentasi gambar 'kecerahan'
datagen_param = ImageDataGenerator(brightness_range=[0.2,1.0])
img_aug(datagen_param)
```
![aug_brigness](/images/aug_brigness.jpg)
```python
# augmentasi gambar 'berbalik'
datagen_param = ImageDataGenerator(horizontal_flip=True)
img_aug(datagen_param)
```
![aug_horflip](/images/aug_horflip.jpg)

## Melatih Model
Pada tahap ini, kita akan melatih model agar komputer dapat mengklasifikasi gambar dengan benar. Model arsitektur deep learning yang digunakan untuk klasifikasi cell malaria ini yaitu CNN (*Convulosional Neural Network*) dengan 3 layer CNN parameter filter (32,3,3), (64,3,3) dan (128,3,3) serta ditambah layer maxpooling 2D (2x2) setelah masing-masing layer CNN dan diakhiri layer Dopout (0.2) serta ctivation yang digunakan *relu*. Selanjutnya, pada full conected layer menggunakan layer Flatten, Dense(512), Droupout(0.5) dan Dense(1) dengan activation yaitu *sigmoid*. Adapun optimizer yang digunakan yaitu Adam dengan learning rate = 0.001 karena optimizer ini termasuk bagus untuk klasifikasi gambar jumlah banyak, serta loss yang digunakan yaitu binary_crossentropy karena dipakai untuk klasifikasi dua object. Berikut *code* yang digunakan:
```python
# merancang arsitektur model
np.random.seed(10)

dp_model = Sequential()

dp_model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 3)))
dp_model.add(MaxPooling2D((2, 2)))
dp_model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
dp_model.add(MaxPooling2D((2, 2)))
dp_model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
dp_model.add(MaxPooling2D((2, 2)))
dp_model.add(Dropout(0.2))

dp_model.add(Flatten())
dp_model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
dp_model.add(Dropout(0.5))
dp_model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=0.001)

dp_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

dp_model.summary()
```
![sumary](/images/model_summary.jpg)

Selanjutnya merancang callback untuk memberhentikan training model saat accuracy/loss sudah stabil setelah beberapa epoch (EarlyStopping), menyimpan weights model saat mencapai nilai acuracy tertinggi/loss terendah (ModelCheckpoint) dan menampilkan grafik model accuracy dan loss dalam sebuah *board* (TensorBoard) dan dilanjutkan dengan training data dengan fit.
```python
# merancang callbacks
earlystop = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

filepath = 'weights-best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

logdir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tens_board = TensorBoard(logdir, histogram_freq=1)


callbacks_list = [earlystop, checkpoint, tens_board]

# melatih model
dp_model.fit(x=train_iterator,
        steps_per_epoch=len(train_iterator),
        epochs=50,
        callbacks=callbacks_list,
        validation_data=val_iterator,
        validation_steps=len(val_iterator),
        verbose = 2)
```
![epoch_acc](/images/model_accuracy.jpg)

> Dengan menggunakan arsitektur model CNN dengan 3 layer serta optimizer Adam dan inputan image 50x50 pixel RGB, telah mampu menghasilkan 2,45 juta parameter yang dapat memprediksi cell malaria yang mencapai accuracy 98,8% pada data validation dan accuracy 98,5% pada data training, dengan hal ini menandakan bahwa dengan parameter yang tidak begitu komplek mampu menghasilkan accuracy yang tinggi dan stabil antara data training dan validation (variance dan bias kecil).

## Evaluasi Model
Pada tahapan ini, akan dilakukan evaluasi model berdasarkan hasil prediksi model yang digunakan terhadap data validation dan testing. Pada tahap ini juga kita akan melihat bagaimana proses dari model CNN dengan melihatkan gambar proses di setiap neuron layer CNN baik filter maupun activationnya.
```python
# memanggil model yang dipakai
model_filename = 'weights-best.h5'

dp_model.load_weights(model_filename)
dp_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
```
### Visualisasi CNN Layers
```python
# memanggil satu gambar untuk divisualisasikan
path = para_data_list[100]
img = image.load_img(path, target_size=(50,50))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images_para = np.vstack([x])

# mempersiapkan parameter untuk fungsi visualisasi
layer_outputs = [layer.output for layer in dp_model.layers]
activation_model = Model(inputs=dp_model.input, outputs=layer_outputs)
activations = activation_model.predict(images_para)

# membuat fungsi untuk visualisai cnn layer
def display_activation(activations, col_size, row_size, act_index): 
  activation = activations[act_index]
  activation_index=0
  fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
  for row in range(0,row_size):
    for col in range(0,col_size):
      ax[row][col].imshow(activation[0, :, :, activation_index])
      activation_index += 1

# membuat fungsi untuk menampilkan filter layer
def display_filter(row_size, col_size, filter_index):
  filters, biases = dp_model.layers[filter_index].get_weights()
  # normalisasi filter ke nilai 0-1 agar dapat divisualisasikan
  fil_min, fil_max = filters.min(), filters.max()
  filters = (filters - fil_min) / (fil_max - fil_min)
  # plot first few filters
  n_filters, ix = row_size, 1
  for i in range(n_filters):
	  # mendapatkan nilai filter
	  f = filters[:, :, :, i]
	  # plot channel secara terpisah
	  for j in range(col_size):
		  ax = plt.subplot(n_filters, col_size, ix)
		  ax.set_xticks([])
		  ax.set_yticks([])
		  # memplot channel filter dengan warna default
		  plt.imshow(f[:, :, j])
		  ix += 1
  plt.show()
```
```python
# menampilkan jumlah dan parameter filter layer model CNN
for layer in dp_model.layers:
	# mengecek layer convolusi 
	if 'conv' not in layer.name:
		continue
	# memperoleh filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
```
![n_layers](/images/n_layers.jpg)

Berikut merupakan tampilan filter layer CNN pada index 0 atau layer pertama (CNN 32, 3, 3). Gambar filter tersebut memperlihatkan filter kernel 3x3 dengan nilai yang bervariasi dapat dilihat dari sebaran warna gambar filter. Pada tampilan ini hanya ditampilkan dengan jumlah 8 x 3 agar mempermudah untuk melihat filternya (untuk jumlah aslinya yaitu 32 x 3).

```python
# menampilkan filter channel conv2d_4
display_filter(8, 3, 0)
```
![filter_lay](/images/layer0_filter.jpg)

Berikut merupakan tampilan gambar activation layer CNN index 0 atau layer pertama dan tampilan gambar activation layer maxpooling kedua atau index 1 (layer kedua). Disini ditampilkan gambar mana saja yang aktif dalam suatu neuron saat di layer tersebut.

```python
# menampilkan feature teraktivasi pada neuron layer model conv2d_4
display_activation(activations, 8, 4, 0)
```
![lay_act](/images/layer0_activ.jpg)
```python
# menampilkan feature teraktivasi pada neuron layer model conv2d_4 + maxpooling(2x2)
display_activation(activations, 8, 4, 1)
```
![lay_act_max](/images/layer0_activ_maxpo.jpg)

> Dari visualisasi tampilan gambar filter dapat dilihat bahwa setiap filter memiliki nilai weight yang berbeda serta jumlah filter yang berbeda juga pada setiap layernya. Dari filter-filter ini nantinya yang menentukan weight mana saja yang aktif di setiap layer. Dimana nilai-nilai ini akan tergantung pada berapa kali *backpropagation* yang dilakukan berdasarkan jumlah epoch. Dari beberapa pengulangan inilah nantinya akan nilai weight yang konstan pada setiap kategori label pada data sehingga komputer belajar dari pola-pola tersebut, dengan demikian komputer dapat mengklasifikasikan gambar dengan prediksi yang lebih akurat.

### TensorBoard
Tensorbord ini merupakan media untuk memvisualisasikan hasil dari model deep learning tensorflow yang diterapkan. Salah satunya dapat menampilkan nilai akurasi dan loss antara data training dan data validation per setiap epochnya. Dari grafik ini dapat dilihat bagaimana kualitas model, apakah menghasilkan prediksi yang sudah baik atau tidak yang dapat dilihat dari kestabilan nilai accuracy atau loss antara data training dan validation.
```python
%load_ext tensorboard
%tensorboard --logdir logs
```
![tenbor](/images/tensorboard.jpg)
```python
# akurasi pada masing-masing dataset
iter_data_all = [train_iterator, val_iterator, test_iterator]
str_dataset = ['Training Data', 'Validation Data', 'Testing data']

for i in range(len(str_dataset)):
  loss, acc = dp_model.evaluate(iter_data_all[i], steps=len(iter_data_all[i]), 
                                verbose=0)
  print('Accuracy on {}: {:.4f} \nLoss on {}: {:.4f}'.format(str_dataset[i],
                                                             acc,
                                                             str_dataset[i], 
                                                             loss),'\n')
```
![acc_all](/images/acc_all.jpg)

### Prediksi Terhadap Data Validation
Pada tahap ini akan dilakukan prediksi terhadap dataset validation dengan menerapkannya pada 5 gambar dari masing-masing kelas/label.
```python
# fungsi menampilkan gambar prediksi validation maupun testing
def prediksi_gambar(path, label):
  fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(12, 5))
  for i in range(5):
    dir = path[i]
    tag = label
    img = image.load_img(dir, target_size=(50,50))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = dp_model.predict(images, batch_size=32)

    if classes[0]==0:
      nama = 'Parasitized'
    elif classes[0]==1:
      nama = 'Uninfected'

    ax[i].imshow(img)
    ax[i].set_title('Actual: {}\nPrediksi: {}'.format(tag, nama))
    ax[i].axis(False)

# membuat list dataset validation pada masing-masing kategori
par_path_val = 'validation/Parasitized'
para_val_dt = list(paths.list_images(par_path_val))

un_path_val = 'validation/Uninfected'
unin_val_dt = list(paths.list_images(un_path_val))
```
```python
# prediksi gambar validation Parasitized
path = para_val_dt
label = 'Parasitized'
prediksi_gambar(path, label)
```
![pred_val_para](/images/prad_val_para.jpg)
```python
# prediksi gambar validation Uninfected
path = unin_val_dt
label = 'Uninfected'
prediksi_gambar(path, label)
```
![pred_val_unin](/images/prad_val_unin.jpg)

Dari 5 gambar acak dari masing-masing label pada dataset validation dapat diprediksi dengan benar keseluruhannya dimana nilai actual sama dengan nilai prediksinya. Berarti model sudah cukup baik untuk memprediksi gambar cell malaria.

### Pediksi Terhadap Data Testing
Pada tahap ini model akan memprediksi data testing atau data actual yang benar-benar belum pernah dilihat oleh model sebelumnya.
```python
# membuat list dataset testing pada masing-masing kategori
par_path_test = 'test/Parasitized'
para_test_dt = list(paths.list_images(par_path_test))

un_path_test = 'test/Uninfected'
unin_test_dt = list(paths.list_images(un_path_test))
```
```python
# prediksi gambar testing Parasitized
path = para_test_dt
label = 'Parasitized'
prediksi_gambar(path, label)
```
![pred_test_para](/images/prad_test_para.jpg)
```python
# prediksi gambar testing Uninfected
path = unin_test_dt
label = 'Uninfected'
prediksi_gambar(path, label)
```
![pred_test_unin](/images/prad_test_unin.jpg)

Dari 5 gambar acak dari masing-masing label pada dataset testing dapat diprediksi dengan benar keseluruhannya dimana nilai actual sama dengan nilai prediksinya. Berarti model sudah cukup baik untuk memprediksi gambar cell malaria pada data real atau data yang belum pernah dilihat oleh  model.

### Metrics Dataset Validation and Testing
Pada tahap ini kita akan melihat barapa jumlah yang dapat diprediksi dengan benar pada keseluruhan dataset validation dan dataset testing melalui confolussion matrix dan classification report (accurcy, recall, precission, dan f1_score)

**VALIDATION DATASET**
```python
# menggabungkan masing-masing kategori feature dataset validation dalam satu list
val_dataset = [para_val_dt, unin_val_dt]

feature_val = []

for data in val_dataset:
  for i in data:
    dir = i
    img = image.load_img(dir, target_size=(50,50))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    feature_val.append(images)

# mengubah tipe feature validation menjadi array dan direshape
feature_val = np.array(feature_val)
feature_val = feature_val.reshape(1325, 50, 50, 3)

# menggabungkan masing-masing kategori label dataset validation dalam satu list
label_val = []

for n in val_dataset:
  for i in range(len(n)):
    if n == para_val_dt:
      label_val.append(0)
    else:
      label_val.append(1)

# mengubah tipe feature validation menjadi array dan tipe integer
label_val = np.array(label_val).astype('int32')

# memprediksi label dari feature dataset validation 
val_pred = dp_model.predict(feature_val, batch_size=32)
# mengubah hasil prediksi validation menjadi integer
val_pred = val_pred.astype('int32')
```
```python
# melihat hasil confusion matrix dataset validation
confusion_mtx_val = confusion_matrix(label_val, val_pred)

plt.figure(figsize=(10,8))
sb.heatmap(confusion_mtx_val, annot=True, fmt='d')
plt.title('Confussion Metrics Validation Data\nParasitized : [0] Uninfected : [1]')
plt.show()
```
![cm_val](/images/cm_val.jpg)
```python
# melihat hasil classification report dataset validation
class_report_val = classification_report(label_val, val_pred)
print(class_report_val)
```
![cr_val](/images/cr_val.jpg)

Keterangan:
- Dari classification report dapat dilihat bahwa rata-rata nilai precission, recall, f1-score dan accuracy 0.99 yang berarti hasil prediksinya sangat bagus terhadap data validation.
- Dari confussin metrics dapat dilihat bahwa hasil yang diprediksi salah dari masing-masing class/label berjumlah sedikit yaitu 9 gambar untuk prediksi *Parasitized* dan 10 gambar untuk prediksi *Uninfected*.

**TESTING DATASET**

```python
# menggabungkan masing-masing kategori feature dataset testing dalam satu list
test_dataset = [para_test_dt, unin_test_dt]

feature_test = []

for data in test_dataset:
  for i in data:
    dir = i
    img = image.load_img(dir, target_size=(50,50))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    feature_test.append(images)

# mengubah tipe feature dataset test menjadi array dan direshape
feature_test = np.array(feature_test)
feature_test = feature_test.reshape(1325, 50, 50, 3)

# menggabungkan masing-masing kategori label dataset test dalam satu list
label_test = []

for n in test_dataset:
  for i in range(len(n)):
    if n == para_test_dt:
      label_test.append(0)
    else:
      label_test.append(1)

# mengubah tipe feature dataset test menjadi array dan tipe integer
label_test = np.array(label_test).astype('int32')

# memprediksi label dari feature dataset test
test_pred = dp_model.predict(feature_test, batch_size=32)
# mengubah hasil prediksi dataset test menjadi integer
test_pred = test_pred.astype('int32')
```
```python
# melihat hasil confusion matrix dataset test
confusion_mtx_test = confusion_matrix(label_test, test_pred)

plt.figure(figsize=(10,8))
sb.heatmap(confusion_mtx_test, annot=True, fmt='d')
plt.title('Confussion Metrics Testing Data\nParasitized : [0] Uninfected : [1]')
plt.show()
```
![cm_test](/images/cm_test.jpg)
```python
# melihat hasil classification report dataset test
class_report_test = classification_report(label_test, test_pred)
print(class_report_test)
```
![cr_test](/images/cr_test.jpg)

Keterangan:
- Dari classification report dapat dilihat bahwa rata-rata nilai precission, recall, f1-score dan accuracy 0.99 yang berarti hasil prediksinya sangat bagus terhadap data testing.
- Dari confussin metrics dapat dilihat bahwa hasil yang diprediksi salah dari masing-masing class/label berjumlah sedikit yaitu 4 gambar untuk prediksi *Parasitized* dan 9 gambar untuk prediksi *Uninfected*.

# Kesimpulan:
- Untuk mendapatkan hasil prediksi yang bagus pada klasifiksai gambar *cell malaria* salah satunya cukup menggunakan augmentasi yang sederhana (tidak mendalam), jumlah data training 90% dataset, ukuran target gambar menggunkan ukuran yang lebih kecil sedikit dari ukuran gambar rata-rata dan arsitektur model CNN yang tidak terlalu komplek (cukup 3 layer CNN + 3 layer maxpooling dan 1 layer dropout) serta optimizer Adam (lr=0.001).
- Model deep learning klasifikasi gambar *cell malaria* ini juga sangat bagus untuk memprediksi orang-orang yang terinfeksi karena dilihat dari metrik confussion prediksi salahnya lebih kecil dibandingkan prediksi pada orang-orang yang tidak terinfeksi, sehingga dengan demikian akan mengurangi risiko penyakit yang semakin parah atau kematian pada orang yang diprediksi tidak terinfeksi padahal terinfeksi penyakit malaria.



---

Source Code : [link](https://github.com/ravide-lubis/BC-ML-IGLO_Final_Project)

Prediction Web App : 

---