

from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image

dataset_path = './dataset/archive/Test Images 3360x32x32/test/'
files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

# print(files)

labels = []
images = np.zeros((len(files), 32, 32))

c = 0
for file in files:
    id = file.split('.')[0].split('_')[1]
    label = file.split('.')[0].split('_')[-1]
    labels.append(int(label))
    image = Image.open(dataset_path + file)
    images[c, :, :] = (np.asarray(image))
    c = c + 1


print(images.shape)
image2 = Image.fromarray(images[90, :, :].reshape(32, 32))

image2.show()

a = np.asarray(labels)

for i in a:
    print(i)

exit(0)




def maked_one_hot_vector():
    dataset_path='./dataset/archive/train images 13440x32x32/train/'
    files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    #print(files)

    labels=[]
    images=np.zeros((len(files),32,32))

    c=0
    for file in files:

        id = file.split('.')[0].split('_')[1]
        label= file.split('.')[0].split('_')[-1]
        labels.append(int(label))
        image = Image.open(dataset_path+file)
        images[c,:,:]=( np.asarray(image))
        c=c+1

    a=np.asarray(labels)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    btch=100
    total=b.shape[0]


    num_btchs=int(total/btch)
    start=0
    end=btch


    new_label_gt = np.zeros((num_btchs,btch,b.shape[1]))
    new_img_gt = np.zeros((num_btchs,btch,32,32))
    for n in range(num_btchs):



        new_label_gt[n,:,:]=b[start:end,:]
        new_img_gt[n, :, :,:] = images[start:end, :,:]
        start = end
        end = end + btch


    print(new_label_gt.shape)
    print(new_img_gt.shape)

    image2 = Image.fromarray(new_img_gt[0,80,:,:].reshape(32,32))

    image2.show()

    print(new_label_gt[0,80,:])



