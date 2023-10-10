import warnings
import sys
import os
from random import shuffle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import nibabel as nib


def normalize_data(data):
    """
    Zero Mean Unit Variance
    """
    mean = (data.mean(axis=(1, 2, 3)))
    std = (data.std(axis=(1, 2, 3)))
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]

    return data

def prepare_augdata(subjects_to_read, file_names):
    data, labels = obtain_data(subjects_to_read, file_names=file_names)
    return data, labels

#

def obtain_data(subjects_to_read, file_names):


    All_data = []
    All_tumor_mask = []


    for index in range(len(subjects_to_read)):

        [folder, file] = os.path.split(subjects_to_read[index])
        [T, subject_ID] = os.path.split(folder)

        print ('-------------------------------------------')
        print ('Reading Subject #', index, subject_ID)
        print ('-------------------------------------------')

        for ind in range(len(file_names)):

            XX1 = file_names[ind].split('.')
            XX2 = XX1[0].split('_')
            number = XX2[len(XX2) - 1]

            T2_path = os.path.join(folder, 'Data_aug', 'DA_T2_images',
                                   ('T2_DA_' + str(number) + '.nii.gz'))

            if os.path.exists(T2_path):

                T2_path = os.path.join(folder, 'Data_aug', 'DA_T2_images',
                                       ('T2_DA_' + str(number) + '.nii.gz'))

                WT_mask = os.path.join(folder, 'Data_aug', 'DA_WT_masks',
                                       ('TM_DA_' + str(number) + '.nii.gz'))
            else:

                T2_path = os.path.join(folder, (subject_ID + '_t2_n4.nii.gz'))
                WT_mask = os.path.join(folder, (subject_ID + '_seg_WT.nii.gz'))


            if os.path.exists(T2_path) and os.path.exists(WT_mask):

                print ('    Reading type', file_names[ind])

                T2 = nib.load(T2_path)
                T2 = np.squeeze(np.array(T2.get_data()))
                a, b, c = T2.shape
                T2 = normalize_data(T2.reshape(1, a, b, c))
                T2 = T2.reshape(a, b, c)

                TM = nib.load(WT_mask)
                TM = TM.get_data()
                TM[TM>0]=1

                TM = TM.astype(np.uint8)

                for j in range(c):

                    temp_data = T2[:, :, j]
                    temp_mask = TM[:, :, j]

                    # Basic augmentation
                    D2 = np.fliplr(temp_data)
                    D3 = np.flipud(temp_data)
                    D4 = np.fliplr(D3)


                    All_data.append(temp_data)
                    All_data.append(D2)
                    All_data.append(D3)
                    All_data.append(D4)


                    if np.any(temp_mask != 0):

                        All_tumor_mask.append(1)
                        All_tumor_mask.append(1)
                        All_tumor_mask.append(1)
                        All_tumor_mask.append(1)

                    else:

                        All_tumor_mask.append(0)
                        All_tumor_mask.append(0)
                        All_tumor_mask.append(0)
                        All_tumor_mask.append(0)

    combined = list(zip(All_data, All_tumor_mask))
    shuffle(combined)
    All_data[:], All_tumor_mask[:] = zip(*combined)

    combined = []

    return All_data, All_tumor_mask


def prepare_data(subject_flist, cases_to_read):

    data = []
    labels = []

    for index in range(cases_to_read):

        [folder, file] = os.path.split(subject_flist[index])
        [QQ, subject_ID] = os.path.split(folder)

        print_statement = '    Reading Subject number' + ' ' + str(index) + ' ' + 'with subject ID -->> ' + subject_ID
        print(print_statement)

        T2_file = subject_flist[index]

        if os.path.exists(T2_file):

            T2 = nib.load(T2_file)
            T2 = T2.get_data()
            T2 = np.float16(T2)
            a, b, c = T2.shape
            T2 = normalize_data(T2.reshape(1, a, b, c))
            T2 = T2.reshape(a, b, c)

            folder, file = os.path.split(T2_file)
            tumor_mask_name = file.replace('_t2', '_seg')
            Mask = os.path.join(folder, tumor_mask_name)
            TM = nib.load(Mask)
            TM = TM.get_data()
            TM[TM > 0] = 1
            TM = np.uint8(TM)

            for ind in range(c):
                temp_mask = TM[:, :, ind]

                temp_data_T2 = T2[:, :, ind]
                data.append(temp_data_T2)

                # Flipping Left to right
                T2_L_R = np.fliplr(temp_data_T2)
                data.append(T2_L_R)

                # Flipping upside down
                T2_UD = np.flipud(temp_data_T2)
                data.append(T2_UD)

                # Flipping upside down to Left to right
                # T2_UD_LR = np.fliplr(T2_UD)
                # data.append(T2_UD_LR)

                if np.any(temp_mask != 0):
                    label_val = 1
                    labels.append(label_val)
                    labels.append(label_val)
                    labels.append(label_val)
                    # labels.append(label_val)
                else:
                    label_val = 0
                    labels.append(label_val)
                    labels.append(label_val)
                    labels.append(label_val)
                    # labels.append(label_val)

    return data, labels



def data_augmentation(data_generator, Data, Labels, augmentation_steps_per_slice):

    Samples = np.expand_dims(np.array(Data), axis=1)
    Sample_labels = np.expand_dims(np.array(Labels), axis=1)
    a, b, c, d = Samples.shape
    Augmentor = data_generator.flow(Samples, Sample_labels, batch_size=a)

    for ind in range(augmentation_steps_per_slice):
        # print (ind)
        batch, batch_labels = Augmentor.next()
        Samples = np.concatenate((Samples, batch), axis=0)
        Sample_labels = np.concatenate((Sample_labels, batch_labels), axis=0)

    FINAL_DATA = np.squeeze(Samples)
    FINAL_LABEL = np.squeeze(Sample_labels)

    return FINAL_DATA, FINAL_LABEL

# Where to save the figures
# PROJECT_ROOT_DIR = '/project/radiology/ANSIR_lab/shared/s204656_workspace/project1/Trained_models/5foldCV_DA_BCW_RFtuned'
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "output_images")
# if not os.path.exists(IMAGES_PATH):
#     os.makedirs(IMAGES_PATH)
# os.makedirs(IMAGES_PATH, exist_ok=True)

#
# def save_fig(fig_id, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print('Saving figure:', fig_id)
#     plt.savefig(path, format=fig_extension, dpi=resolution)
#     print ('------------------------------------')
#     print ("Saving figure completed....")
#     print ('------------------------------------')


def evaluate(model, test_features, test_labels):
    test_samples, test_x, test_y = test_features.shape
    test_pred = model.predict( test_features.reshape( test_samples, test_x * test_y ) )

    accuracy = accuracy_score( test_pred, test_labels )
    precision_value = precision_score( test_pred, test_labels )
    recall_value = recall_score( test_pred, test_labels )

    print ('----------Model Performance-----------')
    print ('--------------------------------------')
    print ('Accuracy  : {:0.2f}%.'.format( accuracy ))
    print ('Precision_Value  : {:0.2f}%.'.format( precision_value ))
    print ('Recall_Value  : {:0.2f}%.'.format( recall_value ))
    print ('--------------------------------------')

    return accuracy, precision_value, recall_value
