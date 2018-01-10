import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 데이터셋 생성하기(부풀리기, Data Augmentation)
data_aug_gen = ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=15,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8, 2.0],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

img = load_img('handwriting_shape/train/triangle/triangle001.png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
# 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='hard_handwriting_shape/train/triangle',
                               save_prefix='tri', save_format='png'):
    i += 1
    if i >= 15:
        break

i = 0
for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='hard_handwriting_shape/test/triangle',
                               save_prefix='tri', save_format='png'):
    i += 1
    if i >= 5:
        break


img = load_img('handwriting_shape/train/rectangle/rectangle001.png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
# 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='hard_handwriting_shape/train/rectangle',
                               save_prefix='rec', save_format='png'):
    i += 1
    if i >= 15:
        break

i = 0
for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='hard_handwriting_shape/test/rectangle',
                               save_prefix='rec', save_format='png'):
    i += 1
    if i >= 5:
        break


img = load_img('handwriting_shape/train/circle/circle001.png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
# 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='hard_handwriting_shape/train/circle',
                               save_prefix='cir', save_format='png'):
    i += 1
    if i >= 15:
        break

i = 0
for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='hard_handwriting_shape/test/circle',
                               save_prefix='cir', save_format='png'):
    i += 1
    if i >= 5:
        break