import dlib
import pandas as pd
# import cv2
import ast
# import matplotlib.pyplot as plt
from skimage import io


data = pd.read_csv('processed.csv')

data1 = pd.read_csv('TFOD_CSV.csv')

# path = data['image_id'][0]
# path = path + '.jpg'
#
# bbox = ast.literal_eval(data['bbox'][0])
#
# img = cv2.imread(f'train/{path}')
#
# img1 = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2])+ int(bbox[0]), int(bbox[3])+int(bbox[1])), (255, 0,0), 2)
#
# cv2.imshow('image', img1)
#
# cv2.waitKey(0)
# # plt.imshow(img)
#
# data['bbox_processed'] = data['bbox'].apply(lambda x: [ast.literal_eval(x)[0], ast.literal_eval(x)[1],ast.literal_eval(x)[0] +ast.literal_eval(x)[2], ast.literal_eval(x)[1] + ast.literal_eval(x)[3]])
# data.to_csv('processed.csv', index=False)


# data['bbox_processed'] = data['bbox_processed'].apply(lambda x: ast.literal_eval(x))

# data['images'] = data['image_id'].apply(lambda x: 'train/' + str(x) + '.jpg')

options = dlib.simple_object_detector_training_options()
img = list(data['images'])
annotation = list(data['bbox'])

boxes = []
images = []
for i in annotation:
    boxes.append(list(map(int, ast.literal_eval(i))))

for im in img:
    images.append(io.imread(im))


options.be_verbose = True
detector = dlib.train_simple_object_detector(images, boxes, options)

detector.save('wheat_detection')