"""
PYTHONPATH=$PYTHONPATH:/home/ubuntu/myproj/models/:/home/ubuntu/myproj/models/research:/home/ubuntu/myproj/models/research/object_detection python main.py

"""
import importlib
import os
import pickle
import sys
from collections.abc import Iterable

import numpy as np
import tensorflow as tf

import general_ensemble as ensemble
from research.object_detection.custom_utils import images as img_tool

importlib.reload(img_tool)


def load_models(model_list):
    """여러 모델을 한번에 로드한다.

    :param model_list: 모델 경로가 적힌 리스트
    :return: SavedModel에서 로드된 모델들 dict
    """
    models = {}
    for model_path in model_list:
        model = tf.saved_model.load(model_path)
        model_name = os.path.split(model_path)[1]
        models[model_name] = model
    return models


def convert_label_id_to_name(ids, labelmap):
    """인식 결과를 가독성 있는 문자열로 반환.

    :param ids: int 원소들의 리스트. 각 원소는 `labelmap`에 정해진 id이다.
        detector의 결과 중 `pred['detection_classes'][0]`이라고 보면 된다.
    :param labelmap: str 원소들의 리스트. 각 원소는 label명이고, 원소의 인덱스가 `ids`의 id에 대응.
    :return: 물체 이름으로 변환된 인식 결과들.
    """
    return np.array([labelmap[int(lbl_id)] for lbl_id in ids])


def boxes_for_ensemble(pred):
    """detector 검출 결과를 general_ensemble이 요구하는 데이터 형식으로 변환.

    :param dict pred: 검출기 하나의 결과.
        `detection_boxes`, `detection_classes`, `detection_scores` 키를 포함하고 있어야 한다.
        `detection_boxes`: [ymin, xmin, ymax, xmax], 이미지 크기에 상대적인 좌표.
        `detection_scores`: [0, 1]의 confidence score.
        `detection_classes`: int형의 사전에 정의된 사물 id.
    :return:
        [bbox_xcen, bbox_ycen, bbox_width, bbox_height, class_id, score] 형식의 1개 이상의 검출 결과 리스트.
    """
    if isinstance(pred['detection_classes'][0], Iterable):
        bboxes = pred['detection_boxes'][0]  # relative pos
        classes = pred['detection_classes'][0]
        scores = pred['detection_scores'][0]

        det_for_detector = []
        for b, c, s in zip(bboxes, classes, scores):
            ymin = b[0].numpy()
            xmin = b[1].numpy()
            ymax = b[2].numpy()
            xmax = b[3].numpy()
            width = xmax - xmin
            height = ymax - ymin
            xcen = xmin + width / 2
            ycen = ymin + height / 2
            det_for_detector.append([xcen, ycen, width, height, c.numpy(), s.numpy()])
        return det_for_detector

    else:
        bboxes = pred['detection_boxes']  # relative pos
        classes = pred['detection_classes']
        scores = pred['detection_scores']

        det_for_detector = []
        for b, c, s in zip(bboxes, classes, scores):
            ymin = b[0]
            xmin = b[1]
            ymax = b[2]
            xmax = b[3]
            width = xmax - xmin
            height = ymax - ymin
            xcen = xmin + width / 2
            ycen = ymin + height / 2
            det_for_detector.append([xcen, ycen, width, height, c, s])
        return det_for_detector


def boxes_for_drawer(dets_from_ensemble):
    """앙상블 결과를 `draw_boxes` 함수가 요구하는 형식으로 변환.

    :param dets_from_ensemble: 앙상블 결과.
    :return: 단일 detector 모델이 출력하는 형식과 유사함.
    """
    ens_bboxes = []
    ens_classes = []
    ens_scores = []
    for det in dets_from_ensemble:
        # bbox 좌표 재조정.
        xcen, ycen, width, height, label, score = det
        xmin = (xcen - width / 2)
        xmax = (xcen + width / 2)
        ymin = (ycen - height / 2)
        ymax = (ycen + height / 2)
        ens_bboxes.append(np.array([ymin, xmin, ymax, xmax]))
        ens_classes.append(label)
        ens_scores.append(score)
    ens_bboxes = np.array(ens_bboxes)
    ens_classes = np.array(ens_classes)
    ens_scores = np.array(ens_scores)
    return ens_bboxes, ens_classes, ens_scores


def ensemble_detect(image_path, models, weights):
    """사물 인식 모델들의 앙상블 결과를 이미지에 그린다.

    :param image_path: jpeg 이미지 경로.
    :param models: detector들의 dict
    :param weights: mAP가 높은 모델은 중요하게 취급하려고 넣은 모델별 가중치.
    :return: 앙상블 결과 담은 dict 객체, annotation들이 그려진 이미지 두 가지를 tuple로.
        앙상블 결과는 다음과 같다.
            'file_name': 검출 대상 이미지 파일명.
            'ens_bboxes': 앙상블 bbox 좌표(ymin, xmin, ymax, xmax) 리스트,
            'ens_classes': 레이블 id 리스트,
            'ens_scores': 확률값 리스트,
            'preds_voters': 앙상블을 이루던 모델들의 결과 모음(tfhub detector의 출력 형식 따름).
    """
    # 입력 이미지 detector에 맞게 변환.
    if not os.path.exists(image_path):
        print(f'The image does not exist.')
        sys.exit(1)
    
    image_tensor = img_tool.load_img(image_path)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    # 각 detector in `models` 결과 도출하기.
    preds = []
    for k, v in models.items():
        pred = v.signatures['serving_default'](image_tensor)
        preds.append(pred)
    
    # 앙상블 결과 도출하기.
    dets_for_ensemble = []
    for pred in preds:
        dets_for_detector = boxes_for_ensemble(pred)
        dets_for_ensemble.append(dets_for_detector)
    ensemble_results = ensemble.general_ensemble(dets_for_ensemble, weights=weights)
    
    # 앙상블 결과를 그리기 함수에 맞는 형식으로 변환.
    ens_bboxes, ens_classes, ens_scores = boxes_for_drawer(ensemble_results)
    
    # 앙상블 결과의 시각화 - 단점: 재현율이 하락함. 장점: 정밀도가 상승함.
    ens_names = convert_label_id_to_name(ens_classes, labels)
    image_with_bbox = img_tool.draw_boxes(
        image_tensor.numpy()[0], ens_bboxes, ens_names, ens_scores)
    # img_tool.display_image(image_with_bbox)  # 실제 그리기가 필요하면 넣어야 함.
    
    # 각 모델들의 결과 중 필수만 남기고 줄이기.
    lightened_preds = []
    for pred in preds:
        lightened_preds.append({
            'detection_boxes': pred['detection_boxes'],
            'detection_classes': pred['detection_classes'],
            'detection_scores': pred['detection_scores'],
        })
    
    return {
               'file_name': os.path.split(image_path)[1],
               'ens_bboxes': ens_bboxes,
               'ens_classes': ens_classes,
               'ens_scores': ens_scores,
               'preds_voters': lightened_preds,  # 앙상블을 이루던 모델들의 결과 모음.
           }, image_with_bbox


model_dir = '../savedmodels'
model_list = [
    'retinanet_resnet50_v1_fpn_1024x1024',  # 38.3
    'retinanet_resnet101_v1_fpn_1024x1024',  # 39.5
    'efficientdet_d1',  # 38.4
]

model_pathes = [os.path.join(model_dir, model) for model in model_list]
for path in model_pathes:
    if not os.path.exists(path):
        print(f'{path} does not exist.')
        sys.exit(1)

models = load_models(model_pathes)

weights = [38.3, 39.5, 38.4]  # mAP 기준으로 산출

# label map
labels = []
label_map_path = 'coco_labelmap'
with open(label_map_path, 'r') as fin:
    content = fin.readlines()
start_idx = 3
offset = 5
for i in range(91):
    line = content[start_idx + offset * i]
    label = line[17:-2]
    labels.append(label)

# 디렉토리 대상 인식 결과 확인 (ensemble)
outfile = '../test_images/ensemble_result.pickle'
image_dir = '../test_images/ensemble-test'
pred_out_dir = '../test_images/ensemble-result'
ensemble_detections = []  # [i]: 한 장 이미지 내 결과

# 앙상블 모델 추론.
for i, image_path in enumerate(os.scandir(image_dir)):
    image_path = os.path.abspath(image_path)
    preds_on_image, image_with_bbox = ensemble_detect(image_path, models, weights)
    basename = os.path.split(image_path)[1]
    print(f'{i + 1} th inference on {basename} done')
    
    # 앙상블 결과를 이미지 상에 그리기.
    ensemble_detections.append(preds_on_image)
    tf.keras.preprocessing.image.save_img(
        os.path.join(pred_out_dir, basename),
        image_with_bbox)

    # metrics를 위한 detection 파일 생성
    # (image 파일과 동일한 이름으로 txt를 생성해서 그 안에 한 줄마다 하나의 detection 결과를 넣는다.
    file_name = os.path.splitext(preds_on_image['file_name'])[0] + '.txt'
    ens_bboxes = preds_on_image['ens_bboxes']
    ens_classes = preds_on_image['ens_classes']
    ens_scores = preds_on_image['ens_scores']
    lines_for_metrics: list = boxes_for_ensemble({
        'detection_boxes': ens_bboxes,
        'detection_classes': ens_classes,
        'detection_scores': ens_scores,
    })  # [id, score, xc, yc, w, h] in relatively

    with open(os.path.join('../test_images/ensemble-result', file_name), 'w') as fout:
        for line in lines_for_metrics:
            xc, yc, w, h, lbl_id, score = line
            line = f'{int(lbl_id)} {score} {xc} {yc} {w} {h}\n'
            fout.write(line)
        
    
# 앙상블을 이루던 각 모델들 결과를 저장.
with open(outfile, 'wb') as fout:
    pickle.dump(ensemble_detections, fout)
print('Pickle file created for all the detections of images.')

