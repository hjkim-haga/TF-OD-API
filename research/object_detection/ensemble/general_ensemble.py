""" 
사물 인식을 위한 앙상블 기법.

여럿의 인식 모델의 결과를 종합한다.
"""


def get_coords(box):
    """물체 주위 bbox의 좌상단 우하단 좌표를 구한다.
    
    :param box: [box_x, box_y, box_w, box_h, class, confidence].
    :return: x0은 상자 좌측, x1는 우측, y0은 상단, y1는 하단 좌표를 의미.
    """
    x0 = float(box[0]) - float(box[2]) / 2
    x1 = float(box[0]) + float(box[2]) / 2
    y0 = float(box[1]) - float(box[3]) / 2
    y1 = float(box[1]) + float(box[3]) / 2
    return x0, x1, y0, y1


def compute_iou(box1, box2):
    """두 bbox 사이 IOU를 계산한다.
    
    :param box1: [box_x, box_y, box_w, box_h, class, confidence]로 나타낸 bbox 정보 1
    :param box2: [box_x, box_y, box_w, box_h, class, confidence]로 나타낸 bbox 정보 2
    :return: 교집합 넓이 / 합집합 넓이
    """
    x11, x12, y11, y12 = get_coords(box1)
    x21, x22, y21, y22 = get_coords(box2)
    
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


def general_ensemble(dets, iou_thresh=0.5, weights=None):
    """일반 앙상블
    같은 클래스이면서 영역이 겹치는 bbox들을 찾는다.
    확신도를 더하면서 이들의 위치의 평균을 구한다.
    다른 가중치 설정을 지닌 디텍터들을 weigh 할 수 있다.
    가중치와 IOU thresh는 최적화될 수 있긴 해도 이 스크립트에 실제 학습은 없다.

    :param dets: 검출들의 목록. 각 검출은 하나의 디텍터에서 출력된 모든 결과.
    dim0는 각 디텍터를 의미, 이후에는 각각의 디텍터들의 검출 결과.
    bbox들의 리스트 형식이며, 각 box는 [box_x, box_y, box_w, box_h, class, confidence]를 포함한다.
    box_x와 box_y는 좌표의 중점이며, box_w와 box_h는 각각 너비와 높이이다.
    int인 class를 제외한 나머지는 float이다.
    :param iou_thresh: 두 상자가 같은 클래스라고 판별해 같은 것으로 여겨지는 IOU의 문턱값.
    :param weights: 가중치(NN의 가중치가 아님)들의 리스트.
    어떤 디텍터들이 다른 것보다 더 믿을만하도 여겨지는지 설명한다.
    리스트의 길이는 검출들의 수와 동일하다. 혹시 None으로 설정됐다면 모든 디텍터들을 같은 정도로 믿을
    수 있다고 간주한다. 가중치들 합이 반드시 1이 될 필요는 없다.
    :return: 입력과 같은 형식인 bbox들의 리스트. 확신도 범위는 [0, 1]이다.
    """
    assert (type(iou_thresh) == float)
    
    ndets = len(dets)
    
    # 가중치 없으면 모두 동일하게 취급.
    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert (len(weights) == ndets)
        
        # 정규화
        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s
    
    out = list()  # 이 함수의 return.
    used = list()
    
    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue
            
            used.append(box)
            # 다른 디텍터 결과를 살펴 나가면서 같은 클래스이면서 겹치는 box를 찾는다.
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]
                if odet == det:
                    continue
                
                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    # 미사용인 경우
                    if not obox in used:
                        # Same class
                        if box[4] == obox[4]:
                            iou = compute_iou(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox
                
                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)
            
            # 이제 다른 모든 디텍터들을 다 살펴 본 상태.
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)
                
                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0
                
                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]  # [0]: 박스 정보, [1]: 가중치
                    wsum += w
                    
                    b = bb[0]
                    xc += w * b[0]
                    yc += w * b[1]
                    bw += w * b[2]
                    bh += w * b[3]
                    conf += w * b[5]
                
                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum
                
                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out


if __name__ == "__main__":
    # `weights`는 hparam. 모델의 표현력(test mAP를 기준으로 해서)에 따라 직접 설정.
    weights = [1.0, 0.1, 0.5]
    dets = [
        [[0.1, 0.1, 1.0, 1.0, 0, 0.9],  # detector 1의 결과
         [1.2, 1.4, 0.5, 1.5, 0, 0.9]],
        [[0.2, 0.1, 0.9, 1.1, 0, 0.8]],  # detector 2의 결과
        [[5.0, 5.0, 1.0, 1.0, 0, 0.5]]  # detector 3의 결과
    ]
    
    ens = general_ensemble(dets, weights=weights)
    print(ens)
