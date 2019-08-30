
def get_iou(preds, gt, thresh=0.5):
    p = preds.clone()
    g = gt.clone()

    p[p < thresh] = 0
    p[p >= thresh] = 1

    s = p + g
    inter_area = s[s == 2].size()[0]
    union_area = s[s > 0].size()[0]

    return inter_area / union_area



