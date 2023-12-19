import os.path as osp
import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES


import cv2 
import cv2 as cv


def frequency_filter(img_c, mask):
    """
    频域滤波
    :param img: 图像
    :param mask: 频域掩码
    :return: filtered img: 滤波后图像
    """
    # Fourier trans
    fft = cv.dft(np.float32(img_c), flags=cv.DFT_COMPLEX_OUTPUT)
    fftc = np.fft.fftshift(fft)
    # filter
    fft_filtering = fftc * mask
    # Fourier invtrans
    ifft = np.fft.ifftshift(fft_filtering)
    image_filtered = cv.idft(ifft)
    image_filtered = cv.magnitude(image_filtered[:, :, 0],
                                   image_filtered[:, :, 1])
    return image_filtered


def low_pass_filter(img, rw, rh):
    """
    低通滤波器
    :param img: 输入图像
    :param radius: 通过半径
    :return: 滤波后图像
    """
    rw = int(rw)
    rh = int(rh)
    # size
    h, w = img.shape[:2]
    # mask
    med_h = int(h / 2)
    med_w = int(w / 2)
    mask = np.zeros((h, w, 2), dtype=np.float32)
    mask[med_h - rh:med_h + rh, med_w - rw:med_w + rw] = 1
    B,G,R = cv.split(img)
    Br = frequency_filter(B, mask)
    Gr = frequency_filter(G, mask)
    Rr = frequency_filter(R, mask)
    res = cv.merge([Br, Gr, Rr])
    res = (res - res.min())/(res.max() - res.min()) * 255
    res = res.astype('uint8')
    return res

@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, low_pass=False, r=(300,300)):
        self.to_float32 = to_float32
        self.low_pass = low_pass
        if self.low_pass:
            self.rw, self.rh = r

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = mmcv.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        if self.low_pass:
            results['img'] =low_pass_filter(img, self.rw, self.rh)
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            if results['img_prefix'] is not None:
                file_path = osp.join(results['img_prefix'],
                                     results['img_info']['filename'])
            else:
                file_path = results['img_info']['filename']
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            print("mask_ann:"+str(mask_ann))
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([0, 0, 0, 0], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
