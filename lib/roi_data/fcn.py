# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Construct minibatches for FCN training. Handles the minibatch blobs
that are specific to FCN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from core.config import cfg
import modeling.FPN as fpn
import utils.blob as blob_utils
import utils.segms as segm_utils

logger = logging.getLogger(__name__)

def get_fcn_blob_names(is_training=True):
    blob_names = []

    if is_training:
        blob_names += ['masks_int32']

        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
            # Support for FPN multi-level rois without bbox reg isn't
            # implemented (... and may never be implemented)
            k_max = cfg.FPN.ROI_MAX_LEVEL
            k_min = cfg.FPN.ROI_MIN_LEVEL
            for lvl in range(k_min, k_max + 1):
                blob_names += ['mask_rois_fpn' + str(lvl)]
            blob_names += ['mask_rois_idx_restore_int32']
    
    return blob_names

def add_fcn_blobs(blobs, im_scales, roidb):

    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        fcn_blobs = _gen_blobs(entry, im_scales[im_i], im_i)
        for k, v in fcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    return True


def _gen_blobs(entry, im_scale, batch_idx):

    """Add Mask R-CNN specific blobs to the input blob dictionary."""
    M = cfg.MRCNN.RESOLUTION

    selected_inds = np.where(entry['gt_classes'] > 0)[0]

    polys = [entry['segms'][i] for i in selected_inds]

    # Class labels and bounding boxes for the polys
    mask_class_labels = entry['gt_classes'][selected_inds]
    mask_rois = np.array(entry['boxes'][selected_inds], dtype='float32')

    # add mask polys
    masks = blob_utils.zeros((selected_inds.shape[0], M**2), int32=True)
    for i in range(len(polys)):
        # Rasterize the polygon mask to an M x M class labels image
        poly_gt = polys[i]
        mask_roi = mask_rois[i]
        mask_class_label = mask_class_labels[i]
        mask = segm_utils.polys_to_mask_wrt_box(poly_gt, mask_roi, M)
        mask = mask_class_label * np.array(mask > 0, dtype=np.int32)
        masks[i, :] = np.reshape(mask, M**2)

    blob_dict = {}
    blob_dict['masks_int32'] = masks

    return blob_dict

def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    rois_blob_name = 'mask_rois'

    """Distribute rois over the different FPN levels."""
    # Get target level for each roi
    # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
    # the box coordinates from columns 1:5
    target_lvls = fpn.map_rois_to_fpn_levels(
        blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
    )
    # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
    fpn.add_multilevel_roi_blobs(
        blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
        lvl_max
    )