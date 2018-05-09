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

"""Construct minibatches for CLSN training. Handles the minibatch blobs
that are specific to CLSN.
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

def get_clsn_blob_names(is_training=True):
    """ CLSN blob names. """
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32']

    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']

    return blob_names


def add_clsn_blobs(blobs, im_scales, roidb):

    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _gen_blobs(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
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
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = entry['boxes'] * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    ''' generate fast rcnn, mask rcnn and keypoint rcnn blobs '''
    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=entry['gt_classes'],
        rois=sampled_rois
        )

    return blob_dict


def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    rois_blob_name = 'rois'

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