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

"""Various network "heads" for predicting masks in FCN.

The design is as follows:

... -> Feature Map -> fcn head -> fcn output -> loss


The fcn head produces a feature representation of the image for the purpose
of mask prediction. The fcn output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
import utils.blob as blob_utils

# ---------------------------------------------------------------------------- #
# FCN outputs and losses
# ---------------------------------------------------------------------------- #
def add_fcn_outputs(model, blob_in, dim):

    """Add FCN specific outputs: either mask logits or probs."""
    dim_out = 1

    # Predict mask using Conv
    # Use GaussianFill for class-agnostic mask prediction; fills based on
    # fan-in can be too large in this case and cause divergence
    blob_out = model.Conv(
        blob_in,
        'mask_fcn_logits',
        dim,
        dim_out,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=('GaussianFill', {'std': 0.001}),
        bias_init=const_fill(0.0)
    )

    if cfg.FCN.UPSAMPLE_RATIO > 1:
        blob_out = model.BilinearInterpolation(
            'mask_fcn_logits', 'mask_fcn_logits_up', dim_out, dim_out,
            cfg.FCN.UPSAMPLE_RATIO
        )

    if not model.train:  # == if test
        blob_out = model.Softmax(blob_out, 'mask_fcn_probs')
    
    return blob_out

def add_fcn_losses(model, blob_mask):
    """Add FCN specific losses."""
    loss_mask = model.net.SoftmaxWithLoss(
        [blob_mask, 'masks_int32'],
        ['mask_prob', 'loss_mask'],
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, ['loss_mask'])
    model.AddLosses('loss_mask')
    return loss_gradients

# ---------------------------------------------------------------------------- #
# FCN heads
# ---------------------------------------------------------------------------- #
def fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale, num_convs=4):

    """v1up design: 4 * (conv 3x3), convT 2x2."""
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.FCN.ROI_XFORM_METHOD,
        resolution=cfg.FCN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FCN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.FCN.DILATION
    dim_inner = cfg.FCN.DIM_REDUCED

    for i in range(num_convs):
        current = model.Conv(
            current,
            '_[mask]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.FCN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.FCN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner