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

"""Various network "heads" for classification prediction.

The design is as follows:


... -> Feature Map -> RoIFeatureXform -> box head -> box cls output -> cls loss

The Classification head produces a feature representation of the RoI for the purpose
of image classification. The box output module convertsthe feature representation 
into classification predictions.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import utils.blob as blob_utils

''' by bacon'''
# ---------------------------------------------------------------------------- #
# clsn R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_clsn_outputs(model, blob_in, dim):

    """Add RoI classification and bounding box regression output ops."""
    model.FC(
        blob_in,
        'cls_logits',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )

    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.net.Sigmoid('cls_logits', 'cls_prob', engine='CUDNN')


def add_clsn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    loss_cls = model.net.SigmoidCrossEntropyLoss(
        ['cls_logits', 'labels_int32'], 
        'loss_cls',
        scale=model.GetLossScale()
    )

    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls])
    model.AddLosses('loss_cls')

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Cls heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):

    # """Add a ReLU MLP with two hidden layers."""
    # hidden_dim = cfg.CLSN.MLP_HEAD_DIM
    # roi_size = cfg.CLSN.ROI_XFORM_RESOLUTION
    # roi_feat = model.RoIFeatureTransform(
    #     blob_in,
    #     'roi_feat',
    #     blob_rois='rois',
    #     method=cfg.CLSN.ROI_XFORM_METHOD,
    #     resolution=roi_size,
    #     sampling_ratio=cfg.CLSN.ROI_XFORM_SAMPLING_RATIO,
    #     spatial_scale=spatial_scale
    # )
    # model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    # model.Relu('fc6', 'fc6')

    model.AveragePool(blob_in, 'average_pool', global_pooling=True)
    model.StopGradient('average_pool', 'average_pool')

    return 'average_pool', dim_in

''' by bacon'''