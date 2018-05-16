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

''' seg-eve '''
def concat_cls_score_bbox_pred(model):
    # flatten 'bbox_pred_w'
    # bbox_pred_w has shape (324, 1024), where 324 is (81, 4) memory structure
    # reshape to (81, 4 * 1024)
    model.net.Reshape(
        'bbox_pred_w', ['bbox_pred_w_flat', '_bbox_pred_w_oldshape'],
        shape=(model.num_classes, -1))
    cls_score_bbox_pred, _ = model.net.Concat(
        ['cls_score_w', 'bbox_pred_w_flat'],
        ['cls_score_bbox_pred', '_cls_score_bbox_pred_split_info'], axis=1)
    return cls_score_bbox_pred


def bbox2mask_weight_transfer(model, class_embed, dim_in, dim_h, dim_out):
    bbox2mask_type = str(cfg.MRCNN.BBOX2MASK.TYPE)

    def _mlp_activation(model, inputs, outputs):
        if cfg.MRCNN.BBOX2MASK.USE_LEAKYRELU:
            model.net.LeakyRelu(inputs, outputs, alpha=0.1)
        else:
            model.net.Relu(inputs, outputs)

    if (not bbox2mask_type) or bbox2mask_type == '1_layer':
        mask_w_flat = model.FC(
            class_embed, 
            'mask_fcn_logits_w_flat', 
            dim_in, 
            dim_out,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif bbox2mask_type == '2_layer':
        mlp_l1 = model.FC(
            class_embed, 
            'bbox2mask_mlp_l1', 
            dim_in, 
            dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        _mlp_activation(model, mlp_l1, mlp_l1)
        mask_w_flat = model.FC(
            mlp_l1, 
            'mask_fcn_logits_w_flat',
             dim_h, 
             dim_out,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif bbox2mask_type == '3_layer':
        mlp_l1 = model.FC(
            class_embed, 
            'bbox2mask_mlp_l1', 
            dim_in, 
            dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        _mlp_activation(model, mlp_l1, mlp_l1)
        mlp_l2 = model.FC(
            mlp_l1, 
            'bbox2mask_mlp_l2', 
            dim_h, 
            dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        _mlp_activation(model, mlp_l2, mlp_l2)
        mask_w_flat = model.FC(
            mlp_l2, 
            'mask_fcn_logits_w_flat', 
            dim_h, 
            dim_out,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
    else:
        raise ValueError('unknown bbox2mask_type {}'.format(bbox2mask_type))

    # mask_w has shape (num_cls, dim_out, 1, 1)
    mask_w = model.net.ExpandDims(
        mask_w_flat, 
        'mask_fcn_logits_w', 
        dims=[2, 3])
    return mask_w
    

def cls_agnostic_mlp_branch(model, blob_in, dim_in, num_cls, dim_h=1024):
    fc_mask_head_type = str(cfg.MRCNN.MLP_MASK_BRANCH_TYPE)
    dim_out = 1 * cfg.MRCNN.RESOLUTION**2

    if (not fc_mask_head_type) or fc_mask_head_type == '1_layer':
        raw_mlp_branch = model.FC(
            blob_in, 
            'mask_mlp_logits_raw', 
            dim_in, 
            dim_out,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif fc_mask_head_type == '2_layer':
        mlp_l1 = model.FC(
            blob_in, 
            'fc_mask_head_mlp_l1', 
            dim_in, 
            dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.net.Relu(mlp_l1, mlp_l1)
        raw_mlp_branch = model.FC(
            mlp_l1, 
            'mask_mlp_logits_raw', 
            dim_h, 
            dim_out,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif fc_mask_head_type == '3_layer':
        mlp_l1 = model.FC(
            blob_in, 
            'fc_mask_head_mlp_l1', 
            dim_in, 
            dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.net.Relu(mlp_l1, mlp_l1)
        mlp_l2 = model.FC(
            mlp_l1, 
            'fc_mask_head_mlp_l2', 
            dim_h, 
            dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.net.Relu(mlp_l2, mlp_l2)
        raw_mlp_branch = model.FC(
            mlp_l2, 
            'mask_mlp_logits_raw', 
            dim_h, 
            dim_out,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    else:
        raise ValueError('unknown fc_mask_head_type {}'.format(fc_mask_head_type))

    mlp_branch, _ = model.net.Reshape(
        raw_mlp_branch,
        ['mask_mlp_logits_reshaped', '_mask_mlp_logits_raw_old_shape'],
        shape=(-1, 1, cfg.MRCNN.RESOLUTION, cfg.MRCNN.RESOLUTION))
    if num_cls > 1:
        mlp_branch = model.net.Tile(
            mlp_branch, 'mask_mlp_logits_tiled', tiles=num_cls, axis=1)

    return mlp_branch


def add_mask_rcnn_outputs(model, blob_in, dim):
    """Add Mask R-CNN specific outputs: either mask logits or probs."""
    dim_out = 1

    if cfg.MRCNN.BBOX2MASK.BBOX2MASK_ON:
        # Use weight transfer function iff BBOX2MASK_ON is True
        # Decide the input to the of weight transfer function
        #   - Case 1) From a pre-trained embedding vector (e.g. GloVe)
        #   - Case 2) From the detection weights in the box head
        if cfg.MRCNN.BBOX2MASK.USE_PRETRAINED_EMBED:
            # Case 1) From a pre-trained embedding vector (e.g. GloVe)
            class_embed = cfg.MRCNN.BBOX2MASK.PRETRAINED_EMBED_NAME
            class_embed_dim = cfg.MRCNN.BBOX2MASK.PRETRAINED_EMBED_DIM
            # This parameter is meant to be initialized from a pretrained model
            # instead of learned from scratch. Hence, the default init is HUGE
            # to cause NaN loss so that the error will not pass silently.
            model.AddParameter(model.param_init_net.GaussianFill(
                [], class_embed, shape=[dim_out, class_embed_dim], std=1e12))
            # Pretrained embedding should be fixed during training (it doesn't
            # make sense to update them)
            model.StopGradient(class_embed, class_embed + '_no_grad')
            class_embed = class_embed + '_no_grad'
        else:
            # Case 2) From the detection weights in the box head
            #   - Subcase a) using cls+box
            #   - Subcase b) using cls
            #   - Subcase c) using box
            # where 'cls' is RoI classification weights 'cls_score_w'
            # and 'box' is bounding box regression weights 'bbox_pred_w'
            if (cfg.MRCNN.BBOX2MASK.INCLUDE_CLS_SCORE and
                    cfg.MRCNN.BBOX2MASK.INCLUDE_BBOX_PRED):
                # Subcase a) using cls+box
                concat_cls_score_bbox_pred(model)
                class_embed = 'cls_score_bbox_pred'
                class_embed_dim = 1024 + 4096
            elif cfg.MRCNN.BBOX2MASK.INCLUDE_CLS_SCORE:
                # Subcase b) using cls
                class_embed = 'cls_score_w'
                class_embed_dim = 1024
            elif cfg.MRCNN.BBOX2MASK.INCLUDE_BBOX_PRED:
                # Subcase c) using box; 'bbox_pred_w' need to be flattened
                model.net.Reshape(
                    'bbox_pred_w', ['bbox_pred_w_flat', '_bbox_pred_w_oldshape'],
                    shape=(model.num_classes, -1))
                class_embed = 'bbox_pred_w_flat'
                class_embed_dim = 4096
            else:
                raise ValueError(
                    'At least one of cfg.MRCNN.BBOX2MASK.INCLUDE_CLS_SCORE and '
                    'cfg.MRCNN.BBOX2MASK.INCLUDE_BBOX_PRED needs to be True')
            # Stop the mask gradient to the detection weights if specified
            if cfg.MRCNN.BBOX2MASK.STOP_DET_W_GRAD:
                model.StopGradient(class_embed, class_embed + '_no_grad')
                class_embed = class_embed + '_no_grad'



        # Use weights transfer function to predict mask weights
        mask_w = bbox2mask_weight_transfer(
            model, class_embed, dim_in=class_embed_dim, dim_h=dim, dim_out=dim)
        # Mask prediction with predicted mask weights (no bias term)
        fcn_branch = model.net.Conv(
            [blob_in, mask_w], 'mask_fcn_logits', kernel=1, pad=0, stride=1)
    else:
        # Predict mask using Conv
        # Use GaussianFill for class-agnostic mask prediction; fills based on
        # fan-in can be too large in this case and cause divergence
        # If using class-agnostic mask, scale down init to avoid NaN loss
        init_filler = (
            cfg.MRCNN.CONV_INIT if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill')
            
        fcn_branch = model.Conv(
            blob_in, 
            'mask_fcn_logits', 
            dim, 
            dim_out, 
            kernel=1, 
            pad=0, 
            stride=1,
            weight_init=(init_filler, {'std': 0.001}),
            bias_init=const_fill(0.0))

    # Add a complementary MLP branch if specified
    if cfg.MRCNN.JOINT_FCN_MLP_HEAD:
        # Use class-agnostic MLP branch, and class-aware FCN branch
        mlp_branch = cls_agnostic_mlp_branch(
            model, blob_in, dim_in=dim * cfg.MRCNN.RESOLUTION**2, num_cls=dim_out)
        blob_out = model.net.Add([mlp_branch, fcn_branch], 'mask_logits')
    elif not cfg.MRCNN.USE_FC_OUTPUT:
        blob_out = fcn_branch

    if not model.train:  # == if test
        blob_out = model.Softmax(blob_out, 'mask_fcn_probs')

    return blob_out
''' seg-eve '''


def add_fcn_losses(model, blob_mask):
    """Add FCN specific losses."""
    _, loss_mask = model.net.SoftmaxWithLoss(
        [blob_mask, 'masks_int32'],
        ['mask_prob', 'loss_mask'],
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_mask])
    model.Accuracy(['mask_prob', 'masks_int32'], 'accuracy_mask')
    model.AddLosses('loss_mask')
    model.AddMetrics('accuracy_mask')
    return loss_gradients
    
# ---------------------------------------------------------------------------- #
# FCN heads
# ---------------------------------------------------------------------------- #
def fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale, num_convs=4):

    dilation = cfg.FCN.DILATION
    dim_inner = cfg.FCN.DIM_REDUCED

    for i in range(num_convs):
        current = model.Conv(
            blob_in,
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