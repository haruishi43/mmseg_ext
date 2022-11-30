#!/usr/bin/env python3

from ..builder import SEGMENTORS
from .base import BaseJointSegDet


@SEGMENTORS.register_module()
class JointEncoderDecoder(BaseJointSegDet):
    """JointEncoderDecoder"""
