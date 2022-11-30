#!/usr/bin/env python3

"""Custom EncoderDecoder that subclasses the original version.

- Support methods that depends on passing input image to the head.
- Need to use mmseg dataset class.
- Only supports mmseg registered LOSSES (SEG_LOSSES).
"""

from mmseg.models.segmentors import EncoderDecoder

from ..builder import SEGMENTORS


@SEGMENTORS.register_module()
class CustomEncoderDecoder(EncoderDecoder):
    """Custom Encoder Decoder segmentors.

    - pass input image to decode head
    """

    def __init__(
        self,
        *args,
        pass_input_image=False,
        **kwargs,
    ):
        super(CustomEncoderDecoder, self).__init__(*args, **kwargs)

        # pass input image to decode head
        self.pass_input_image = pass_input_image

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        if self.pass_input_image:
            # HACK: append input
            x = (*x, img)

        return x
