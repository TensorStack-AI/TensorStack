// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.Upscaler.Common
{
    public record UpscaleImageOptions : UpscaleOptions
    {
        public UpscaleImageOptions(ImageTensor input, bool tileMode = false, int maxTileSize = 512, int tileOverlap = 16)
            : base(tileMode, maxTileSize, tileOverlap)
        {
            Input = input;
        }

        public ImageTensor Input { get; }
    }
}
