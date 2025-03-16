// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.Upscaler.Common
{
    public record UpscaleVideoOptions : UpscaleOptions
    {
        public UpscaleVideoOptions(VideoTensor input, bool tileMode = false, int maxTileSize = 512, int tileOverlap = 16)
            : base(tileMode, maxTileSize, tileOverlap)
        {
            Input = input;
        }

        public VideoTensor Input { get; }
    }
}
