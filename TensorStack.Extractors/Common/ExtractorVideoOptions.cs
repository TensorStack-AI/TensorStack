// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.Extractors.Common
{
    public record ExtractorVideoOptions : ExtractorOptions
    {
        public ExtractorVideoOptions(VideoTensor input, bool mergeInput = false, TileMode tileMode = TileMode.None, int maxTileSize = 512, int tileOverlap = 16)
            : base(mergeInput, tileMode, maxTileSize, tileOverlap)
        {
            Input = input;
        }

        public VideoTensor Input { get; }
    }
}
