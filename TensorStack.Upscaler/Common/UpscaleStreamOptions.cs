// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Video;

namespace TensorStack.Upscaler.Common
{
    public record UpscaleStreamOptions : UpscaleOptions
    {
        public UpscaleStreamOptions(IAsyncEnumerable<VideoFrame> input, bool tileMode = false, int maxTileSize = 512, int tileOverlap = 16)
            : base(tileMode, maxTileSize, tileOverlap)
        {
            Input = input;
        }

        public IAsyncEnumerable<VideoFrame> Input { get; }
    }
}
