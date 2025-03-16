// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Video;

namespace TensorStack.Extractors.Common
{
    public record ExtractorStreamOptions : ExtractorOptions
    {
        public ExtractorStreamOptions(IAsyncEnumerable<VideoFrame> input, bool mergeInput = false, bool tileMode = false, int maxTileSize = 512, int tileOverlap = 16)
            : base(mergeInput, tileMode, maxTileSize, tileOverlap)
        {
            Input = input;
        }

        public IAsyncEnumerable<VideoFrame> Input { get; }
    }
}
