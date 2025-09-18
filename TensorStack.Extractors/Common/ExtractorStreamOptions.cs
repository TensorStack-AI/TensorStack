// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.Common.Video;

namespace TensorStack.Extractors.Common
{
    public record ExtractorStreamOptions : ExtractorOptions
    {
        public IAsyncEnumerable<VideoFrame> Input { get; }
    }
}
