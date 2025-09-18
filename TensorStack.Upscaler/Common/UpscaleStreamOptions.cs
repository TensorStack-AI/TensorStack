// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Video;

namespace TensorStack.Upscaler.Common
{
    public record UpscaleStreamOptions : UpscaleOptions
    {
        /// <summary>
        /// Gets the stream input.
        /// </summary>
        public IAsyncEnumerable<VideoFrame> Stream { get; init; }
    }
}
