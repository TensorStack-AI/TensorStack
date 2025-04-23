// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.Common.Video;

namespace TensorStack.Upscaler.Common
{
    public sealed record UpscaleStreamOptions : UpscaleOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleStreamOptions"/> class.
        /// </summary>
        /// <param name="stream">The input stream.</param>
        /// <param name="tileMode">Enable/Disable TileMode, splitting image into smaller tiles to save memory.</param>
        /// <param name="maxTileSize">The maximum size of the tile for TileMode</param>
        /// <param name="tileOverlap">The tile overlap in pixels to avoid visible seams.</param>
        public UpscaleStreamOptions(IAsyncEnumerable<VideoFrame> stream, TileMode tileMode = TileMode.None, int maxTileSize = 512, int tileOverlap = 16)
            : base(tileMode, maxTileSize, tileOverlap)
        {
            Stream = stream;
        }

        /// <summary>
        /// Gets the stream input.
        /// </summary>
        public IAsyncEnumerable<VideoFrame> Stream { get; }
    }
}
