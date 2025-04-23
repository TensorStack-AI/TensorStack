// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.Upscaler.Common
{
    public sealed record UpscaleVideoOptions : UpscaleOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleOptions"/> class.
        /// </summary>
        /// <param name="video">The video input.</param>
        /// <param name="tileMode">Enable/Disable TileMode, splitting image into smaller tiles to save memory.</param>
        /// <param name="maxTileSize">The maximum size of the tile for TileMode</param>
        /// <param name="tileOverlap">The tile overlap in pixels to avoid visible seams.</param>
        public UpscaleVideoOptions(VideoTensor video, TileMode tileMode = TileMode.None, int maxTileSize = 512, int tileOverlap = 16)
            : base(tileMode, maxTileSize, tileOverlap)
        {
            Video = video;
        }

        /// <summary>
        /// Gets the video input.
        /// </summary>
        public VideoTensor Video { get; }
    }
}
