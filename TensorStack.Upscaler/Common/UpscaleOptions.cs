// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Upscaler.Common
{
    /// <summary>
    /// Default UpscaleOptions.
    /// </summary>
    public abstract record UpscaleOptions : IRunOptions
    {
        /// <summary>
        /// Enable/Disable TileMode, splitting image into smaller tiles to save memory.
        /// </summary>
        public bool IsTileEnabled { get; set; }

        /// <summary>
        /// The maximum size of the tile.
        /// </summary>
        public int MaxTileSize { get; init; }

        /// <summary>
        /// The tile overlap in pixels to avoid visible seams.
        /// </summary>
        public int TileOverlap { get; init; }
    }



    /// <summary>
    /// Image UpscaleOptions.
    /// </summary>
    public sealed record UpscaleImageOptions : UpscaleOptions
    {
        /// <summary>
        /// Gets the image input.
        /// </summary>
        public ImageTensor Image { get; init; }
    }



    /// <summary>
    /// Video UpscaleOptions.
    /// </summary>
    public sealed record UpscaleVideoOptions : UpscaleOptions
    {
        /// <summary>
        /// Gets the video input.
        /// </summary>
        public VideoTensor Video { get; init; }
    }



    /// <summary>
    /// Stream UpscaleOptions.
    /// </summary>
    public sealed record UpscaleStreamOptions : UpscaleOptions
    {
        /// <summary>
        /// Gets the stream input.
        /// </summary>
        public IAsyncEnumerable<VideoFrame> Stream { get; init; }
    }
}
