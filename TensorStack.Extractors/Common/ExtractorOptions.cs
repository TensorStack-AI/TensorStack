// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Extractors.Common
{
    /// <summary>
    /// Default ExtractorOptions.
    /// </summary>
    public record ExtractorOptions : IRunOptions
    {
        /// <summary>
        /// Megre the input and output result into a new tensor.
        /// </summary>
        public bool MergeInput { get; init; }

        /// <summary>
        /// Enable/Disable TileMode, splitting image into smaller tiles to save memory.
        /// </summary>
        public TileMode TileMode { get; init; }

        /// <summary>
        /// The maximum size of the tile.
        /// </summary>
        public int MaxTileSize { get; init; }

        /// <summary>
        /// The tile overlap in pixels to avoid visible seams.
        /// </summary>
        public int TileOverlap { get; init; }

        /// <summary>
        /// Gets a value indicating whether the output is inverted.
        /// </summary>
        public bool IsInverted { get; init; }
    }


    /// <summary>
    /// Image ExtractorOptions.
    /// </summary>
    public record ExtractorImageOptions : ExtractorOptions
    {
        /// <summary>
        /// Gets the image.
        /// </summary>
        public ImageTensor Image { get; init; }
    }


    /// <summary>
    /// Video ExtractorOptions.
    /// </summary>
    public record ExtractorVideoOptions : ExtractorOptions
    {
        /// <summary>
        /// Gets the video.
        /// </summary>
        public VideoTensor Video { get; init; }
    }


    /// <summary>
    /// Stream ExtractorOptions.
    /// </summary>
    public record ExtractorStreamOptions : ExtractorOptions
    {
        /// <summary>
        /// Gets the stream.
        /// </summary>
        public IAsyncEnumerable<VideoFrame> Stream { get; init; }
    }
}
