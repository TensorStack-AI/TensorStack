// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.Common.Pipeline;

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
        public TileMode TileMode { get; init; }

        /// <summary>
        /// The maximum size of the tile.
        /// </summary>
        public int MaxTileSize { get; init; }

        /// <summary>
        /// The tile overlap in pixels to avoid visible seams.
        /// </summary>
        public int TileOverlap { get; init; }
    }
}
