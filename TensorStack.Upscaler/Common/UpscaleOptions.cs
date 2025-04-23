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
        /// Initializes a new instance of the <see cref="UpscaleOptions"/> class.
        /// </summary>
        /// <param name="tileMode">Enable/Disable TileMode, splitting image into smaller tiles to save memory.</param>
        /// <param name="maxTileSize">The maximum size of the tile for TileMode</param>
        /// <param name="tileOverlap">The tile overlap in pixels to avoid visible seams.</param>
        protected UpscaleOptions(TileMode tileMode = TileMode.None, int maxTileSize = 512, int tileOverlap = 16)
        {
            TileMode = tileMode;
            MaxTileSize = maxTileSize;
            TileOverlap = tileOverlap;
        }

        /// <summary>
        /// Enable/Disable TileMode, splitting image into smaller tiles to save memory.
        /// </summary>
        public TileMode TileMode { get; }

        /// <summary>
        /// The maximum size of the tile.
        /// </summary>
        public int MaxTileSize { get; }

        /// <summary>
        /// The tile overlap in pixels to avoid visible seams.
        /// </summary>
        public int TileOverlap { get; }
    }
}
