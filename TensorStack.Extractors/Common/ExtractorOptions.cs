// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Pipeline;

namespace TensorStack.Extractors.Common
{
    /// <summary>
    /// Default ExtractorOptions.
    /// </summary>
    public record ExtractorOptions : RunOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ExtractorOptions"/> class.
        /// </summary>
        /// <param name="tileMode">Enable/Disable TileMode, splitting image into smaller tiles to save memory.</param>
        /// <param name="maxTileSize">The maximum size of the tile for TileMode</param>
        /// <param name="tileOverlap">The tile overlap in pixels to avoid visible seams.</param>
        public ExtractorOptions(bool mergeInput = false, bool tileMode = false, int maxTileSize = 512, int tileOverlap = 16)
        {
            MergeInput = mergeInput;
            TileMode = tileMode;
            MaxTileSize = maxTileSize;
            TileOverlap = tileOverlap;
        }

        /// <summary>
        /// Megre the input and output result into a new tensor.
        /// </summary>
        public bool MergeInput { get; set; }

        /// <summary>
        /// Enable/Disable TileMode, splitting image into smaller tiles to save memory.
        /// </summary>
        public bool TileMode { get; }

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
