// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;

namespace TensorStack.Upscaler.Common
{
    public sealed record UpscaleImageOptions : UpscaleOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleImageOptions"/> class.
        /// </summary>
        /// <param name="image">The image input.</param>
        /// <param name="tileMode">Enable/Disable TileMode, splitting image into smaller tiles to save memory.</param>
        /// <param name="maxTileSize">The maximum size of the tile for TileMode</param>
        /// <param name="tileOverlap">The tile overlap in pixels to avoid visible seams.</param>
        public UpscaleImageOptions(ImageTensor image, bool tileMode = false, int maxTileSize = 512, int tileOverlap = 16)
            : base(tileMode, maxTileSize, tileOverlap)
        {
            Image = image;
        }

        /// <summary>
        /// Gets the image input.
        /// </summary>
        public ImageTensor Image { get; }
    }
}
