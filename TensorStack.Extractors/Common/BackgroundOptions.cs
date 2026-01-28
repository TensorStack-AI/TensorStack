// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;

namespace TensorStack.Extractors.Common
{
    /// <summary>
    /// Default BackgroundOptions.
    /// </summary>
    public record BackgroundOptions : IRunOptions
    {
        /// <summary>
        /// Gets a value indicating whether the output is inverted.
        /// </summary>
        public BackgroundMode Mode { get; init; }

        public bool IsTransparentSupported { get; init; } = true;
        public float MaskFill { get; set; } = 1;
    }


    public record BackgroundImageOptions : BackgroundOptions
    {
        /// <summary>
        /// Gets the input.
        /// </summary>
        public ImageTensor Image { get; init; }
    }


    public enum BackgroundMode
    {
        MaskBackground = 0,
        MaskForeground = 1,

        RemoveBackground = 10,
        RemoveForeground = 11,
    }
}
