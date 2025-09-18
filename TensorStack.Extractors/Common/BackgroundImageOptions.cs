// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;

namespace TensorStack.Extractors.Common
{
    /// <summary>
    /// Default ExtractorOptions.
    /// </summary>
    public record BackgroundImageOptions : IRunOptions
    {
        /// <summary>
        /// Gets a value indicating whether the output is inverted.
        /// </summary>
        public BackgroundMode Mode { get; init; }

        /// <summary>
        /// Gets the input.
        /// </summary>
        public ImageTensor Input { get; init; }
    }

    public enum BackgroundMode
    {
        MaskBackground = 0,
        MaskForeground = 1,

        RemoveBackground = 10,
        RemoveForeground = 11,
    }
}
