// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.Upscaler.Common
{
    public record UpscaleImageOptions : UpscaleOptions
    {
        /// <summary>
        /// Gets the image input.
        /// </summary>
        public ImageTensor Image { get; init; }
    }
}
