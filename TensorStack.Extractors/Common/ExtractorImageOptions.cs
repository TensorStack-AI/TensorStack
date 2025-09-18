// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.Extractors.Common
{
    public record ExtractorImageOptions : ExtractorOptions
    {
        public ImageTensor Input { get; init; }
    }
}
