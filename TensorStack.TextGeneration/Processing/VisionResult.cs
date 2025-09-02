// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Processing
{
    public record VisionResult(Tensor<float> Embeds, Tensor<long> Mask);
}
