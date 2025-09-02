// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Tensor;

namespace TensorStack.Transformers.Processing
{
    public interface ILogitsProcessor
    {
        /// <summary>
        /// Processes the specified inputs logita.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="logits">The logits.</param>
        public void Process(List<long> inputs, Tensor<float> logits);
    }
}
