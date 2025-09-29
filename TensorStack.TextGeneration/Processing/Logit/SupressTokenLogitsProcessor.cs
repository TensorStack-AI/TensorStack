// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Processing
{
    public class SupressTokenLogitsProcessor : ILogitsProcessor
    {
        private readonly int[] _supressTokens;

        /// <summary>
        /// Initializes a new instance of the <see cref="SupressTokenLogitsProcessor"/> class.
        /// </summary>
        /// <param name="bosTokenId">The bos token identifier.</param>
        public SupressTokenLogitsProcessor(int[] supressTokens)
        {
            _supressTokens = supressTokens;
        }

        /// <summary>
        /// Processes the specified inputs logita.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="logits">The logits.</param>
        public void Process(List<long> inputs, Tensor<float> logits)
        {
            foreach (var suppressToken in _supressTokens)
            {
                logits[0, suppressToken] = float.MinValue;
            }
        }
    }
}
