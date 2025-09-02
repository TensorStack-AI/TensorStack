// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Processing
{
    public class BOSLogitsProcessor : ILogitsProcessor
    {
        private long _bosTokenId;

        /// <summary>
        /// Initializes a new instance of the <see cref="BOSLogitsProcessor"/> class.
        /// </summary>
        /// <param name="bosTokenId">The bos token identifier.</param>
        public BOSLogitsProcessor(long bosTokenId)
        {
            _bosTokenId = bosTokenId;
        }

        /// <summary>
        /// Processes the specified inputs logita.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="logits">The logits.</param>
        public void Process(List<long> inputs, Tensor<float> logits)
        {
            if (inputs.Count == 0)
            {
                logits.Fill(float.NegativeInfinity);
                logits[0, 0] = float.NegativeZero;
            }
        }
    }
}
