// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Tensor;

namespace TensorStack.Florence.Processing
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
        public void Process(List<long>[] inputs, Tensor<float> logits)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var inputIds = inputs[i];
                if (inputIds.Count == 0)
                {
                    logits.GetBatchAsSpan(i).Fill(float.NegativeInfinity);
                    logits[i, 0] = float.NegativeZero;
                }
            }
        }
    }
}
