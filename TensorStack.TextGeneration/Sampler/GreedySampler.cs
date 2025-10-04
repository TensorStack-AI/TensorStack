// Copyright(c) TensorStack.All rights reserved. // Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Sampler
{
    public class GreedySampler : SamplerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GreedySampler"/> class.
        /// </summary>
        /// <param name="options">The options.</param>
        public GreedySampler(GenerateOptions options)
            : base(options) { }


        /// <summary>
        /// Samples the specified logits.
        /// </summary>
        /// <param name="logits">The logits.</param>
        /// <param name="topK">The top k.</param>
        /// <param name="topP">The top p.</param>
        /// <param name="temperature">The temperature.</param>
        /// <returns>LogitResult[].</returns>
        public override LogitResult[] Sample(Tensor<float> logits, int topK = 1, float topP = 1f, float temperature = 1f)
        {
            ApplyTemperature(logits, temperature);
            var topkLogits = SelectTopK(logits, topK);
            var probabilities = GetProbabilities(topkLogits);
            var candidates = SelectTopP(probabilities, topP);
            return candidates.ToArray();
        }

    }
}