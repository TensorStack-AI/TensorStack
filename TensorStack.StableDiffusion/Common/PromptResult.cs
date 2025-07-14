// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.StableDiffusion.Common
{
    public record PromptResult
    {
        private readonly Tensor<float> _promptEmbeds;
        private readonly Tensor<float> _promptPooledEmbeds;
        private readonly Tensor<float> _negativePromptEmbeds;
        private readonly Tensor<float> _negativePromptPooledEmbeds;

        public PromptResult(Tensor<float> promptEmbeds, Tensor<float> pooledPromptEmbeds, Tensor<float> negativePromptEmbeds, Tensor<float> negativePooledPromptEmbeds)
        {
            _promptEmbeds = promptEmbeds;
            _promptPooledEmbeds = pooledPromptEmbeds;
            _negativePromptEmbeds = negativePromptEmbeds;
            _negativePromptPooledEmbeds = negativePooledPromptEmbeds;
        }


        public Tensor<float> PromptEmbeds => _promptEmbeds;
        public Tensor<float> PromptPooledEmbeds => _promptPooledEmbeds;
        public Tensor<float> NegativePromptEmbeds => _negativePromptEmbeds;
        public Tensor<float> NegativePromptPooledEmbeds => _negativePromptPooledEmbeds;

        public Tensor<float> GetPromptEmbeds(bool classifierFreeGuidance)
        {
            if (classifierFreeGuidance)
                return _negativePromptEmbeds.Concatenate(_promptEmbeds);

            return _promptEmbeds;
        }

        public Tensor<float> GetPromptPooledEmbeds(bool classifierFreeGuidance)
        {
            if (classifierFreeGuidance)
                return _negativePromptPooledEmbeds.Concatenate(_promptPooledEmbeds);

            return _promptPooledEmbeds;
        }
    }
}
