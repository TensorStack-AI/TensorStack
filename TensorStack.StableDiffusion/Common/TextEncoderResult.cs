// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.StableDiffusion.Common
{
    public record TextEncoderResult
    {
        private readonly Tensor<float> _textEmbeds;
        private readonly Tensor<float>[] _hiddenStates;

        public TextEncoderResult(Tensor<float>[] hiddenStates, Tensor<float> textEmbeds)
        {
            _textEmbeds = textEmbeds;
            _hiddenStates = hiddenStates;
        }

        public TextEncoderResult(Tensor<float> hiddenStates, Tensor<float> textEmbeds)
            : this([hiddenStates], textEmbeds) { }


        public Tensor<float> TextEmbeds => _textEmbeds;
        public Tensor<float> HiddenStates => _hiddenStates[0];


        public Tensor<float> GetHiddenStates(int index)
        {
            if (index > 0)
                return _hiddenStates[^index];

            return _hiddenStates[0];
        }
    }
}
