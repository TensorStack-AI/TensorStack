using System;
using TensorStack.Common.Tensor;

namespace TensorStack.StableDiffusion.Common
{
    public record EncoderCache
    {
        public ImageTensor InputImage { get; init; }
        public Tensor<float> CacheResult { get; init; }

        public bool IsValid(ImageTensor input)
        {
            if (input is null || InputImage is null)
                return false;

            if (!InputImage.Span.SequenceEqual(input.Span))
                return false;

            return true;
        }
    }
}
