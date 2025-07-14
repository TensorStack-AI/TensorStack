// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.StableDiffusion.Common
{
    public class SchedulerResult
    {
        public SchedulerResult(Tensor<float> sample)
        {
            Sample = sample;
        }


        public Tensor<float> Sample { get; set; }
    }
}
