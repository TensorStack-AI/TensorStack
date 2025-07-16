// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;

namespace TensorStack.StableDiffusion.Schedulers
{
    public class FlowMatchEulerDynamicScheduler : FlowMatchEulerDiscreteScheduler
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDynamicScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public FlowMatchEulerDynamicScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Computes the next prediction steps
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="previousSample">The previous sample.</param>
        /// <returns>SchedulerResult.</returns>
        public override SchedulerResult Step(int timestep, Tensor<float> sample, Tensor<float> previousSample)
        {
            CurrentStep++;
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];
            var sigmaNext = Sigmas[stepIndex + 1];

            var noise = CreateRandomSample(sample.Dimensions);
            var prevSample = noise
                .MultiplyTo(sigmaNext)
                .AddTo(previousSample
                    .Subtract(sample.MultiplyTo(sigma))
                    .MultiplyTo(1f - sigmaNext));
            return new SchedulerResult(prevSample);
        }

    }
}
