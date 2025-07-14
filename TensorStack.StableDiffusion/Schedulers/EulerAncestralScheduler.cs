// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;

namespace TensorStack.StableDiffusion.Schedulers
{
    public sealed class EulerAncestralScheduler : EulerScheduler
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EulerAncestralScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public EulerAncestralScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Computes the next prediction steps
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="previousSample">The previous sample.</param>
        public override SchedulerResult Step(int timestep, Tensor<float> sample, Tensor<float> previousSample)
        {
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = CreatePredictedSample(sample, previousSample, sigma);

            var sigmaFrom = Sigmas[stepIndex];
            var sigmaTo = Sigmas[stepIndex + 1];

            var sigmaFromLessSigmaTo = MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2);
            var sigmaUpResult = MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo / MathF.Pow(sigmaFrom, 2);
            var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

            var sigmaDownResult = MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2);
            var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

            // 2. Convert to an ODE derivative
            var derivative = previousSample
                .SubtractTo(predOriginalSample)
                .DivideTo(sigma);

            var delta = sigmaDown - sigma;
            var prevSample = previousSample.AddTo(derivative.MultiplyTo(delta));
            var noise = CreateRandomSample(prevSample.Dimensions);
            prevSample = prevSample.AddTo(noise.MultiplyTo(sigmaUp));
            return new SchedulerResult(prevSample);
        }

    }
}
