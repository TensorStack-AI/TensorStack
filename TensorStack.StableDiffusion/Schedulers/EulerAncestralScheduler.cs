// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Linq;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Helpers;

namespace TensorStack.StableDiffusion.Schedulers
{
    public sealed class EulerAncestralScheduler : SchedulerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EulerAncestralScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public EulerAncestralScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var sigmas = Sigmas.ToArray();
            var timesteps = CreateTimestepSpacing();
            var logSigmas = ArrayHelpers.Log(sigmas);
            var range = ArrayHelpers.Range(0, sigmas.Length, true);
            sigmas = Interpolate(timesteps, range, sigmas);

            if (Options.UseKarrasSigmas)
            {
                sigmas = ConvertToKarras(sigmas);
                timesteps = SigmaToTimestep(sigmas, logSigmas);
            }

            Sigmas = [.. sigmas, 0f];

            SetInitNoiseSigma();

            return timesteps.Select(x => (int)Math.Round(x))
                 .OrderByDescending(x => x)
                 .ToArray();
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public override Tensor<float> ScaleInput(Tensor<float> sample, int timestep)
        {
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];
            sigma = MathF.Sqrt(MathF.Pow(sigma, 2f) + 1f);
            return sample.Divide(sigma, true);
        }


        /// <summary>
        /// Processes a inference step for the specified model output.
        /// </summary>
        /// <param name="sample">The model output.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="previousSample">The sample.</param>
        /// <param name="order">The order.</param>
        /// <returns></returns>
        public override SchedulerResult Step(Tensor<float> sample, int timestep, Tensor<float> previousSample)
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
                .Subtract(predOriginalSample, true)
                .Divide(sigma, true);

            var delta = sigmaDown - sigma;
            var prevSample = previousSample.Add(derivative.Multiply(delta, true), true);
            var noise = CreateRandomSample(prevSample.Dimensions);
            prevSample = prevSample.Add(noise.Multiply(sigmaUp, true), true);
            return new SchedulerResult(prevSample);
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        public override Tensor<float> ScaleNoise(Tensor<float> sample, Tensor<float> noise, int timestep)
        {
            var index = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[index];
            return noise
                .Multiply(sigma)
                .Add(sample);
        }

    }
}
