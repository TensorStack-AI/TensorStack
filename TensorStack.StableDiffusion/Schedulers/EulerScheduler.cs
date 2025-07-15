// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Linq;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Helpers;

namespace TensorStack.StableDiffusion.Schedulers
{
    public class EulerScheduler : SchedulerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EulerScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public EulerScheduler(ISchedulerOptions options) : base(options) { }


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
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        public override Tensor<float> ScaleInput(int timestep, Tensor<float> sample)
        {
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];
            sigma = MathF.Sqrt(MathF.Pow(sigma, 2f) + 1f);
            return sample.DivideTo(sigma);
        }


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
            float s_churn = 0f;
            float s_tmin = 0f;
            float s_tmax = float.PositiveInfinity;
            float s_noise = 1f;

            var stepIndex = Timesteps.IndexOf(timestep);
            float sigma = Sigmas[stepIndex];

            float gamma = s_tmin <= sigma && sigma <= s_tmax ? (float)Math.Min(s_churn / (Sigmas.Length - 1f), Math.Sqrt(2.0f) - 1.0f) : 0f;
            var noise = CreateRandomSample(sample.Dimensions);
            var epsilon = noise.MultiplyTo(s_noise);
            float sigmaHat = sigma * (1.0f + gamma);

            if (gamma > 0)
                previousSample = previousSample.AddTo(epsilon.MultiplyTo(MathF.Sqrt(MathF.Pow(sigmaHat, 2f) - MathF.Pow(sigma, 2f))));

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = Options.PredictionType != PredictionType.Epsilon
                ? CreatePredictedSample(sample, previousSample, sigma)
                : previousSample.SubtractTo(sample.MultiplyTo(sigmaHat));

            // 2. Convert to an ODE derivative
            var derivative = previousSample
                .SubtractTo(predOriginalSample)
                .DivideTo(sigmaHat);

            var delta = Sigmas[stepIndex + 1] - sigmaHat;
            return new SchedulerResult(previousSample.AddTo(derivative.MultiplyTo(delta)));
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        public override Tensor<float> ScaleNoise(int timestep, Tensor<float> sample, Tensor<float> noise)
        {
            var index = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[index];
            return noise
                .Multiply(sigma)
                .Add(sample);
        }

    }
}
