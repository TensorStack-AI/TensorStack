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
    public class FlowMatchEulerDiscreteScheduler : SchedulerBase
    {
        private float _sigmaMin;
        private float _sigmaMax;

        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDiscreteScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public FlowMatchEulerDiscreteScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        /// <param name="strength">The strength.</param>
        public override void Initialize(float strength)
        {
            base.Initialize(strength);
            var timesteps = ArrayHelpers.Linspace(1, Options.TrainTimesteps, Options.TrainTimesteps);
            var sigmas = timesteps
                .Select(x => x / Options.TrainTimesteps)
                .Select(sigma => Options.Shift * sigma / (1f + (Options.Shift - 1f) * sigma))
                .ToArray();
            _sigmaMin = sigmas.Min();
            _sigmaMax = sigmas.Max();
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var timesteps = ArrayHelpers.Linspace(SigmaToTimestep(_sigmaMin), SigmaToTimestep(_sigmaMax), Options.Steps);
            if (Options.Steps == 1)
                timesteps = [Options.TrainTimesteps];

            var sigmas = timesteps
                .Select(x => x / Options.TrainTimesteps)
                .Select(sigma => Options.Shift * sigma / (1f + (Options.Shift - 1f) * sigma))
                .Reverse();

            Sigmas = [.. sigmas, 0f];

            var timestepValues = sigmas
                 .Select(sigma => sigma * Options.TrainTimesteps)
                 .Select(x => (int)Math.Round(x))
                 .OrderByDescending(x => x)
                 .ToArray();
            return timestepValues;
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        public override Tensor<float> ScaleInput(int timestep, Tensor<float> sample)
        {
            return sample;
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
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];
            var sigmaNext = Sigmas[stepIndex + 1];

            var prevSample = sample
                .MultiplyTo(sigmaNext - sigma)
                .AddTo(previousSample);
            return new SchedulerResult(prevSample);
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
                .Add(sample.MultiplyTo(1f - sigma));
        }


        /// <summary>
        /// Sigmas to timestep.
        /// </summary>
        /// <param name="sigma">The sigma.</param>
        /// <returns>System.Single.</returns>
        private float SigmaToTimestep(float sigma)
        {
            return sigma * Options.TrainTimesteps;
        }
    }
}
