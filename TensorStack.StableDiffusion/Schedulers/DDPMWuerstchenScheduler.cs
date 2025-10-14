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
    public class DDPMWuerstchenScheduler : SchedulerBase
    {
        private float _s;
        private float _scaler;
        private float _initAlphaCumprod;
        private float _timestepRatio = 1000f;

        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMWuerstchenScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public DDPMWuerstchenScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        /// <param name="strength">The strength.</param>
        public override void Initialize(float strength)
        {
            base.Initialize(strength);
            _s = 0.008f;
            _scaler = 1.0f;
            _initAlphaCumprod = MathF.Pow(MathF.Cos(_s / (1f + _s) * MathF.PI * 0.5f), 2f);
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var timesteps = ArrayHelpers.Linspace(0, _timestepRatio, Options.Steps + 1);
            return timesteps
                .Skip(1)
                .Select(x => (int)Math.Round(x))
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
            var currentTimestep = timestep / _timestepRatio;
            var previousTimestep = GetPreviousTimestep(timestep) / _timestepRatio;

            var alpha_cumprod = GetAlphaCumprod(currentTimestep);
            var alpha_cumprod_prev = GetAlphaCumprod(previousTimestep);
            var alpha = alpha_cumprod / alpha_cumprod_prev;

            var predictedSample = previousSample
                .SubtractTo(sample.MultiplyTo(1f - alpha).DivideTo(MathF.Sqrt(1f - alpha_cumprod)))
                .MultiplyTo(MathF.Sqrt(1f / alpha))
                .AddTo(CreateRandomSample(sample.Dimensions)
                .MultiplyTo(MathF.Sqrt((1f - alpha) * (1f - alpha_cumprod_prev) / (1f - alpha_cumprod))));

            return new SchedulerResult(predictedSample);
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        public override Tensor<float> ScaleNoise(int timestep, Tensor<float> sample, Tensor<float> noise)
        {
            var index = timestep / _timestepRatio;
            var alphaProd = GetAlphaCumprod(index);
            var sqrtAlpha = MathF.Sqrt(alphaProd);
            var sqrtOneMinusAlpha = MathF.Sqrt(1.0f - alphaProd);
            return noise
                .MultiplyTo(sqrtOneMinusAlpha)
                .AddTo(sample.MultiplyTo(sqrtAlpha));
        }


        private float GetAlphaCumprod(float timestep)
        {
            if (_scaler > 1.0f)
                timestep = 1f - MathF.Pow(1f - timestep, _scaler);
            else if (_scaler < 1.0f)
                timestep = MathF.Pow(timestep, _scaler);

            var alphaCumprod = MathF.Pow(MathF.Cos((timestep + _s) / (1f + _s) * MathF.PI * 0.5f), 2f) / _initAlphaCumprod;
            return Math.Clamp(alphaCumprod, 0.0001f, 0.9999f);
        }
    }
}
