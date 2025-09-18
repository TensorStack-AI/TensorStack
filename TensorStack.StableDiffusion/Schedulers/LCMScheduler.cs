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
    public class LCMScheduler : SchedulerBase
    {
        private float _finalAlphaCumprod;

        /// <summary>
        /// Initializes a new instance of the <see cref="LCMScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public LCMScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        /// <param name="strength">The strength.</param>
        public override void Initialize(float strength)
        {
            base.Initialize(strength);
            bool setAlphaToOne = false;
            _finalAlphaCumprod = setAlphaToOne
                ? 1.0f
            : AlphasCumProd.First();
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            // LCM Timesteps Setting
            // Currently, only linear spacing is supported.
            var timeIncrement = Options.TrainTimesteps / Options.Steps - 1;

            //# LCM Training Steps Schedule
            var lcmOriginTimesteps = Enumerable.Range(1, Options.Steps)
                .Select(x => x * timeIncrement)
                .ToArray();

            if (Options.Steps <= 1)
                return [lcmOriginTimesteps[^1]];

            var steps = ArrayHelpers.Linspace(0, lcmOriginTimesteps.Length - 1, Options.Steps)
               .Select(x => (int)Math.Round(x))
               .Select(x => lcmOriginTimesteps[x])
               .OrderByDescending(x => x)
               .ToArray();
            return steps;
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
            //# Latent Consistency Models paper https://arxiv.org/abs/2310.04378
            int currentTimestep = timestep;

            // 1. get previous step value
            int prevIndex = Timesteps.IndexOf(currentTimestep) + 1;
            int previousTimestep = prevIndex < Timesteps.Count
                ? Timesteps[prevIndex]
                : currentTimestep;

            //# 2. compute alphas, betas
            float alphaProdT = AlphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0
                ? AlphasCumProd[previousTimestep]
                : _finalAlphaCumprod;
            float betaProdT = 1f - alphaProdT;
            float betaProdTPrev = 1f - alphaProdTPrev;

            float alphaProdTSqrt = MathF.Sqrt(alphaProdT);
            float betaProdTSqrt = MathF.Sqrt(betaProdT);
            float betaProdTPrevSqrt = MathF.Sqrt(betaProdTPrev);
            float alphaProdTPrevSqrt = MathF.Sqrt(alphaProdTPrev);


            // 3.Get scalings for boundary conditions
            (float cSkip, float cOut) = GetBoundaryConditionScalings(currentTimestep);


            //# 4. compute predicted original sample from predicted noise also called "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            Tensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                predOriginalSample = previousSample
                    .SubtractTo(sample.MultiplyTo(betaProdTSqrt))
                    .DivideTo(alphaProdTSqrt);
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                predOriginalSample = sample;
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                predOriginalSample = previousSample
                    .MultiplyTo(alphaProdTSqrt)
                    .SubtractTo(sample.MultiplyTo(betaProdTSqrt));
            }


            //# 5. Clip or threshold "predicted x_0"
            // TODO: Threshold and Clipping

            //# 6. Denoise model output using boundary conditions
            var denoised = previousSample
                .MultiplyTo(cSkip)
                .AddTo(predOriginalSample.MultiplyTo(cOut));


            //# 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
            //# Noise is not used on the final timestep of the timestep schedule.
            //# This also means that noise is not used for one-step sampling.
            if (Timesteps.IndexOf(currentTimestep) != Options.Steps - 1)
            {
                var noise = CreateRandomSample(sample.Dimensions);
                predOriginalSample = noise
                    .MultiplyTo(betaProdTPrevSqrt)
                    .AddTo(denoised.MultiplyTo(alphaProdTPrevSqrt));
            }
            else
            {
                predOriginalSample = denoised;
            }

            return new SchedulerResult(predOriginalSample);
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        public override Tensor<float> ScaleNoise(int timestep, Tensor<float> sample, Tensor<float> noise)
        {
            int index = Timesteps.IndexOf(timestep);
            float alphaProd = AlphasCumProd[index];
            float sqrtAlpha = MathF.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = MathF.Sqrt(1.0f - alphaProd);
            return noise
                .Multiply(sqrtOneMinusAlpha)
                .Add(sample.MultiplyTo(sqrtAlpha));
        }


        /// <summary>
        /// Gets the boundary condition scalings.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public (float cSkip, float cOut) GetBoundaryConditionScalings(float timestep)
        {
            //self.sigma_data = 0.5  # Default: 0.5
            var sigmaData = 0.5f;
            var timestepScaling = 10f;
            var scaledTimestep = timestepScaling * timestep;

            float c = MathF.Pow(scaledTimestep, 2f) + MathF.Pow(sigmaData, 2f);
            float cSkip = MathF.Pow(sigmaData, 2f) / c;
            float cOut = scaledTimestep / MathF.Pow(c, 0.5f);
            return (cSkip, cOut);
        }

    }
}
