// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Linq;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Schedulers
{
    public class DDIMScheduler : SchedulerBase
    {
        private float _finalAlphaCumprod;

        /// <summary>
        /// Initializes a new instance of the <see cref="DDIMScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public DDIMScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        /// <param name="strength">The strength.</param>
        public override void Initialize(float strength)
        {
            base.Initialize(strength);
            bool setAlphaToOne = true;
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
            var timesteps = CreateTimestepSpacing();
            return timesteps
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
            int currentTimestep = timestep;
            int currentTimestepIndex = Timesteps.IndexOf(currentTimestep);
            int previousTimestepIndex = currentTimestepIndex + 1;
            int previousTimestep = Timesteps.ElementAtOrDefault(previousTimestepIndex);

            //# 1. compute alphas, betas
            float alphaProdT = AlphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0 ? AlphasCumProd[previousTimestep] : _finalAlphaCumprod;
            float betaProdT = 1f - alphaProdT;

            //# 2. compute predicted original sample from predicted noise also called
            //# "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            Tensor<float> predEpsilon = null;
            Tensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                var sampleBeta = previousSample.SubtractTo(sample.MultiplyTo(MathF.Sqrt(betaProdT)));
                predOriginalSample = sampleBeta.DivideTo(MathF.Sqrt(alphaProdT));
                predEpsilon = sample;
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                predOriginalSample = sample;
                predEpsilon = previousSample.SubtractTo(predOriginalSample
                    .MultiplyTo(MathF.Sqrt(alphaProdT)))
                    .DivideTo(MathF.Sqrt(betaProdT));
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                var tmp = previousSample.MultiplyTo(MathF.Pow(alphaProdT, 0.5f));
                var tmp2 = sample.MultiplyTo(MathF.Pow(betaProdT, 0.5f));
                predOriginalSample = tmp.Subtract(tmp2);

                var tmp3 = sample.MultiplyTo(MathF.Pow(alphaProdT, 0.5f));
                var tmp4 = previousSample.MultiplyTo(MathF.Pow(betaProdT, 0.5f));
                predEpsilon = tmp3.Add(tmp4);
            }

            //# 3. Clip or threshold "predicted x_0"
            if (Options.Thresholding)
            {
                // TODO:
                // predOriginalSample = ThresholdSample(predOriginalSample);
            }
            else if (Options.ClipSample)
            {
                predOriginalSample = predOriginalSample.ClipTo(-Options.ClipSampleRange, Options.ClipSampleRange);
            }

            //# 4. compute variance: "sigma_t(η)" -> see formula (16)
            //# σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            var eta = 0f;
            var variance = GetVariance(currentTimestep, previousTimestep);
            var stdDevT = eta * MathF.Pow(variance, 0.5f);

            var useClippedModelOutput = false;
            if (useClippedModelOutput)
            {
                //# the pred_epsilon is always re-derived from the clipped x_0 in Glide
                predEpsilon = previousSample
                    .SubtractTo(predOriginalSample.MultiplyTo(MathF.Pow(alphaProdT, 0.5f)))
                    .DivideTo(MathF.Pow(betaProdT, 0.5f));
            }

            //# 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var predSampleDirection = predEpsilon.MultiplyTo(MathF.Pow(1.0f - alphaProdTPrev - MathF.Pow(stdDevT, 2f), 0.5f));

            //# 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var prevSample = predSampleDirection.AddTo(predOriginalSample.MultiplyTo(MathF.Pow(alphaProdTPrev, 0.5f)));

            if (eta > 0)
                prevSample = prevSample.AddTo(CreateRandomSample(sample.Dimensions).MultiplyTo(stdDevT));

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
            float alphaProd = AlphasCumProd[timestep];
            float sqrtAlpha = MathF.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = MathF.Sqrt(1.0f - alphaProd);
            return noise
                .MultiplyTo(sqrtOneMinusAlpha)
                .AddTo(sample.MultiplyTo(sqrtAlpha));
        }


        /// <summary>
        /// Gets the variance.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="previousTimestep">The previous timestep.</param>
        /// <returns>System.Single.</returns>
        private float GetVariance(int timestep, int previousTimestep)
        {
            float alphaProdT = AlphasCumProd[timestep];
            float alphaProdTPrev = previousTimestep >= 0
                ? AlphasCumProd[timestep]
                : _finalAlphaCumprod;

            float betaProdT = 1f - alphaProdT;
            float betaProdTPrev = 1f - alphaProdTPrev;
            float variance = betaProdTPrev / betaProdT * (1f - alphaProdT / alphaProdTPrev);
            return variance;
        }
    }
}
