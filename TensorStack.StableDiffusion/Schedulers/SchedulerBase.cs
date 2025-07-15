// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Helpers;

namespace TensorStack.StableDiffusion.Schedulers
{
    public abstract class SchedulerBase : IScheduler
    {
        private readonly ISchedulerOptions _options;
        private Random _random;
        private int _startStep = 0;
        private List<int> _timesteps;
        private float _startSigma = 1f;

        /// <summary>
        /// Initializes a new instance of the <see cref="SchedulerBase"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public SchedulerBase(ISchedulerOptions schedulerOptions)
        {
            _options = schedulerOptions;
        }

        /// <summary>
        /// Gets or sets the sigmas.
        /// </summary>
        protected float[] Sigmas { get; set; }

        /// <summary>
        /// Gets or sets the alphas.
        /// </summary>
        protected float[] Alphas { get; set; }

        /// <summary>
        /// Gets or sets the betas.
        /// </summary>
        protected float[] Betas { get; set; }

        /// <summary>
        /// Gets or sets the alphas cum product.
        /// </summary>
        protected float[] AlphasCumProd { get; set; }

        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        protected IReadOnlyList<int> Timesteps => _timesteps;

        /// <summary>
        /// Gets the scheduler options.
        /// </summary>
        public ISchedulerOptions Options => _options;

        /// <summary>
        /// Gets the random initiated with the seed.
        /// </summary>
        public Random Random => _random;

        /// <summary>
        /// Gets the initial noise sigma.
        /// </summary>
        public float StartSigma => _startSigma;

        /// <summary>
        /// Gets or sets the order.
        /// </summary>
        public virtual int Order { get; } = 1;

        /// <summary>
        /// Gets a value indicating if this step is final order.
        /// </summary>
        public virtual bool IsFinalOrder { get; } = true;

        /// <summary>
        /// Gets or sets the total steps.
        /// </summary>
        public int TotalSteps { get; set; }

        /// <summary>
        /// Gets or sets the current step.
        /// </summary>
        public int CurrentStep { get; set; }

        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected abstract int[] SetTimesteps();

        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <returns></returns>
        public abstract Tensor<float> ScaleInput(int timestep, Tensor<float> sample);

        /// <summary>
        /// Computes the next prediction steps
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="previousSample">The previous sample.</param>
        public abstract SchedulerResult Step(int timestep, Tensor<float> sample, Tensor<float> previousSample);

        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        /// <returns></returns>
        public abstract Tensor<float> ScaleNoise(int timestep, Tensor<float> sample, Tensor<float> noise);


        /// <summary>
        /// Gets the start timestep.
        /// </summary>
        public int GetStartTimestep()
        {
            return Timesteps[_startStep];
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        public IReadOnlyList<int> GetTimesteps()
        {
            if (!Options.Timesteps.IsNullOrEmpty())
                return Options.Timesteps;

            return [.. Timesteps.Skip(_startStep)];
        }


        /// <summary>
        /// Creates a random sample with the specified dimesions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public virtual Tensor<float> CreateRandomSample(ReadOnlySpan<int> dimensions)
        {
            return Random.NextTensor(dimensions);
        }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        public virtual void Initialize(float strength)
        {
            Betas = CreateBetaSchedule();
            Alphas = Betas
                .Select(beta => 1.0f - beta)
                .ToArray();
            AlphasCumProd = Alphas
                .Select((alpha, i) => Alphas.Take(i + 1).Aggregate((a, b) => a * b))
                .ToArray();
            Sigmas = AlphasCumProd
                 .Select(alpha_prod => MathF.Sqrt((1f - alpha_prod) / alpha_prod))
                 .OrderByDescending(x => x)
                 .ToArray();

            _timesteps = [.. SetTimesteps()];
            _startStep = GetStartStep(strength);
            _random = new Random(_options.Seed);

            CurrentStep = 0;
            TotalSteps = Order > 1 ? (_timesteps.Count + 1) / 2 : _timesteps.Count;
        }


        /// <summary>
        /// Gets the beta schedule.
        /// </summary>
        /// <returns></returns>
        protected virtual float[] CreateBetaSchedule()
        {
            var betas = Enumerable.Empty<float>();
            if (Options.TrainedBetas != null)
            {
                betas = Options.TrainedBetas;
            }
            else if (Options.BetaSchedule == BetaScheduleType.Linear)
            {
                betas = ArrayHelpers.Linspace(Options.BetaStart, Options.BetaEnd, Options.TrainTimesteps);
            }
            else if (Options.BetaSchedule == BetaScheduleType.ScaledLinear)
            {
                var start = MathF.Sqrt(Options.BetaStart);
                var end = MathF.Sqrt(Options.BetaEnd);
                betas = ArrayHelpers.Linspace(start, end, Options.TrainTimesteps).Select(x => x * x);
            }
            else if (Options.BetaSchedule == BetaScheduleType.SquaredCosCapV2)
            {
                betas = CreateBetasForAlphaBar();
            }
            else if (Options.BetaSchedule == BetaScheduleType.Sigmoid)
            {
                var mul = Options.BetaEnd - Options.BetaStart;
                var betaSig = ArrayHelpers.Linspace(-6f, 6f, Options.TrainTimesteps);
                var sigmoidBetas = betaSig
                    .Select(beta => 1.0f / (1.0f + MathF.Exp(-beta)))
                    .ToArray();
                betas = sigmoidBetas
                    .Select(x => (x * mul) + Options.BetaStart)
                    .ToArray();
            }
            return betas.ToArray();
        }


        /// <summary>
        /// Gets the timestep spacing.
        /// </summary>
        /// <returns></returns>
        protected virtual float[] CreateTimestepSpacing()
        {
            if (Options.Steps <= 1)
                return [Options.TrainTimesteps - 1];

            if (Options.TimestepSpacing == TimestepSpacingType.Leading)
            {
                var stepRatio = Options.TrainTimesteps / Options.Steps;
                return Enumerable.Range(0, Options.Steps)
                    .Select(x => ((float)x * stepRatio) + Options.StepsOffset)
                    .OrderByDescending(x => x)
                    .ToArray();
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Trailing)
            {
                var stepRatio = Options.TrainTimesteps / Math.Max(1, Options.Steps);
                var result = Enumerable.Range(0, Options.TrainTimesteps + 1)
                    .Where((number, index) => index % stepRatio == 0 && number > 0)
                    .Select(x => MathF.Max(0, x - 1f))
                    .OrderByDescending(x => x)
                    .ToArray();
                return result;
            }

            // TimestepSpacingType.Linspace
            return ArrayHelpers.Linspace(0, Options.TrainTimesteps - 1, Options.Steps)
                .OrderByDescending(x => x)
                .ToArray();
        }


        /// <summary>
        /// Gets the predicted sample.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="sigma">The sigma.</param>
        /// <returns></returns>
        protected virtual Tensor<float> CreatePredictedSample(Tensor<float> modelOutput, Tensor<float> sample, float sigma)
        {
            Tensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                predOriginalSample = sample.SubtractTo(modelOutput.MultiplyTo(sigma));
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                var sigmaSqrt = MathF.Sqrt(sigma * sigma + 1);
                predOriginalSample = sample.DivideTo(sigmaSqrt)
                    .AddTo(modelOutput.MultiplyTo(-sigma / sigmaSqrt));
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                //prediction_type not implemented yet: sample
                predOriginalSample = sample.Clone();
            }
            return predOriginalSample;
        }


        /// <summary>
        /// Sets the initial noise sigma.
        /// </summary>
        /// <param name="initNoiseSigma">The initial noise sigma.</param>
        protected void SetInitNoiseSigma()
        {
            var maxSigma = Sigmas.Max();
            var initNoiseSigma = Options.TimestepSpacing == TimestepSpacingType.Linspace || Options.TimestepSpacing == TimestepSpacingType.Trailing
                ? maxSigma
                : MathF.Sqrt(maxSigma * maxSigma + 1f);
            _startSigma = initNoiseSigma;
        }


        /// <summary>
        /// Gets the previous timestep.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected int GetPreviousTimestep(int timestep)
        {
            var index = Timesteps.IndexOf(timestep) + 1;
            if (index > Timesteps.Count - 1)
                return 0;

            return Timesteps[index];
        }


        /// <summary>
        /// Gets the betas for alpha bar.
        /// </summary>
        /// <param name="maxBeta">The maximum beta.</param>
        /// <param name="alphaTransformType">Type of the alpha transform.</param>
        /// <returns></returns>
        protected float[] CreateBetasForAlphaBar()
        {
            Func<float, float> alphaBarFn = null;
            if (_options.AlphaTransformType == AlphaTransformType.Cosine)
            {
                alphaBarFn = t => MathF.Pow(MathF.Cos((t + 0.008f) / 1.008f * MathF.PI / 2.0f), 2.0f);
            }
            else if (_options.AlphaTransformType == AlphaTransformType.Exponential)
            {
                alphaBarFn = t => MathF.Exp(t * -12.0f);
            }

            return Enumerable
                .Range(0, _options.TrainTimesteps)
                .Select(i =>
                {
                    var t1 = (float)i / _options.TrainTimesteps;
                    var t2 = (float)(i + 1) / _options.TrainTimesteps;
                    return MathF.Min(1f - alphaBarFn(t2) / alphaBarFn(t1), _options.MaximumBeta);
                }).ToArray();
        }


        /// <summary>
        /// Interpolates the specified timesteps.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="range">The range.</param>
        /// <param name="sigmas">The sigmas.</param>
        /// <returns></returns>
        protected virtual float[] Interpolate(float[] timesteps, float[] range, float[] sigmas)
        {
            var result = new float[timesteps.Length];
            for (int i = 0; i < timesteps.Length; i++)
            {
                float t = timesteps[i];
                int index = ArrayHelpers.BinarySearchDescending(range, t);
                if (index >= 0)
                {
                    // Exact match
                    result[i] = sigmas[index];
                }
                else
                {
                    index = ~index;
                    if (index == 0)
                    {
                        // t < range[0], clamp to first
                        result[i] = sigmas[0];
                    }
                    else if (index >= range.Length)
                    {
                        // t > range[^1], clamp to last
                        result[i] = sigmas[^1];
                    }
                    else
                    {
                        // Interpolate between index - 1 and index
                        float t0 = range[index - 1];
                        float t1 = range[index];
                        float s0 = sigmas[index - 1];
                        float s1 = sigmas[index];
                        float factor = (t - t0) / (t1 - t0);
                        result[i] = s0 + factor * (s1 - s0);
                    }
                }
            }

            return result;
        }


        /// <summary>
        /// Converts sigmas to karras.
        /// </summary>
        /// <param name="inSigmas">The in sigmas.</param>
        /// <returns></returns>
        protected float[] ConvertToKarras(float[] inSigmas)
        {
            // Get the minimum and maximum values from the input sigmas
            float sigmaMin = inSigmas[^1];
            float sigmaMax = inSigmas[0];

            // Set the value of rho, which is used in the calculation
            float rho = 7.0f; // 7.0 is the value used in the paper

            // Create a linear ramp from 0 to 1
            float[] ramp = Enumerable.Range(0, _options.Steps)
                .Select(i => (float)i / (_options.Steps - 1))
                .ToArray();

            // Calculate the inverse of sigmaMin and sigmaMax raised to the power of 1/rho
            float minInvRho = MathF.Pow(sigmaMin, 1.0f / rho);
            float maxInvRho = MathF.Pow(sigmaMax, 1.0f / rho);

            // Calculate the Karras noise schedule using the formula from the paper
            float[] sigmas = new float[_options.Steps];
            for (int i = 0; i < _options.Steps; i++)
            {
                sigmas[i] = MathF.Pow(maxInvRho + ramp[i] * (minInvRho - maxInvRho), rho);
            }
            return sigmas;
        }


        /// <summary>
        /// Create timesteps form sigmas
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <param name="logSigmas">The log sigmas.</param>
        /// <returns></returns>
        protected float[] SigmaToTimestep(float[] sigmas, float[] logSigmas)
        {
            var sigmasReversed = sigmas.Reverse().ToArray();
            var logSigmasReversed = logSigmas.Reverse().ToArray();
            var timesteps = new float[sigmasReversed.Length];
            for (int i = 0; i < sigmasReversed.Length; i++)
            {
                float logSigma = MathF.Log(sigmasReversed[i].ZeroIfNan());
                float[] dists = new float[logSigmasReversed.Length];

                for (int j = 0; j < logSigmasReversed.Length; j++)
                {
                    dists[j] = logSigma - logSigmasReversed[j];
                }

                int lowIdx = 0;
                int highIdx = 1;

                for (int j = 0; j < logSigmasReversed.Length - 1; j++)
                {
                    if (dists[j] >= 0)
                    {
                        lowIdx = j;
                        highIdx = j + 1;
                    }
                }

                float low = logSigmasReversed[lowIdx];
                float high = logSigmasReversed[highIdx];

                float w = (low - logSigma) / (low - high);
                w = Math.Clamp(w, 0, 1);

                float ti = (1 - w) * lowIdx + w * highIdx;
                timesteps[i] = ti;
            }

            return timesteps;
        }


        /// <summary>
        /// Gets the start step.
        /// </summary>
        private int GetStartStep(float strength)
        {
            var initial = Math.Min((int)(Options.Steps * strength), Options.Steps);
            return Math.Min(Math.Max(Options.Steps - initial, 0), Options.Steps - 1);
        }

        #region IDisposable

        private bool disposed = false;


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                // Dispose managed resources here.
                _timesteps?.Clear();
                Sigmas = null;
                Alphas = null;
                Betas = null;
                AlphasCumProd = null;
            }

            // Dispose unmanaged resources here (if any).
            disposed = true;
        }


        /// <summary>
        /// Finalizes an instance of the <see cref="SchedulerBase"/> class.
        /// </summary>
        ~SchedulerBase()
        {
            Dispose(false);
        }

        #endregion
    }
}