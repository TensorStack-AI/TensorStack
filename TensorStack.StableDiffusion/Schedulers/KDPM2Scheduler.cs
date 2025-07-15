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
    public class KDPM2Scheduler : SchedulerBase
    {
        private int _stepIndex;
        private float[] _sigmasInterpol;
        private Tensor<float> _sample;

        /// <summary>
        /// Initializes a new instance of the <see cref="KDPM2Scheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public KDPM2Scheduler(ISchedulerOptions options) : base(options) { }

        /// <summary>
        /// Gets or sets the order.
        /// </summary>
        public override int Order => 2;


        /// <summary>
        /// Gets a value indicating if this step is final order.
        /// </summary>
        public override bool IsFinalOrder => _sample is not null;


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        /// <param name="strength">The strength.</param>
        public override void Initialize(float strength)
        {
            base.Initialize(strength);
            _stepIndex = 0;
            _sample = null;
            Options.TimestepSpacing = TimestepSpacingType.Trailing;
        }


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

            sigmas = [.. sigmas, 0f];
            var sigmasInterpol = InterpolateSigmas(sigmas);

            Sigmas = RepeatInterleave(sigmas);
            _sigmasInterpol = RepeatInterleave(sigmasInterpol);

            SetInitNoiseSigma();

            var timestepsInterpol = SigmaToTimestep(sigmasInterpol, logSigmas);
            var timestepResult = InterpolateTimesteps(timestepsInterpol, timesteps);
            return timestepResult;
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        public override Tensor<float> ScaleInput(int timestep, Tensor<float> sample)
        {
            var sigma = _sample is null
                ? Sigmas[_stepIndex]
                : _sigmasInterpol[_stepIndex];

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
            float sigma;
            float sigmaInterpol;
            float sigmaNext;
            bool isFirstPass = _sample is null;
            if (isFirstPass)
            {
                sigma = Sigmas[_stepIndex];
                sigmaInterpol = _sigmasInterpol[_stepIndex + 1];
                sigmaNext = Sigmas[_stepIndex + 1];
            }
            else
            {
                sigma = Sigmas[_stepIndex - 1];
                sigmaInterpol = _sigmasInterpol[_stepIndex];
                sigmaNext = Sigmas[_stepIndex];
            }

            //# currently only gamma=0 is supported. This usually works best anyways.
            float gamma = 0f;
            float sigmaHat = sigma * (gamma + 1f);
            var sigmaInput = isFirstPass ? sigmaHat : sigmaInterpol;
            var predOriginalSample = CreatePredictedSample(sample, previousSample, sigmaInput);

            Tensor<float> sampleResult;
            if (isFirstPass)
            {
                CurrentStep++;
                var derivative = previousSample
                 .SubtractTo(predOriginalSample)
                 .DivideTo(sigmaHat);

                var delta = sigmaInterpol - sigmaHat;
                sampleResult = previousSample.AddTo(derivative.MultiplyTo(delta));

                _sample = previousSample.Clone();
            }
            else
            {
              
                var derivative = previousSample
                    .SubtractTo(predOriginalSample)
                    .DivideTo(sigmaInterpol);

                var delta = sigmaNext - sigmaHat;
                sampleResult = _sample.AddTo(derivative.MultiplyTo(delta));

                _sample = null;
            }

            _stepIndex += 1;
            return new SchedulerResult(sampleResult);
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        public override Tensor<float> ScaleNoise(int timestep, Tensor<float> sample, Tensor<float> noise)
        {
            var sigma = Sigmas[_stepIndex];
            return noise
                .MultiplyTo(sigma)
                .AddTo(sample);
        }


        /// <summary>
        /// Repeats the interleave.
        /// </summary>
        /// <param name="input">The input.</param>
        public float[] RepeatInterleave(float[] input)
        {
            int index = 0;
            int resultLength = 1 + (input.Length - 1) * 2 + 1;
            float[] result = new float[resultLength];
            result[index++] = input[0].ZeroIfNan();
            for (int i = 1; i < input.Length; i++)
            {
                result[index++] = input[i].ZeroIfNan();
                result[index++] = input[i].ZeroIfNan();
            }
            result[index] = input[^1].ZeroIfNan();
            return result;
        }


        /// <summary>
        /// Interpolates the sigmas.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        public float[] InterpolateSigmas(float[] sigmas)
        {
            var logSigmas = new float[sigmas.Length];
            var rolledLogSigmas = new float[sigmas.Length];
            var result = new float[sigmas.Length];

            for (int i = 0; i < sigmas.Length; i++)
                logSigmas[i] = MathF.Log(sigmas[i]);

            rolledLogSigmas[0] = logSigmas[sigmas.Length - 1];
            for (int i = 1; i < sigmas.Length; i++)
                rolledLogSigmas[i] = logSigmas[i - 1];

            for (int i = 0; i < sigmas.Length; i++)
            {
                float lerp = logSigmas[i] + 0.5f * (rolledLogSigmas[i] - logSigmas[i]);
                result[i] = MathF.Exp(lerp);
            }
            return result;
        }


        /// <summary>
        /// Interpolates the timesteps.
        /// </summary>
        /// <param name="timestepsInterpol">The timesteps interpol.</param>
        /// <param name="timesteps">The timesteps.</param>
        private int[] InterpolateTimesteps(float[] timestepsInterpol, float[] timesteps)
        {
            var sliceTimesteps = timesteps[1..];
            var sliceInterpol = timestepsInterpol[1..^1];
            var interleaved = new List<int>();
            for (int i = 0; i < sliceTimesteps.Length; i++)
            {
                interleaved.Add((int)Math.Round(sliceInterpol[i]));
                interleaved.Add((int)Math.Round(sliceTimesteps[i]));
            }

            interleaved.Add((int)timesteps[0]);
            return interleaved
                .OrderByDescending(x => x)
                .ToArray();
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _sigmasInterpol = null;
            base.Dispose(disposing);
        }
    }
}
