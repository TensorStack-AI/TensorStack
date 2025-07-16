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
    public class KDPM2AncestralScheduler : SchedulerBase
    {
        private int _stepIndex;
        private float[] _sigmasInterpol;
        private Tensor<float> _previousSample;
        private float[] _sigmas_up;
        private float[] _sigmas_down;

        /// <summary>
        /// Initializes a new instance of the <see cref="KDPM2AncestralScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public KDPM2AncestralScheduler(ISchedulerOptions options) : base(options) { }

        /// <summary>
        /// Gets or sets the order.
        /// </summary>
        public override int Order => 2;


        /// <summary>
        /// Gets a value indicating if this step is final order.
        /// </summary>
        public override bool IsFinalOrder => _previousSample is not null;


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        /// <param name="strength">The strength.</param>
        public override void Initialize(float strength)
        {
            _stepIndex = 0;
            _previousSample = null;
            Options.TimestepSpacing = TimestepSpacingType.Trailing;
            base.Initialize(strength);
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

            //compute up and down sigmas
            var sigmas_next = RollLeft(sigmas);
            sigmas_next[^1] = 0.0f;
            var sigmas_up = ComputeSigmasUp(sigmas, sigmas_next);
            var sigmas_down = ComputeSigmasDown(sigmas_next, sigmas_up);
            sigmas_down[^1] = 0.0f;

            //compute interpolated sigmas
            var sigmas_interpol = InterpolateSigmas(sigmas, sigmas_down);

            Sigmas = RepeatInterleave(sigmas);
            _sigmasInterpol = RepeatInterleave(sigmas_interpol);
            _sigmas_up = RepeatInterleave(sigmas_up);
            _sigmas_down = RepeatInterleave(sigmas_down);

            SetInitNoiseSigma();

            var timestepsInterpol = SigmaToTimestep(sigmas_interpol, logSigmas)
                .OrderByDescending(x => x)
                .ToArray();
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
            var sigma = _previousSample is null
                ? Sigmas[_stepIndex]
                : _sigmasInterpol[_stepIndex - 1];

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
            float sigmaUp;
            float sigmaDown = 0f;
            bool isFirstPass = _previousSample is null;
            if (isFirstPass)
            {
                sigma = Sigmas[_stepIndex];
                sigmaInterpol = _sigmasInterpol[_stepIndex];
                sigmaUp = _sigmas_up[_stepIndex];
                if (_stepIndex > 0)
                    sigmaDown = _sigmas_down[_stepIndex - 1];
            }
            else
            {
                sigma = Sigmas[_stepIndex - 1];
                sigmaInterpol = _sigmasInterpol[_stepIndex - 1];
                sigmaUp = _sigmas_up[_stepIndex - 1];
                sigmaDown = _sigmas_down[_stepIndex - 1];
            }

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
                sampleResult = previousSample.AddTo(derivative.Multiply(delta));

                _previousSample = previousSample.Clone();
            }
            else
            {
                var derivative = previousSample
                    .SubtractTo(predOriginalSample)
                    .DivideTo(sigmaInterpol);

                var delta = sigmaDown - sigmaHat;
                var noise = CreateRandomSample(_previousSample.Dimensions);
                sampleResult = _previousSample
                    .AddTo(derivative.Multiply(delta))
                    .AddTo(noise.Multiply(sigmaUp));

                _previousSample = null;
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
        /// Computes the sigmas up.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <param name="sigmasNext">The sigmas next.</param>
        /// <returns>System.Single[].</returns>
        private float[] ComputeSigmasUp(float[] sigmas, float[] sigmasNext)
        {
            var sigmasUp = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length; i++)
            {
                float sigmaSq = (sigmas[i] * sigmas[i]).ZeroIfNan();
                float sigmaNextSq = (sigmasNext[i] * sigmasNext[i]).ZeroIfNan();
                float value = sigmaNextSq * (sigmaSq - sigmaNextSq) / sigmaSq;
                sigmasUp[i] = MathF.Sqrt(value);
            }
            return sigmasUp;
        }


        /// <summary>
        /// Computes the sigmas down.
        /// </summary>
        /// <param name="sigmasNext">The sigmas next.</param>
        /// <param name="sigmasUp">The sigmas up.</param>
        /// <returns>System.Single[].</returns>
        private float[] ComputeSigmasDown(float[] sigmasNext, float[] sigmasUp)
        {
            var sigmasDown = new float[sigmasNext.Length];
            for (int i = 0; i < sigmasNext.Length; i++)
            {
                float value = (sigmasNext[i] * sigmasNext[i]).ZeroIfNan() - (sigmasUp[i] * sigmasUp[i]).ZeroIfNan();
                sigmasDown[i] = MathF.Sqrt(value);
            }

            return sigmasDown;
        }


        /// <summary>
        /// Interpolates the sigmas.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <param name="sigmasDown">The sigmas down.</param>
        /// <returns>System.Single[].</returns>
        private float[] InterpolateSigmas(float[] sigmas, float[] sigmasDown)
        {
            var sigmasInterpol = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length; i++)
            {
                float logSigma = MathF.Log(sigmas[i]).ZeroIfNan();
                float logSigmaDown = MathF.Log(sigmasDown[i]).ZeroIfNan();

                float lerp = float.Lerp(logSigma, logSigmaDown, 0.5f);
                sigmasInterpol[i] = MathF.Exp(lerp);
            }

            if (sigmas.Length >= 2)
            {
                sigmasInterpol[^2] = 0f;
                sigmasInterpol[^1] = 0f;
            }
            return sigmasInterpol;
        }


        /// <summary>
        /// Interpolates the timesteps.
        /// </summary>
        /// <param name="timestepsInterpol">The timesteps interpol.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns>System.Int32[].</returns>
        private int[] InterpolateTimesteps(float[] timestepsInterpol, float[] timesteps)
        {
            var sliceTimesteps = timesteps[1..];
            var sliceInterpol = timestepsInterpol[..^2];
            var interleaved = new List<int>();
            for (int i = 0; i < sliceInterpol.Length; i++)
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
        /// Rolls array to the left.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <returns>System.Single[].</returns>
        private float[] RollLeft(float[] sigmas)
        {
            var result = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length - 1; i++)
                result[i] = sigmas[i + 1];

            result[^1] = sigmas[0];
            return result;
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _sigmas_up = null;
            _sigmas_down = null;
            _sigmasInterpol = null;
            _previousSample = null;
            base.Dispose(disposing);
        }

    }
}
