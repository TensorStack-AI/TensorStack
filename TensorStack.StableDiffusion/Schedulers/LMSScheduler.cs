// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Helpers;

namespace TensorStack.StableDiffusion.Schedulers
{
    public class LMSScheduler : SchedulerBase
    {
        private int _order = 4;
        private Queue<Tensor<float>> _derivatives;

        /// <summary>
        /// Initializes a new instance of the <see cref="LMSScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public LMSScheduler(ISchedulerOptions options) : base(options) { }

        public override void Initialize(float strength)
        {
            base.Initialize(strength);
            _derivatives = new Queue<Tensor<float>>();
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
            int stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = CreatePredictedSample(sample, previousSample, sigma);

            // 2. Convert to an ODE derivative
            var derivativeSample = previousSample
                .SubtractTo(predOriginalSample)
                .DivideTo(sigma);

            _derivatives.Enqueue(derivativeSample);
            if (_derivatives.Count > _order)
                _derivatives.Dequeue();

            // 3. compute linear multistep coefficients
            _order = Math.Min(stepIndex + 1, _order);
            var lmsCoeffs = Enumerable.Range(0, _order)
                .Select(currOrder => GetLmsCoefficient(_order, stepIndex, currOrder));

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = _derivatives.Reverse();

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs
                .Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative))
                .ToArray();

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new Tensor<float>[_derivatives.Count];
            for (int i = 0; i < lmsCoeffsAndDerivatives.Length; i++)
            {
                // Multiply to coeff by each derivatives to create the new tensors
                var (lmsCoeff, derivative) = lmsCoeffsAndDerivatives[i];
                lmsDerProduct[i] = derivative.MultiplyTo(lmsCoeff);
            }

            // Add the sumed tensor to the sample
            return new SchedulerResult(previousSample.AddTo(lmsDerProduct.SumTensors(sample.Dimensions)));
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


        /// <summary>
        /// Gets the LMS coefficient.
        /// </summary>
        /// <param name="order">The order.</param>
        /// <param name="t">The t.</param>
        /// <param name="currentOrder">The current order.</param>
        /// <returns></returns>
        private float GetLmsCoefficient(int order, int t, int currentOrder)
        {
            return MathHelpers.IntegrateOnClosedInterval(tau => GetLmsDerivative(tau, order, t, currentOrder), Sigmas[t], Sigmas[t + 1], 1e-4);
        }


        /// <summary>
        /// LMSs the derivative.
        /// </summary>
        /// <param name="tau">The tau.</param>
        /// <param name="order">The order.</param>
        /// <param name="t">The t.</param>
        /// <param name="currentOrder">The current order.</param>
        /// <returns>System.Double.</returns>
        private double GetLmsDerivative(double tau, int order, int t, int currentOrder)
        {
            double prod = 1.0;
            for (int k = 0; k < order; k++)
            {
                if (currentOrder == k)
                    continue;

                prod *= (tau - Sigmas[t - k]) / (Sigmas[t - currentOrder] - Sigmas[t - k]);
            }
            return prod;
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            _derivatives?.Clear();
            base.Dispose(disposing);
        }
    }
}
