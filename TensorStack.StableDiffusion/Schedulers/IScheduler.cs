// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;

namespace TensorStack.StableDiffusion.Schedulers
{
    public interface IScheduler : IDisposable
    {
        /// <summary>
        /// Gets the initial noise sigma.
        /// </summary>
        float StartSigma { get; }

        /// <summary>
        /// Initialize
        /// </summary>
        /// <param name="strength">The strength.</param>
        void Initialize(float strength);

        /// <summary>
        /// Gets the start timestep.
        /// </summary>
        int GetStartTimestep();

        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        IReadOnlyList<int> GetTimesteps();

        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <returns></returns>
        Tensor<float> ScaleInput(int timestep, Tensor<float> sample);

        /// <summary>
        /// Computes the next prediction steps
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="previousSample">The previous sample.</param>
        SchedulerResult Step(int timestep, Tensor<float> sample, Tensor<float> previousSample);

        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        Tensor<float> ScaleNoise(int timestep, Tensor<float> sample, Tensor<float> noise);

        /// <summary>
        /// Creates a random sample with the specified dimesions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        Tensor<float> CreateRandomSample(ReadOnlySpan<int> dimensions);
    }
}