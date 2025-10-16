// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;
using TensorStack.Video.Common;
using TensorStack.Video.Models;

namespace TensorStack.Video.Pipelines
{
    /// <summary>
    /// Video Interpolation Pipeline.
    /// </summary>
    public class InterpolationPipeline :
          IPipeline<VideoTensor, InterpolationVideoOptions>,
          IPipelineStream<VideoFrame, InterpolationStreamOptions>
    {
        private readonly InterpolationModel _model;

        /// <summary>
        /// Initializes a new instance of the <see cref="InterpolationPipeline"/> class.
        /// </summary>
        /// <param name="model">The model.</param>
        public InterpolationPipeline(InterpolationModel model)
        {
            _model = model;
        }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        public async Task LoadAsync(CancellationToken cancellationToken = default)
        {
            await _model.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        public async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await _model.UnloadAsync();
        }


        /// <summary>
        /// Run Interpolation asynchronously.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<VideoTensor> RunAsync(InterpolationVideoOptions options, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var frameIndex = 0;
            var totalFrames = options.Video.Frames * options.Multiplier;
            var newFrameRate = options.Video.FrameRate * options.Multiplier;

            var results = new List<ImageTensor>();
            var previousFrame = default(ImageTensor);
            var extraFramePositions = GetFlowEstimationKeyFrames(options.Video.Frames, options.Multiplier);
            foreach (var frame in options.Video.GetFrames())
            {
                var currentFrame = frame.CloneAs();
                if (frameIndex >= totalFrames)
                    break;

                if (previousFrame != null)
                {
                    var timesteps = GetTimesteps(frameIndex, options.Multiplier, extraFramePositions);
                    foreach (var timestep in timesteps)
                    {
                        var newFrame = await RunInterpolationAsync(currentFrame, previousFrame, timestep, cancellationToken);
                        results.Add(newFrame);
                        frameIndex++;
                    }
                }

                previousFrame = currentFrame.CloneAs();
                results.Add(currentFrame);
                frameIndex++;
            }

            return new VideoTensor(results.Join(), newFrameRate);
        }


        /// <summary>
        /// Run Interpolation asynchronously.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async IAsyncEnumerable<VideoFrame> RunAsync(InterpolationStreamOptions options, IProgress<RunProgress> progressCallback = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var frameIndex = 0;
            var totalFrames = options.FrameCount * options.Multiplier;
            var newFrameRate = options.FrameRate * options.Multiplier;

            var previousFrame = default(ImageTensor);
            var extraFramePositions = GetFlowEstimationKeyFrames(options.FrameCount, options.Multiplier);
            await foreach (var frame in options.Stream)
            {
                var currentFrame = frame.Frame.CloneAs();
                if (frameIndex >= totalFrames)
                    yield break;

                if (previousFrame != null)
                {
                    var timesteps = GetTimesteps(frameIndex, options.Multiplier, extraFramePositions);
                    foreach (var timestep in timesteps)
                    {
                        var newFrame = await RunInterpolationAsync(currentFrame, previousFrame, timestep, cancellationToken);
                        yield return new VideoFrame(frameIndex, newFrame, newFrameRate);
                        frameIndex++;
                        ReportProgress(progressCallback, frameIndex, totalFrames);
                    }
                }

                previousFrame = currentFrame.CloneAs();
                yield return new VideoFrame(frameIndex, currentFrame, newFrameRate);
                frameIndex++;

                ReportProgress(progressCallback, frameIndex, totalFrames);
            }
        }

        private void ReportProgress(IProgress<RunProgress> progressCallback, int value, int maximum, string message = default)
        {
            if (progressCallback == null)
                return;

            progressCallback?.Report(new RunProgress(value, maximum, message: message));
        }

        /// <summary>
        /// Run interpolation.
        /// </summary>
        /// <param name="frameTensor">The frame tensor.</param>
        /// <param name="previousFrameTensor">The previous frame tensor.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<ImageTensor> RunInterpolationAsync(ImageTensor frameTensor, ImageTensor previousFrameTensor, float timestep, CancellationToken cancellationToken = default)
        {
            var metadata = await _model.LoadAsync(cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();

            frameTensor.NormalizeZeroOne();
            previousFrameTensor.NormalizeZeroOne();
            using (var modelParameters = new ModelParameters(metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(previousFrameTensor.GetChannels(3));
                modelParameters.AddInput(frameTensor.GetChannels(3));
                modelParameters.AddScalarInput(timestep);

                // Outputs
                modelParameters.AddOutput([1, 3, frameTensor.Dimensions[2], frameTensor.Dimensions[3]]);
                using (var results = await _model.RunInferenceAsync(modelParameters))
                {
                    frameTensor.NormalizeOneOne();
                    previousFrameTensor.NormalizeOneOne();
                    return results[0]
                        .ToTensor()
                        .NormalizeOneOne()
                        .AsImageTensor();
                }
            }
        }


        /// <summary>
        /// Gets the interpolation timesteps.
        /// </summary>
        /// <param name="frameIndex">Index of the frame.</param>
        /// <param name="multiplier">The multiplier.</param>
        /// <param name="extraFramePositions">The extra frame positions.</param>
        /// <returns>System.Single[].</returns>
        private static float[] GetTimesteps(int frameIndex, int multiplier, int[] extraFramePositions)
        {
            return extraFramePositions.Contains(frameIndex)
                ? GetFlowEstimationTimesteps(multiplier)
                : GetFlowEstimationTimesteps(multiplier - 1);
        }


        /// <summary>
        /// Gets the flow estimation timesteps.
        /// </summary>
        /// <param name="parts">The parts.</param>
        private static float[] GetFlowEstimationTimesteps(int parts)
        {
            float[] result = new float[parts];
            for (int i = 0; i < parts; i++)
            {
                result[i] = (i + 1) / (float)(parts + 1);
            }
            return result;
        }


        /// <summary>
        /// Gets the flow estimation key frames.
        /// </summary>
        /// <param name="frameCount">The frame count.</param>
        /// <param name="multiplier">The multiplier.</param>
        private static int[] GetFlowEstimationKeyFrames(int frameCount, int multiplier)
        {
            int targetCount = frameCount * multiplier;
            int extraFramesNeeded = targetCount - (frameCount - 1) * (multiplier - 1) - frameCount;
            if (multiplier == 2)
                return [frameCount - 1];
            else if (multiplier == 3)
                return [0, frameCount / 2, frameCount - 1];
            return Enumerable.Range(0, multiplier)
                .Select(i => (int)Math.Round(i * (frameCount - 1) / (double)(multiplier - 1)))
                .Distinct()
                .ToArray();
        }


        /// <summary>
        /// Disposes this pipeline.
        /// </summary>
        public void Dispose()
        {
            _model.Dispose();
        }


        /// <summary>
        /// Creates Interpolation Pipeline for the specified provider.
        /// </summary>
        /// <param name="provider">The provider.</param>
        /// <returns>InterpolationPipeline.</returns>
        public static InterpolationPipeline Create(ExecutionProvider provider)
        {
            var config = new ModelConfig
            {
                Path = "Interpolation.onnx",
                ExecutionProvider = provider,
            };
            return Create(config);
        }


        /// <summary>
        /// Creates an custom InterpolationPipeline
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>InterpolationPipeline.</returns>
        public static InterpolationPipeline Create(ModelConfig configuration)
        {
            return new InterpolationPipeline(InterpolationModel.Create(configuration));
        }

    }
}
