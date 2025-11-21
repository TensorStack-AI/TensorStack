// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Schedulers;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.Wan
{
    public abstract class WanBase : PipelineBase
    {
        private readonly int _outputFrameRate = 16;

        /// <summary>
        /// Initializes a new instance of the <see cref="WanBase"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public WanBase(TransformerWanModel transformer, T5Tokenizer tokenizer, T5EncoderModel textEncoder, AutoEncoderModel autoEncoder, ILogger logger = default) : base(logger)
        {
            Transformer = transformer;
            Tokenizer = tokenizer;
            TextEncoder = textEncoder;
            AutoEncoder = autoEncoder;
            Initialize();
            Logger?.LogInformation("[WanPipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="WanBase"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public WanBase(WanConfig configuration, ILogger logger = default) : this(
            new TransformerWanModel(configuration.Transformer),
            new T5Tokenizer(configuration.Tokenizer),
            new T5EncoderModel(configuration.TextEncoder),
            new AutoEncoderModel(configuration.AutoEncoder),
            logger)
        {
            Name = configuration.Name;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.Wan;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.Wan);

        /// <summary>
        /// Gets the tokenizer.
        /// </summary>
        public T5Tokenizer Tokenizer { get; init; }

        /// <summary>
        /// Gets the TextEncoder.
        /// </summary>
        public T5EncoderModel TextEncoder { get; init; }

        /// <summary>
        /// Gets the transformer.
        /// </summary>
        public TransformerWanModel Transformer { get; init; }

        /// <summary>
        /// Gets the automatic encoder.
        /// </summary>
        public AutoEncoderModel AutoEncoder { get; init; }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            // Wan pipelines are lazy loaded on first run
            return Task.CompletedTask;
        }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await Task.WhenAll
            (
                Transformer.UnloadAsync(),
                TextEncoder.UnloadAsync(),
                AutoEncoder.EncoderUnloadAsync(),
                AutoEncoder.DecoderUnloadAsync()
            );
            Logger?.LogInformation("[{PipeLineType}] Pipeline Unloaded", PipelineType);
        }


        /// <summary>
        /// Validates the options.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override void ValidateOptions(GenerateOptions options)
        {
            base.ValidateOptions(options);
            if (!Transformer.HasControlNet && options.HasControlNet)
                throw new ArgumentException("Model does not support ControlNet");
        }


        /// <summary>
        /// Creates the prompt input embeddings.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<PromptResult> CreatePromptAsync(IPipelineOptions options, CancellationToken cancellationToken = default)
        {
            var cachedPrompt = GetPromptCache(options);
            if (cachedPrompt is not null)
                return cachedPrompt;

            // Tokenizer
            var conditionalTokens = await TokenizePromptAsync(options.Prompt, cancellationToken);
            var unconditionalTokens = await TokenizePromptAsync(options.NegativePrompt, cancellationToken);

            // TextEncoder
            var conditionalEmbeds = await EncodePromptAsync(conditionalTokens, cancellationToken);
            var unconditionalEmbeds = await EncodePromptAsync(unconditionalTokens, cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)
                await TextEncoder.UnloadAsync();

            return SetPromptCache(options, new PromptResult(conditionalEmbeds.HiddenStates, conditionalEmbeds.TextEmbeds, unconditionalEmbeds.HiddenStates, unconditionalEmbeds.TextEmbeds));
        }


        /// <summary>
        /// Tokenize prompt with Tokenizer3
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected virtual async Task<TokenizerResult> TokenizePromptAsync(string inputText, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[TokenizePrompt3Async] Begin Tokenizer");
            var tokenizerResult = await Tokenizer.EncodeAsync(inputText);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[TokenizePrompt3Async] Tokenizer Complete");
            return tokenizerResult;
        }


        /// <summary>
        /// Encode prompt tokens with TextEncoder
        /// </summary>
        /// <param name="promptTokens">The prompt tokens.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        protected virtual async Task<TextEncoderResult> EncodePromptAsync(TokenizerResult promptTokens, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodePrompt3Async] Begin TextEncoder3");
            var textEncoderResult = await TextEncoder.RunAsync(promptTokens, cancellationToken);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodePrompt3Async] TextEncoder3 Complete");
            return textEncoderResult;
        }


        /// <summary>
        /// Decode the model latents to video
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<VideoTensor> DecodeLatentsAsync(IPipelineOptions options, Tensor<float> latents, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[DecodeLatentsAsync] Begin AutoEncoder Decode");
            var decoderResult = await AutoEncoder.DecodeAsync(latents, disableShift: true, disableScale: true, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled)
                await AutoEncoder.DecoderUnloadAsync();

            decoderResult = decoderResult
                .Permute([0, 2, 1, 3, 4])
                .Reshape(decoderResult.Dimensions[1..]);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[DecodeLatentsAsync] AutoEncoder Decode Complete");
            return decoderResult.AsVideoTensor(_outputFrameRate);
        }


        /// <summary>
        /// Encode the image to model latents
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="image">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> EncodeLatentsAsync(IPipelineOptions options, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodeLatentsAsync] Begin AutoEncoder Encode");
            var cacheResult = GetEncoderCache(options);
            if (cacheResult is not null)
            {
                Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodeLatentsAsync] AutoEncoder Encode Complete, Cached Result.");
                return cacheResult;
            }

            var inputTensor = options.InputImage.ResizeImage(options.Width, options.Height);
            var encoderResult = await AutoEncoder.EncodeAsync(inputTensor, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled)
                await AutoEncoder.EncoderUnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodeLatentsAsync] AutoEncoder Encode Complete");
            return SetEncoderCache(options, encoderResult);
        }


        /// <summary>
        /// Run Transformer model inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<Tensor<float>> RunInferenceAsync(IPipelineOptions options, IScheduler scheduler, PromptResult prompt, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunInferenceAsync] Begin Transformer Inference");

            // Prompt
            var isGuidanceEnabled = IsGuidanceEnabled(options);
            var promptEmbedsCond = prompt.PromptEmbeds;
            var promptEmbedsUncond = prompt.NegativePromptEmbeds;

            // Latents
            var latents = await CreateLatentInputAsync(options, scheduler, cancellationToken);

            // Load Model
            await LoadTransformerAsync(options, progressCallback, cancellationToken);

            // Timesteps
            var timesteps = scheduler.GetTimesteps();
            for (int i = 0; i < timesteps.Count; i++)
            {
                var timestep = timesteps[i];
                var steptime = Stopwatch.GetTimestamp();
                cancellationToken.ThrowIfCancellationRequested();

                // Inputs.
                var latentInput = scheduler.ScaleInput(timestep, latents);

                // Inference
                var conditional = await Transformer.RunAsync(timestep, latentInput, promptEmbedsCond, cancellationToken: cancellationToken);
                if (isGuidanceEnabled)
                {
                    var unconditional = await Transformer.RunAsync(timestep, latentInput, promptEmbedsUncond, cancellationToken: cancellationToken);
                    conditional = ApplyGuidance(conditional, unconditional, options.GuidanceScale);
                }

                // Scheduler
                var stepResult = scheduler.Step(timestep, conditional, latents);

                // Result
                latents = stepResult.Sample;

                // Progress
                if (scheduler.IsFinalOrder)
                    progressCallback.Notify(scheduler.CurrentStep, scheduler.TotalSteps, latents, steptime);

                Logger.LogEnd(LogLevel.Debug, steptime, $"[RunInferenceAsync] Step: {i + 1}/{timesteps.Count}");
            }

            // Unload
            if (options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled)
                await Transformer.UnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[RunInferenceAsync] Transformer Inference Complete");
            return latents;
        }


        /// <summary>
        /// Create latent input.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> CreateLatentInputAsync(IPipelineOptions options, IScheduler scheduler, CancellationToken cancellationToken = default)
        {
            var dimensions = new int[] { 1, AutoEncoder.LatentChannels, 21, options.Height / AutoEncoder.Scale, options.Width / AutoEncoder.Scale };
            var noiseTensor = scheduler.CreateRandomSample(dimensions);
            if (options.HasInputImage)
            {
                var timestep = scheduler.GetStartTimestep();
                var encoderResult = await EncodeLatentsAsync(options, cancellationToken);
                return scheduler.ScaleNoise(timestep, encoderResult, noiseTensor);
            }
            return noiseTensor;
        }


        /// <summary>
        /// Gets the model optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        private ModelOptimization GetOptimizations(IPipelineOptions generateOptions, IProgress<GenerateProgress> progressCallback = null)
        {
            var optimizations = new ModelOptimization(Optimization.None);
            if (Transformer.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }
            return optimizations;
        }


        /// <summary>
        /// Determines whether classifier-free guidance is enabled
        /// </summary>
        /// <param name="options">The options.</param>
        private bool IsGuidanceEnabled(IPipelineOptions options)
        {
            return options.GuidanceScale > 1;
        }


        /// <summary>
        /// Load Transformer with optimizations
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task LoadTransformerAsync(IPipelineOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var optimizations = GetOptimizations(options, progressCallback);
            await Transformer.LoadAsync(optimizations, cancellationToken);
        }


        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override async Task CheckPipelineState(IPipelineOptions options)
        {
            // Check Transformer/ControlNet status
            if (options.HasControlNet && Transformer.IsLoaded())
                await Transformer.UnloadAsync();
            if (!options.HasControlNet && Transformer.IsControlNetLoaded())
                await Transformer.UnloadControlNetAsync();

            // Check LowMemory status
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled) && TextEncoder.IsLoaded())
                await TextEncoder.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Transformer.IsLoaded())
                await Transformer.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled) && AutoEncoder.IsEncoderLoaded())
                await AutoEncoder.EncoderUnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled) && AutoEncoder.IsDecoderLoaded())
                await AutoEncoder.DecoderUnloadAsync();
        }


        /// <summary>
        /// Configures the supported schedulers.
        /// </summary>
        protected override IReadOnlyList<SchedulerType> ConfigureSchedulers()
        {
            return [SchedulerType.FlowMatchEulerDiscrete, SchedulerType.FlowMatchEulerDynamic];
        }


        /// <summary>
        /// Configures the default SchedulerOptions.
        /// </summary>
        protected override GenerateOptions ConfigureDefaultOptions()
        {
            var options = new GenerateOptions
            {
                Steps = 50,
                Shift = 3f,
                Width = 832,
                Height = 480,
                GuidanceScale = 5f,
                Scheduler = SchedulerType.FlowMatchEulerDiscrete
            };

            return options;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        private bool _disposed;
        protected override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            if (disposing)
            {
                Tokenizer?.Dispose();
                TextEncoder?.Dispose();
                Transformer?.Dispose();
                AutoEncoder?.Dispose();
            }
            _disposed = true;
        }

    }
}
