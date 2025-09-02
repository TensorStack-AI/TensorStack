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
using TensorStack.StableDiffusion.Helpers;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Schedulers;
using TensorStack.StableDiffusion.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.Flux
{
    public abstract class FluxBase : PipelineBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FluxBase"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public FluxBase(TransformerFluxModel transformer, CLIPTokenizer tokenizer, T5Tokenizer tokenizer2, CLIPTextModel textEncoder, T5EncoderModel textEncoder2, AutoEncoderModel autoEncoder, ILogger logger = default) : base(logger)
        {
            Transformer = transformer;
            Tokenizer = tokenizer;
            Tokenizer2 = tokenizer2;
            TextEncoder = textEncoder;
            TextEncoder2 = textEncoder2;
            AutoEncoder = autoEncoder;
            Initialize();
            Logger?.LogInformation("[FluxPipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="FluxBase"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public FluxBase(FluxConfig configuration, ILogger logger = default) : this(
            new TransformerFluxModel(configuration.Transformer),
            new CLIPTokenizer(configuration.Tokenizer),
            new T5Tokenizer(configuration.Tokenizer2),
            new CLIPTextModel(configuration.TextEncoder),
            new T5EncoderModel(configuration.TextEncoder2),
            new AutoEncoderModel(configuration.AutoEncoder),
            logger)
        {
            Name = configuration.Name;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.Flux;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.Flux);

        /// <summary>
        /// Gets the tokenizer.
        /// </summary>
        public CLIPTokenizer Tokenizer { get; init; }

        /// <summary>
        /// Gets the tokenizer2.
        /// </summary>
        public T5Tokenizer Tokenizer2 { get; init; }

        /// <summary>
        /// Gets the TextEncoder.
        /// </summary>
        public CLIPTextModel TextEncoder { get; init; }

        /// <summary>
        /// Gets the TextEncoder2.
        /// </summary>
        public T5EncoderModel TextEncoder2 { get; init; }

        /// <summary>
        /// Gets the transformer.
        /// </summary>
        public TransformerFluxModel Transformer { get; init; }

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
            // Flux pipelines are lazy loaded on first run
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
                TextEncoder2.UnloadAsync(),
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
            // Tokenize2
            var promptTokens = await TokenizePromptAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await TokenizePromptAsync(options.NegativePrompt, cancellationToken);
            var maxTokenLength = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Tokenizer2
            var prompt2Tokens = await TokenizePrompt2Async(options.Prompt, cancellationToken);
            var negativePrompt2Tokens = await TokenizePrompt2Async(options.NegativePrompt, cancellationToken);

            // TextEncoder
            var promptEmbeddings = await EncodePromptAsync(promptTokens, maxTokenLength, cancellationToken);
            var negativePromptEmbeddings = await EncodePromptAsync(negativePromptTokens, maxTokenLength, cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)
                await TextEncoder.UnloadAsync();

            // TextEncoder2
            var prompt2Embeddings = await EncodePrompt2Async(prompt2Tokens, cancellationToken);
            var negativePrompt2Embeddings = await EncodePrompt2Async(negativePrompt2Tokens, cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)
                await TextEncoder2.UnloadAsync();

            // Prompt
            var promptEmbeds = prompt2Embeddings.HiddenStates;
            var promptPooledEmbeds = promptEmbeddings.TextEmbeds;
            promptPooledEmbeds = promptPooledEmbeds.Reshape([promptPooledEmbeds.Dimensions[^2], promptPooledEmbeds.Dimensions[^1]]).FirstBatch();

            // Negative promt
            var negativePromptEmbeds = negativePrompt2Embeddings.HiddenStates;
            var negativePromptPooledEmbeds = negativePromptEmbeddings.TextEmbeds;
            negativePromptPooledEmbeds = negativePromptPooledEmbeds.Reshape([negativePromptPooledEmbeds.Dimensions[^2], negativePromptPooledEmbeds.Dimensions[^1]]).FirstBatch();

            return new PromptResult(promptEmbeds, promptPooledEmbeds, negativePromptEmbeds, negativePromptPooledEmbeds);
        }


        /// <summary>
        /// Tokenize prompt
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<TokenizerResult> TokenizePromptAsync(string inputText, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[TokenizePromptAsync] Begin Tokenizer");
            var tokenizerResult = await PromptParser.TokenizePromptAsync(Tokenizer, inputText);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[TokenizePromptAsync] Tokenizer Complete");
            return tokenizerResult;
        }


        /// <summary>
        /// Tokenize prompt with Tokenizer2
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected virtual async Task<TokenizerResult> TokenizePrompt2Async(string inputText, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[TokenizePrompt2Async] Begin Tokenizer2");
            var tokenizerResult = await Tokenizer2.EncodeAsync(inputText);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[TokenizePrompt2Async] Tokenizer2 Complete");
            return tokenizerResult;
        }


        /// <summary>
        /// Encode prompt tokens.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<TextEncoderResult> EncodePromptAsync(TokenizerResult inputTokens, int minimumLength, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodePromptAsync] Begin TextEncoder");
            var textEncoderResult = await PromptParser.EncodePromptAsync(TextEncoder, inputTokens, minimumLength, 0, cancellationToken);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodePromptAsync] TextEncoder Complete");
            return textEncoderResult;
        }


        /// <summary>
        /// Encode prompt tokens with TextEncoder2
        /// </summary>
        /// <param name="promptTokens">The prompt tokens.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        protected virtual async Task<TextEncoderResult> EncodePrompt2Async(TokenizerResult promptTokens, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodePrompt2Async] Begin TextEncoder2");
            var textEncoderResult = await TextEncoder2.RunAsync(promptTokens, cancellationToken);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodePrompt2Async] TextEncoder2 Complete");
            return textEncoderResult;
        }


        /// <summary>
        /// Decode the model latents to image
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<ImageTensor> DecodeLatentsAsync(IPipelineOptions options, Tensor<float> latents, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[DecodeLatentsAsync] Begin AutoEncoder Decode");
            var decoderResult = await AutoEncoder.DecodeAsync(latents, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled)
                await AutoEncoder.DecoderUnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[DecodeLatentsAsync] AutoEncoder Decode Complete");
            return decoderResult.AsImageTensor();
        }


        /// <summary>
        /// Encode the image to model latents
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="image">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> EncodeLatentsAsync(IPipelineOptions options, ImageTensor image, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodeLatentsAsync] Begin AutoEncoder Encode");
            var inputTensor = image.ResizeImage(options.Width, options.Height);
            var encoderResult = await AutoEncoder.EncodeAsync(inputTensor, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled)
                await AutoEncoder.EncoderUnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodeLatentsAsync] AutoEncoder Encode Complete");
            return encoderResult;
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
            var conditionalEmbeds = prompt.PromptEmbeds;
            var conditionalPooledEmbeds = prompt.PromptPooledEmbeds;
            var unconditionalEmbeds = prompt.NegativePromptEmbeds;
            var unconditionalPooledEmbeds = prompt.NegativePromptPooledEmbeds;

            // Latents
            var latents = await CreateLatentInputAsync(options, scheduler, cancellationToken);

            // Create ImageIds
            var imgIds = CreateLatentImageIds(options);

            // Create TextIds
            var txtIds = CreateLatentTextIds(conditionalEmbeds);

            // Guidance
            var guidanceScale = options.GuidanceScale2;

            // Load Model
            var metadata = await LoadTransformerAsync(options, progressCallback, cancellationToken);

            // Legacy Models
            if (metadata.Inputs[4].Dimensions.Length == 3)
            {
                imgIds = imgIds.Reshape([1, .. imgIds.Dimensions]);
                txtIds = txtIds.Reshape([1, .. txtIds.Dimensions]);
            }

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
                var conditional = await Transformer.RunAsync
                (
                    timestep,
                    latentInput,
                    conditionalEmbeds,
                    conditionalPooledEmbeds,
                    imgIds,
                    txtIds,
                    guidanceScale,
                    cancellationToken: cancellationToken
                );

                // Guidance
                if (isGuidanceEnabled)
                {
                    var unconditional = await Transformer.RunAsync
                    (
                        timestep,
                        latentInput,
                        unconditionalEmbeds,
                        unconditionalPooledEmbeds,
                        imgIds,
                        txtIds,
                        guidanceScale,
                        cancellationToken: cancellationToken
                    );
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
            return UnpackLatents(latents, options.Width, options.Height);
        }



        /// <summary>
        /// Create latent input.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> CreateLatentInputAsync(IPipelineOptions options, IScheduler scheduler, CancellationToken cancellationToken = default)
        {
            if (options.HasInputImage)
            {
                var timestep = scheduler.GetStartTimestep();
                var encoderResult = await EncodeLatentsAsync(options, options.InputImage, cancellationToken);
                var noiseTensor = scheduler.CreateRandomSample(encoderResult.Dimensions);
                return PackLatents(scheduler.ScaleNoise(timestep, encoderResult, noiseTensor));
            }

            var height = options.Height * 2 / AutoEncoder.LatentChannels;
            var width = options.Width * 2 / AutoEncoder.LatentChannels;
            return PackLatents(scheduler.CreateRandomSample([1, AutoEncoder.LatentChannels, height, width]));
        }


        /// <summary>
        /// Prepares the latent image ids.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected Tensor<float> CreateLatentImageIds(IPipelineOptions options)
        {
            var height = options.Height  / AutoEncoder.LatentChannels;
            var width = options.Width  / AutoEncoder.LatentChannels;
            var latentIds = new Tensor<float>([height, width, 3]);

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    latentIds[i, j, 1] += i;

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    latentIds[i, j, 2] += j;

            return latentIds.Reshape([latentIds.Dimensions[0] * latentIds.Dimensions[1], latentIds.Dimensions[2]]);
        }


        /// <summary>
        /// Prepares the latent text ids.
        /// </summary>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        protected Tensor<float> CreateLatentTextIds(Tensor<float> promptEmbeds)
        {
            return new Tensor<float>([promptEmbeds.Dimensions[1], 3]);
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
        private async Task<ModelMetadata> LoadTransformerAsync(IPipelineOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var optimizations = GetOptimizations(options, progressCallback);
            return await Transformer.LoadAsync(optimizations, cancellationToken);
        }

        /// <summary>
        /// Packs the latents.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected Tensor<float> PackLatents(Tensor<float> latents)
        {
            var height = latents.Dimensions[2] / 2;
            var width = latents.Dimensions[3] / 2;
            latents = latents.Reshape([1, AutoEncoder.LatentChannels, height, 2, width, 2]);
            latents = latents.Permute([0, 2, 4, 1, 3, 5]);
            latents = latents.Reshape([1, height * width, AutoEncoder.LatentChannels * 4]);
            return latents;
        }


        /// <summary>
        /// Unpacks the latents.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        protected Tensor<float> UnpackLatents(Tensor<float> latents, int width, int height)
        {
            var channels = latents.Dimensions[2];
            height = height / AutoEncoder.LatentChannels;
            width = width / AutoEncoder.LatentChannels;
            latents = latents.Reshape([1, height, width, channels / 4, 2, 2]);
            latents = latents.Permute([0, 3, 1, 4, 2, 5]);
            latents = latents.Reshape([1, channels / (2 * 2), height * 2, width * 2]);
            return latents;
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
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Transformer.IsControlNetLoaded())
                await Transformer.UnloadControlNetAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled) && TextEncoder2.IsLoaded())
                await TextEncoder2.UnloadAsync();
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
                Steps = 28,
                Shift = 1f,
                Width = 1024,
                Height = 1024,
                GuidanceScale = 0f,
                GuidanceScale2 = 3.5f,
                Scheduler = SchedulerType.FlowMatchEulerDiscrete
            };

            // SD3-Turbo Models , 4 Steps, No Guidance
            if (Transformer.ModelType == ModelType.Turbo)
            {
                return options with
                {
                    Steps = 4,
                    Shift = 1f,
                    Width = 1024,
                    Height = 1024,
                    GuidanceScale = 0,
                    GuidanceScale2 = 0,
                    Scheduler = SchedulerType.FlowMatchEulerDiscrete
                };
            }

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
                Tokenizer2?.Dispose();
                TextEncoder?.Dispose();
                TextEncoder2?.Dispose();
                Transformer?.Dispose();
                AutoEncoder?.Dispose();
            }
            _disposed = true;
        }
    }
}
