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

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusion3
{
    public abstract class StableDiffusion3Base : PipelineBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Base"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="tokenizer3">The tokenizer3.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="textEncoder3">The text encoder3.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusion3Base(TransformerSD3Model transformer, CLIPTokenizer tokenizer, CLIPTokenizer tokenizer2, T5Tokenizer tokenizer3, CLIPTextModel textEncoder, CLIPTextModelWithProjection textEncoder2, T5EncoderModel textEncoder3, AutoEncoderModel autoEncoder, ILogger logger = default) : base(logger)
        {
            Transformer = transformer;
            Tokenizer = tokenizer;
            Tokenizer2 = tokenizer2;
            TextEncoder = textEncoder;
            TextEncoder2 = textEncoder2;
            AutoEncoder = autoEncoder;
            Tokenizer3 = tokenizer3;
            TextEncoder3 = textEncoder3;
            Initialize();
            Logger?.LogInformation("[StableDiffusion3Pipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Base"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusion3Base(StableDiffusion3Config configuration, ILogger logger = default) : this(
            new TransformerSD3Model(configuration.Transformer),
            new CLIPTokenizer(configuration.Tokenizer),
            new CLIPTokenizer(configuration.Tokenizer2),
            new T5Tokenizer(configuration.Tokenizer3),
            new CLIPTextModel(configuration.TextEncoder),
            new CLIPTextModelWithProjection(configuration.TextEncoder2),
            new T5EncoderModel(configuration.TextEncoder3),
            new AutoEncoderModel(configuration.AutoEncoder),
            logger)
        {
            Name = configuration.Name;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusion3;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.StableDiffusion3);

        /// <summary>
        /// Gets the tokenizer.
        /// </summary>
        public CLIPTokenizer Tokenizer { get; init; }

        /// <summary>
        /// Gets the tokenizer2.
        /// </summary>
        public CLIPTokenizer Tokenizer2 { get; init; }

        /// <summary>
        /// Gets the tokenizer3.
        /// </summary>
        public T5Tokenizer Tokenizer3 { get; init; }

        /// <summary>
        /// Gets the TextEncoder.
        /// </summary>
        public CLIPTextModel TextEncoder { get; init; }

        /// <summary>
        /// Gets the TextEncoder2.
        /// </summary>
        public CLIPTextModelWithProjection TextEncoder2 { get; init; }

        /// <summary>
        /// Gets the TextEncoder3.
        /// </summary>
        public T5EncoderModel TextEncoder3 { get; init; }

        /// <summary>
        /// Gets the transformer.
        /// </summary>
        public TransformerSD3Model Transformer { get; init; }

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
            // StableDiffusion3 pipelines are lazy loaded on first run
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
                TextEncoder3.UnloadAsync(),
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
            // Tokenizer
            var promptTokens = await TokenizePromptAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await TokenizePromptAsync(options.NegativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Tokenizer2
            var prompt2Tokens = await TokenizePrompt2Async(options.Prompt, cancellationToken);
            var negativePrompt2Tokens = await TokenizePrompt2Async(options.NegativePrompt, cancellationToken);

            // Tokenizer3
            var prompt3Tokens = await TokenizePrompt3Async(options.Prompt, cancellationToken);
            var negativePrompt3Tokens = await TokenizePrompt3Async(options.NegativePrompt, cancellationToken);

            // TextEncoder
            var prompt1Embeddings = await EncodePromptAsync(promptTokens, maxPromptTokenCount, cancellationToken);
            var negativePrompt1Embeddings = await EncodePromptAsync(negativePromptTokens, maxPromptTokenCount, cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)
                await TextEncoder.UnloadAsync();

            // TextEncoder2
            var hiddenStateIndex = 2 + options.ClipSkip;
            var prompt2Embeddings = await EncodePrompt2Async(prompt2Tokens, maxPromptTokenCount, hiddenStateIndex, cancellationToken);
            var negativePrompt2Embeddings = await EncodePrompt2Async(negativePrompt2Tokens, maxPromptTokenCount, hiddenStateIndex, cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)
                await TextEncoder2.UnloadAsync();

            // TextEncoder3
            var prompt3Embeddings = await EncodePrompt3Async(prompt3Tokens, cancellationToken);
            var negativePrompt3Embeddings = await EncodePrompt3Async(negativePrompt3Tokens, cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)
                await TextEncoder3.UnloadAsync();

            // Positive Prompt
            var promptEmbeds = prompt1Embeddings.HiddenStates;
            var promptPooledEmbeds = prompt1Embeddings.TextEmbeds;
            var promptPooledEmbeds2 = prompt2Embeddings.TextEmbeds.FirstBatch().Repeat(promptPooledEmbeds.Dimensions[0]);
            promptEmbeds = promptEmbeds.Concatenate(prompt2Embeddings.HiddenStates, 2);
            promptEmbeds = promptEmbeds.PadEnd(prompt3Embeddings.HiddenStates.Dimensions[^1] - promptEmbeds.Dimensions[^1]);
            promptEmbeds = promptEmbeds.Concatenate(prompt3Embeddings.HiddenStates, 1);
            promptPooledEmbeds = promptPooledEmbeds.Reshape([promptPooledEmbeds.Dimensions[^2], promptPooledEmbeds.Dimensions[^1]]).FirstBatch();
            promptPooledEmbeds = promptPooledEmbeds.Concatenate(promptPooledEmbeds2, 1);

            // Negative Prompt
            var negativePromptEmbeds = negativePrompt1Embeddings.HiddenStates;
            var negativePromptPooledEmbeds = negativePrompt1Embeddings.TextEmbeds;
            var negativePromptPooledEmbeds2 = negativePrompt2Embeddings.TextEmbeds.FirstBatch().Repeat(negativePromptPooledEmbeds.Dimensions[0]);
            negativePromptEmbeds = negativePromptEmbeds.Concatenate(negativePrompt2Embeddings.HiddenStates, 2);
            negativePromptEmbeds = negativePromptEmbeds.PadEnd(negativePrompt3Embeddings.HiddenStates.Dimensions[^1] - negativePromptEmbeds.Dimensions[^1]);
            negativePromptEmbeds = negativePromptEmbeds.Concatenate(negativePrompt3Embeddings.HiddenStates, 1);
            negativePromptPooledEmbeds = negativePromptPooledEmbeds.Reshape([negativePromptPooledEmbeds.Dimensions[^2], negativePromptPooledEmbeds.Dimensions[^1]]).FirstBatch();
            negativePromptPooledEmbeds = negativePromptPooledEmbeds.Concatenate(negativePromptPooledEmbeds2, 1);

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
        private async Task<TokenizerResult> TokenizePrompt2Async(string inputText, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[TokenizePrompt2Async] Begin Tokenizer2");
            var tokenizerResult = await PromptParser.TokenizePromptAsync(Tokenizer2, inputText);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[TokenizePrompt2Async] Tokenizer2 Complete");
            return tokenizerResult;
        }


        /// <summary>
        /// Tokenize prompt with Tokenizer3
        /// </summary>
        /// <param name="inputText">The input text.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected virtual async Task<TokenizerResult> TokenizePrompt3Async(string inputText, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[TokenizePrompt3Async] Begin Tokenizer3");
            var tokenizerResult = await Tokenizer3.EncodeAsync(inputText);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[TokenizePrompt3Async] Tokenizer3 Complete");
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
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <param name="hiddenStateIndex">Index of the hidden state.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<TextEncoderResult> EncodePrompt2Async(TokenizerResult inputTokens, int minimumLength, int hiddenStateIndex, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodePrompt2Async] Begin TextEncoder2");
            var textEncoderResult = await PromptParser.EncodePromptAsync(TextEncoder2, inputTokens, minimumLength, hiddenStateIndex, cancellationToken);
            var hiddenStates = textEncoderResult.HiddenStates;
            var pooledPromptEmbeds = textEncoderResult.TextEmbeds;
            int[] pooledPromptDimensions = pooledPromptEmbeds.Dimensions.Length == 2
                ? [pooledPromptEmbeds.Dimensions[0], pooledPromptEmbeds.Dimensions[1]]
                : [pooledPromptEmbeds.Dimensions[1], pooledPromptEmbeds.Dimensions[2]];
            pooledPromptEmbeds = pooledPromptEmbeds.Reshape(pooledPromptDimensions).FirstBatch();
            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodePrompt2Async] TextEncoder2 Complete");
            return new TextEncoderResult(hiddenStates, pooledPromptEmbeds);
        }


        /// <summary>
        /// Encode prompt tokens with TextEncoder3
        /// </summary>
        /// <param name="promptTokens">The prompt tokens.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        protected virtual async Task<TextEncoderResult> EncodePrompt3Async(TokenizerResult promptTokens, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodePrompt3Async] Begin TextEncoder3");
            var textEncoderResult = await TextEncoder3.RunAsync(promptTokens, cancellationToken);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodePrompt3Async] TextEncoder3 Complete");
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
            var promptEmbedsCond = prompt.PromptEmbeds;
            var pooledPromptEmbedsCond = prompt.PromptPooledEmbeds;
            var promptEmbedsUncond = prompt.NegativePromptEmbeds;
            var pooledPromptEmbedsUncond = prompt.NegativePromptPooledEmbeds;

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
                var conditional = await Transformer.RunAsync(timestep, latentInput, promptEmbedsCond, pooledPromptEmbedsCond, cancellationToken: cancellationToken);
                if (isGuidanceEnabled)
                {
                    var unconditional = await Transformer.RunAsync(timestep, latentInput, promptEmbedsUncond, pooledPromptEmbedsUncond, cancellationToken: cancellationToken);
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
        /// Run Transformer model inference with ControlNet
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<Tensor<float>> RunInferenceAsync(IPipelineOptions options, ControlNetModel controlNet, IScheduler scheduler, PromptResult prompt, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunInferenceAsync] Begin Transformer + ControlNet Inference");

            // Prompt
            var isGuidanceEnabled = IsGuidanceEnabled(options);
            var promptEmbedsCond = prompt.PromptEmbeds;
            var pooledPromptEmbedsCond = prompt.PromptPooledEmbeds;
            var promptEmbedsUncond = prompt.NegativePromptEmbeds;
            var pooledPromptEmbedsUncond = prompt.NegativePromptPooledEmbeds;

            // Latents
            var latents = await CreateLatentInputAsync(options, scheduler, cancellationToken);

            // Control Image
            var controlImage = await CreateControlInputAsync(options, cancellationToken);

            // Load Model
            await LoadTransformerAsync(options, progressCallback, cancellationToken);
            await controlNet.LoadAsync(cancellationToken: cancellationToken);

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
                    controlNet,
                    controlImage,
                    options.ControlNetStrength,
                    timestep,
                    latentInput,
                    promptEmbedsCond,
                    pooledPromptEmbedsCond,
                    cancellationToken: cancellationToken
                );

                // Guidance
                if (isGuidanceEnabled)
                {
                    var unconditional = await Transformer.RunAsync
                    (
                        controlNet,
                        controlImage,
                        options.ControlNetStrength,
                        timestep,
                        latentInput,
                        promptEmbedsUncond,
                        pooledPromptEmbedsUncond,
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
                await Task.WhenAll(Transformer.UnloadAsync(), controlNet.UnloadAsync());

            Logger.LogEnd(LogLevel.Debug, timestamp, "[RunInferenceAsync] Transformer + ControlNet Inference Complete");
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
            var dimensions = new int[] { 1, AutoEncoder.LatentChannels, options.Height / AutoEncoder.Scale, options.Width / AutoEncoder.Scale };
            var noiseTensor = scheduler.CreateRandomSample(dimensions);
            if (options.HasInputImage)
            {
                var timestep = scheduler.GetStartTimestep();
                var encoderResult = await EncodeLatentsAsync(options, options.InputImage, cancellationToken);
                return scheduler.ScaleNoise(timestep, encoderResult, noiseTensor);
            }
            return noiseTensor;
        }


        /// <summary>
        /// Creates the control input.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private Task<ImageTensor> CreateControlInputAsync(IPipelineOptions options, CancellationToken cancellationToken = default)
        {
            var controlImageTensor = options.InputControlImage.ResizeImage(options.Width, options.Height);
            if (options.ControlNet.InvertInput)
                controlImageTensor.Invert();

            return Task.FromResult(controlImageTensor.NormalizeZeroOne().AsImageTensor());
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
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Transformer.IsControlNetLoaded())
                await Transformer.UnloadControlNetAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled) && TextEncoder3.IsLoaded())
                await TextEncoder3.UnloadAsync();
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
                Shift = 3f,
                Width = 1024,
                Height = 1024,
                GuidanceScale = 3f,
                Scheduler = SchedulerType.FlowMatchEulerDiscrete
            };

            // SD3-Turbo Models , 4 Steps, No Guidance
            if (Transformer.ModelType == ModelType.Turbo)
            {
                return options with
                {
                    Steps = 4,
                    Shift = 3f,
                    Width = 1024,
                    Height = 1024,
                    GuidanceScale = 0,
                    Scheduler = SchedulerType.FlowMatchEulerDynamic
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
                Tokenizer3?.Dispose();
                TextEncoder?.Dispose();
                TextEncoder2?.Dispose();
                TextEncoder3?.Dispose();
                Transformer?.Dispose();
                AutoEncoder?.Dispose();
            }
            _disposed = true;
        }

    }
}
