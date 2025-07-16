// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Helpers;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Schedulers;
using TensorStack.StableDiffusion.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusionXL
{
    public class StableDiffusionXLPipeline : PipelineBase, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLPipeline"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionXLPipeline(UNetConditionalModel unet, CLIPTokenizer tokenizer, CLIPTokenizer tokenizer2, CLIPTextModel textEncoder, CLIPTextModelWithProjection textEncoder2, AutoEncoderModel autoEncoder, ILogger logger = default) : base(logger)
        {
            Unet = unet;
            Tokenizer = tokenizer;
            Tokenizer2 = tokenizer2;
            TextEncoder = textEncoder;
            TextEncoder2 = textEncoder2;
            AutoEncoder = autoEncoder;
            Initialize();
            Logger?.LogInformation("[StableDiffusionXLPipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionXLPipeline(StableDiffusionXLConfig configuration, ILogger logger = default) : this(
            new UNetConditionalModel(configuration.Unet),
            new CLIPTokenizer(configuration.Tokenizer),
            new CLIPTokenizer(configuration.Tokenizer2),
            new CLIPTextModel(configuration.TextEncoder),
            new CLIPTextModelWithProjection(configuration.TextEncoder2),
            new AutoEncoderModel(configuration.AutoEncoder),
            logger)
        {
            Name = configuration.Name;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusionXL;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.StableDiffusionXL);

        /// <summary>
        /// Gets the tokenizer.
        /// </summary>
        public CLIPTokenizer Tokenizer { get; init; }

        /// <summary>
        /// Gets the tokenizer.
        /// </summary>
        public CLIPTokenizer Tokenizer2 { get; init; }

        /// <summary>
        /// Gets the text encoder.
        /// </summary>
        public CLIPTextModel TextEncoder { get; init; }

        /// <summary>
        /// Gets the text encoder.
        /// </summary>
        public CLIPTextModelWithProjection TextEncoder2 { get; init; }

        /// <summary>
        /// Gets the unet.
        /// </summary>
        public UNetConditionalModel Unet { get; init; }

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
            // StableDiffusionXL pipelines are lazy loaded on first run
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
                Unet.UnloadAsync(),
                TextEncoder.UnloadAsync(),
                TextEncoder2.UnloadAsync(),
                AutoEncoder.EncoderUnloadAsync(),
                AutoEncoder.DecoderUnloadAsync()
            );
            Logger?.LogInformation("[{PipeLineType}] Pipeline Unloaded", PipelineType);
        }


        /// <summary>
        /// Run ImageTensor pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<ImageTensor> RunAsync(GenerateOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            ValidateOptions(options);

            var prompt = await CreatePromptAsync(options, cancellationToken);
            using (var scheduler = CreateScheduler(options))
            {
                var latents = !options.HasControlNet
                    ? await RunInferenceAsync(options, scheduler, prompt, progressCallback, cancellationToken)
                    : await RunInferenceAsync(options, options.ControlNet, scheduler, prompt, progressCallback, cancellationToken);
                return await DecodeLatentsAsync(options, latents, cancellationToken);
            }
        }


        /// <summary>
        /// Validates the options.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override void ValidateOptions(GenerateOptions options)
        {
            base.ValidateOptions(options);
            if (!Unet.HasControlNet && options.HasControlNet)
                throw new ArgumentException("Model does not support ControlNet");
        }


        /// <summary>
        /// Creates the prompt input embeddings.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<PromptResult> CreatePromptAsync(IPipelineOptions options, CancellationToken cancellationToken = default)
        {
            // Tokenizer
            var promptTokens = await TokenizePromptAsync(options.Prompt, cancellationToken);
            var negativePromptTokens = await TokenizePromptAsync(options.NegativePrompt, cancellationToken);
            var maxPromptTokenCount = Math.Max(promptTokens.InputIds.Length, negativePromptTokens.InputIds.Length);

            // Tokenizer2
            var prompt2Tokens = await TokenizePrompt2Async(options.Prompt, cancellationToken);
            var negativePrompt2Tokens = await TokenizePrompt2Async(options.NegativePrompt, cancellationToken);

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

            // Prompt embeds
            var pooledPromptEmbeds = prompt2Embeddings.TextEmbeds;
            var promptEmbeddings = prompt1Embeddings.HiddenStates.Concatenate(prompt2Embeddings.HiddenStates, 2);

            // Negative Prompt embeds
            var pooledNegativePromptEmbeds = negativePrompt2Embeddings.TextEmbeds;
            var negativePromptEmbeddings = negativePrompt1Embeddings.HiddenStates.Concatenate(negativePrompt2Embeddings.HiddenStates, 2);

            return new PromptResult(promptEmbeddings, pooledPromptEmbeds, negativePromptEmbeddings, pooledNegativePromptEmbeds);
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
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[TokenizePrompt2Async] Begin Tokenizer");
            var tokenizerResult = await PromptParser.TokenizePromptAsync(Tokenizer2, inputText);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[TokenizePrompt2Async] Tokenizer Complete");
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
            var textEncoderResult = await PromptParser.EncodePromptAsync(TextEncoder2, inputTokens, minimumLength, hiddenStateIndex, cancellationToken);
            var hiddenStates = textEncoderResult.HiddenStates;
            var pooledPromptEmbeds = textEncoderResult.TextEmbeds;
            int[] pooledPromptDimensions = pooledPromptEmbeds.Dimensions.Length == 2
                ? [pooledPromptEmbeds.Dimensions[0], pooledPromptEmbeds.Dimensions[1]]
                : [pooledPromptEmbeds.Dimensions[1], pooledPromptEmbeds.Dimensions[2]];
            pooledPromptEmbeds = pooledPromptEmbeds.Reshape(pooledPromptDimensions).FirstBatch();
            return new TextEncoderResult(hiddenStates, pooledPromptEmbeds);
        }


        /// <summary>
        /// Decode the model latents to image
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<ImageTensor> DecodeLatentsAsync(IPipelineOptions options, Tensor<float> latents, CancellationToken cancellationToken = default)
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
        /// Run UNET model inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> RunInferenceAsync(IPipelineOptions options, IScheduler scheduler, PromptResult prompt, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunInferenceAsync] Begin Unet Inference");

            // Prompt
            var isGuidanceEnabled = IsGuidanceEnabled(options);
            var promptEmbedsCond = prompt.PromptEmbeds;
            var pooledPromptEmbedsCond = prompt.PromptPooledEmbeds;
            var promptEmbedsUncond = prompt.NegativePromptEmbeds;
            var pooledPromptEmbedsUncond = prompt.NegativePromptPooledEmbeds;

            // Latents
            var latents = await CreateLatentInputAsync(options, scheduler, cancellationToken);

            // Get TimeIds
            var timeIds = GetAddTimeIds(options);

            // Load Model
            await LoadUnetAsync(options, progressCallback, cancellationToken);

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
                var conditional = await Unet.RunAsync(timestep, latentInput, promptEmbedsCond, pooledPromptEmbedsCond, timeIds, cancellationToken: cancellationToken);
                if (isGuidanceEnabled)
                {
                    var unconditional = await Unet.RunAsync(timestep, latentInput, promptEmbedsUncond, pooledPromptEmbedsUncond, timeIds, cancellationToken: cancellationToken);
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
                await Unet.UnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[RunInferenceAsync] Unet Inference Complete");
            return latents;
        }


        /// <summary>
        /// Run UNET model inference with ControlNet
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="controlNet">The control net.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> RunInferenceAsync(IPipelineOptions options, ControlNetModel controlNet, IScheduler scheduler, PromptResult prompt, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunInferenceAsync] Begin Unet + ControlNet Inference");

            // Prompt
            var isGuidanceEnabled = IsGuidanceEnabled(options);
            var promptEmbedsCond = prompt.PromptEmbeds;
            var pooledPromptEmbedsCond = prompt.PromptPooledEmbeds;
            var promptEmbedsUncond = prompt.NegativePromptEmbeds;
            var pooledPromptEmbedsUncond = prompt.NegativePromptPooledEmbeds;

            // Latents
            var latents = await CreateLatentInputAsync(options, scheduler, cancellationToken);

            // Get TimeIds
            var timeIds = GetAddTimeIds(options);

            // Control Image
            var controlImage = await CreateControlInputAsync(options, cancellationToken);

            // Load Model
            await LoadUnetAsync(options, progressCallback, cancellationToken);
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
                var conditional = await Unet.RunAsync
                (
                    controlNet,
                    controlImage,
                    options.ControlNetStrength,
                    timestep, 
                    latentInput, 
                    promptEmbedsCond, 
                    pooledPromptEmbedsCond,
                    timeIds,
                    cancellationToken: cancellationToken
                );

                // Guidance
                if (isGuidanceEnabled)
                {
                    var unconditional = await Unet.RunAsync
                    (
                        controlNet,
                        controlImage,
                        options.ControlNetStrength,
                        timestep, 
                        latentInput,
                        promptEmbedsUncond,
                        pooledPromptEmbedsUncond,
                        timeIds, 
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
                await Task.WhenAll(Unet.UnloadAsync(), controlNet.UnloadAsync());

            Logger.LogEnd(LogLevel.Debug, timestamp, "[RunInferenceAsync] Unet + ControlNet Inference Complete");
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
            return noiseTensor.Multiply(scheduler.StartSigma);
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
        /// Gets the add AddTimeIds.
        /// </summary>
        /// <param name="options">The pipeline options.</param>
        /// <returns></returns>
        protected Tensor<float> GetAddTimeIds(IPipelineOptions options)
        {
            float[] result = Unet.ModelType == ModelType.Refiner
                ? [options.Height, options.Width, 0, 0, options.AestheticScore]
                : [options.Height, options.Width, 0, 0, options.Height, options.Width];
            return new Tensor<float>(result, [1, result.Length]);
        }


        /// <summary>
        /// Gets the model optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        private ModelOptimization GetOptimizations(IPipelineOptions generateOptions, IProgress<GenerateProgress> progressCallback = null)
        {
            var optimizations = new ModelOptimization(Optimization.None);
            if (Unet.HasOptimizationsChanged(optimizations))
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
        /// Load unet with optimizations
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task LoadUnetAsync(IPipelineOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var optimizations = GetOptimizations(options, progressCallback);
            await Unet.LoadAsync(optimizations, cancellationToken);
        }


        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override async Task CheckPipelineState(IPipelineOptions options)
        {
            // Check Unet/ControlNet status
            if (options.HasControlNet && Unet.IsLoaded())
                await Unet.UnloadAsync();
            if (!options.HasControlNet && Unet.IsControlNetLoaded())
                await Unet.UnloadControlNetAsync();

            // Check LowMemory status
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled) && TextEncoder.IsLoaded())
                await TextEncoder.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled) && TextEncoder2.IsLoaded())
                await TextEncoder2.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Unet.IsLoaded())
                await Unet.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Unet.IsControlNetLoaded())
                await Unet.UnloadControlNetAsync();
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
            return
            [
                SchedulerType.LMS,
                SchedulerType.Euler,
                SchedulerType.EulerAncestral,
                SchedulerType.DDPM,
                SchedulerType.DDIM,
                SchedulerType.KDPM2,
                SchedulerType.KDPM2Ancestral,
                SchedulerType.LCM
            ];
        }


        /// <summary>
        /// Configures the default SchedulerOptions.
        /// </summary>
        protected override GenerateOptions ConfigureDefaultOptions()
        {
            var options = new GenerateOptions
            {
                Steps = 28,
                Width = 1024,
                Height = 1024,
                GuidanceScale = 5f,
                Scheduler = SchedulerType.DDPM,
                TimestepSpacing = TimestepSpacingType.Trailing
            };

            // SDXL-Turbo Models , 4 Steps, No Guidance
            if (Unet.ModelType == ModelType.Turbo)
            {
                return options with
                {
                    Steps = 4,
                    Width = 512,
                    Height = 512,
                    GuidanceScale = 0,
                    Scheduler = SchedulerType.EulerAncestral
                };
            }

            return options;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Tokenizer?.Dispose();
            Tokenizer2?.Dispose();
            TextEncoder?.Dispose();
            TextEncoder2?.Dispose();
            Unet?.Dispose();
            AutoEncoder?.Dispose();
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// Create StableDiffusionXL pipeline from StableDiffusionConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionXLPipeline.</returns>
        public static StableDiffusionXLPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionXLPipeline(StableDiffusionXLConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusionXL pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionXLPipeline.</returns>
        public static StableDiffusionXLPipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionXLPipeline(StableDiffusionXLConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }
    }
}
