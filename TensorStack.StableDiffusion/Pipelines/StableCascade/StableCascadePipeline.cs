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

namespace TensorStack.StableDiffusion.Pipelines.StableCascade
{
    public class StableCascadePipeline : PipelineBase, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        private readonly float _latentDimScale = 10.67f;
        private readonly float _resolutionMultiple = 42.67f;

        /// <summary>
        /// Initializes a new instance of the <see cref="StableCascadePipeline"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableCascadePipeline(StableCascadeUNet priorUent, StableCascadeUNet decoderUnet, CLIPTokenizer tokenizer, CLIPTextModelWithProjection textEncoder, PaellaVQModel imageDecoder, CLIPVisionModelWithProjection imageEncoder, ILogger logger = default) : base(logger)
        {
            PriorUnet = priorUent;
            DecoderUnet = decoderUnet;
            Tokenizer = tokenizer;
            TextEncoder = textEncoder;
            ImageDecoder = imageDecoder;
            ImageEncoder = imageEncoder;
            Initialize();
            Logger?.LogInformation("[StableCascadePipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="StableCascadePipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public StableCascadePipeline(StableCascadeConfig configuration, ILogger logger = default) : this(
            new StableCascadeUNet(configuration.PriorUnet),
            new StableCascadeUNet(configuration.DecoderUnet),
            new CLIPTokenizer(configuration.Tokenizer),
            new CLIPTextModelWithProjection(configuration.TextEncoder),
            new PaellaVQModel(configuration.ImageDecoder),
            new CLIPVisionModelWithProjection(configuration.ImageEncoder),
            logger)
        {
            Name = configuration.Name;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableCascade;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.StableCascade);

        /// <summary>
        /// Gets the tokenizer.
        /// </summary>
        public CLIPTokenizer Tokenizer { get; init; }

        /// <summary>
        /// Gets the text encoder.
        /// </summary>
        public CLIPTextModelWithProjection TextEncoder { get; init; }

        /// <summary>
        /// Gets the prior unet.
        /// </summary>
        public StableCascadeUNet PriorUnet { get; init; }

        /// <summary>
        /// Gets the decoder unet.
        /// </summary>
        public StableCascadeUNet DecoderUnet { get; init; }

        /// <summary>
        /// Gets the image decoder.
        /// </summary>
        public PaellaVQModel ImageDecoder { get; init; }

        /// <summary>
        /// Gets the image encoder.
        /// </summary>
        public CLIPVisionModelWithProjection ImageEncoder { get; init; }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            // StableCascade pipelines are lazy loaded on first run
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
                PriorUnet.UnloadAsync(),
                DecoderUnet.UnloadAsync(),
                TextEncoder.UnloadAsync(),
                ImageDecoder.UnloadAsync(),
                ImageEncoder.UnloadAsync()
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

            // Prior Unet
            var priorPerformGuidance = options.GuidanceScale > 0;
            var priorPromptEmbeds = prompt.GetPromptEmbeds(priorPerformGuidance);
            var priorPooledPromptEmbeds = prompt.GetPromptPooledEmbeds(priorPerformGuidance);
            var priorLatents = await RunPriorAsync(options, priorPromptEmbeds, priorPooledPromptEmbeds, priorPerformGuidance, progressCallback, cancellationToken);

            // Decoder Unet
            var decodeSchedulerOptions = options with
            {
                Steps = options.Steps2,
                GuidanceScale = options.GuidanceScale2
            };
            var decoderPerformGuidance = decodeSchedulerOptions.GuidanceScale > 0;
            var decoderPromptEmbeds = prompt.GetPromptEmbeds(decoderPerformGuidance);
            var decoderPooledPromptEmbeds = prompt.GetPromptPooledEmbeds(decoderPerformGuidance);
            var decoderLatents = await RunDecoderAsync(decodeSchedulerOptions, priorLatents, decoderPromptEmbeds, decoderPooledPromptEmbeds, decoderPerformGuidance, progressCallback, cancellationToken);

            // Decode Latents
            return await DecodeLatentsAsync(options, decoderLatents, cancellationToken);
        }


        /// <summary>
        /// Validates the options.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override void ValidateOptions(GenerateOptions options)
        {
            base.ValidateOptions(options);
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

            // TextEncoder
            var hiddenStateIndex = 1 + options.ClipSkip;
            var promptEmbeddings = await EncodePromptAsync(promptTokens, maxPromptTokenCount, hiddenStateIndex, cancellationToken);
            var negativePromptEmbeddings = await EncodePromptAsync(negativePromptTokens, maxPromptTokenCount, hiddenStateIndex, cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)
                await TextEncoder.UnloadAsync();

            var textEmbeds = promptEmbeddings.TextEmbeds.Rank == 3
                ? promptEmbeddings.TextEmbeds
                : promptEmbeddings.TextEmbeds.Reshape([1, .. promptEmbeddings.TextEmbeds.Dimensions]);

            var negativeTextEmbeds = negativePromptEmbeddings.TextEmbeds.Rank == 3
               ? negativePromptEmbeddings.TextEmbeds
               : negativePromptEmbeddings.TextEmbeds.Reshape([1, .. negativePromptEmbeddings.TextEmbeds.Dimensions]);

            return new PromptResult(promptEmbeddings.HiddenStates, textEmbeds, negativePromptEmbeddings.HiddenStates, negativeTextEmbeds);
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
        /// Encode prompt tokens.
        /// </summary>
        /// <param name="inputTokens">The input tokens.</param>
        /// <param name="minimumLength">The minimum length.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<TextEncoderResult> EncodePromptAsync(TokenizerResult inputTokens, int minimumLength, int hiddenStateIndex, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodePromptAsync] Begin TextEncoder");
            var textEncoderResult = await PromptParser.EncodePromptAsync(TextEncoder, inputTokens, minimumLength, hiddenStateIndex, cancellationToken);
            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodePromptAsync] TextEncoder Complete");
            return textEncoderResult;
        }


        /// <summary>
        /// Run Prior Unet
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="promptEmbeds">The prompt embeds.</param>
        /// <param name="pooledPromptEmbeds">The pooled prompt embeds.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> RunPriorAsync(GenerateOptions options, Tensor<float> promptEmbeds, Tensor<float> pooledPromptEmbeds, bool performGuidance, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunPriorAsync] Begin Prior Inference");
            using (var scheduler = CreateScheduler(options))
            {
                // Create latent sample
                var latents = await CreatePriorLatentInputAsync(options, scheduler, cancellationToken);

                var image = await EncodeLatentsAsync(options, options.InputImage, cancellationToken);

                // Get Model metadata
                var metadata = await PriorUnet.LoadAsync(cancellationToken: cancellationToken);

                // Timesteps
                var timesteps = scheduler.GetTimesteps();
                for (int i = 0; i < timesteps.Count; i++)
                {
                    var timestep = timesteps[i];
                    var steptime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Inputs.
                    var inputImage = image.WithGuidance(performGuidance);
                    var inputLatent = scheduler.ScaleInput(timestep, latents).WithGuidance(performGuidance);

                    // Inference
                    using (var modelParameters = new ModelParameters(metadata, cancellationToken))
                    {
                        modelParameters.AddInput(inputLatent.AsTensorSpan());
                        modelParameters.AddScalarInput(timestep / 1000f);
                        modelParameters.AddInput(pooledPromptEmbeds.AsTensorSpan());
                        modelParameters.AddInput(promptEmbeds.AsTensorSpan());
                        modelParameters.AddInput(inputImage.AsTensorSpan());
                        modelParameters.AddOutput(inputLatent.Dimensions);

                        using (var results = await PriorUnet.RunInferenceAsync(modelParameters))
                        {
                            var prediction = results[0].ToTensor();

                            // Perform guidance
                            if (performGuidance)
                                prediction = ApplyGuidance(prediction, options.GuidanceScale);

                            // Scheduler Step
                            var stepResult = scheduler.Step(timestep, prediction, latents);

                            // Result
                            latents = stepResult.Sample;

                            // Progress
                            if (scheduler.IsFinalOrder)
                                progressCallback.Notify(scheduler.CurrentStep, scheduler.TotalSteps, latents, steptime);

                            Logger.LogEnd(LogLevel.Debug, steptime, $"[RunPriorAsync] Step: {i + 1}/{timesteps.Count}");
                        }
                    }
                }

                // Unload if required
                if (options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled)
                    await PriorUnet.UnloadAsync();

                Logger.LogEnd(LogLevel.Debug, timestamp, "[RunPriorAsync] Prior Inference Complete");
                return latents;
            }
        }


        /// <summary>
        /// Run Decoder Unet.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="priorLatents">The prior latents.</param>
        /// <param name="promptEmbeds">The prompt embeds.</param>
        /// <param name="pooledPromptEmbeds">The pooled prompt embeds.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> RunDecoderAsync(GenerateOptions options, Tensor<float> priorLatents, Tensor<float> promptEmbeds, Tensor<float> pooledPromptEmbeds, bool performGuidance, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunDecoderAsync] Begin Decoder Inference");
            using (var scheduler = CreateScheduler(options))
            {
                // Latents
                var latents = await CreateDecoderLatentsAsync(options, scheduler, priorLatents, cancellationToken);

                // EFFNet
                var effnet = priorLatents.WithGuidance(performGuidance);

                // Load Model
                var metadata = await DecoderUnet.LoadAsync(cancellationToken: cancellationToken);

                // Timesteps
                var timesteps = scheduler.GetTimesteps();
                for (int i = 0; i < timesteps.Count; i++)
                {
                    var timestep = timesteps[i];
                    var steptime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Inputs.
                    var inputLatents = scheduler.ScaleInput(timestep, latents).WithGuidance(performGuidance);

                    // Inference
                    using (var modelParameters = new ModelParameters(metadata, cancellationToken))
                    {
                        modelParameters.AddInput(inputLatents.AsTensorSpan());
                        modelParameters.AddScalarInput(timestep / 1000f);
                        modelParameters.AddInput(pooledPromptEmbeds.AsTensorSpan());
                        modelParameters.AddInput(effnet.AsTensorSpan());
                        modelParameters.AddOutput(inputLatents.Dimensions);

                        using (var results = await DecoderUnet.RunInferenceAsync(modelParameters))
                        {
                            var prediction = results[0].ToTensor();

                            // Perform guidance
                            if (performGuidance)
                                prediction = ApplyGuidance(prediction, options.GuidanceScale);

                            // Scheduler Step
                            var stepResult = scheduler.Step(timestep, prediction, latents);

                            // Result
                            latents = stepResult.Sample;

                            // Progress
                            if (scheduler.IsFinalOrder)
                                progressCallback.Notify(scheduler.CurrentStep, scheduler.TotalSteps, latents, steptime);

                            Logger.LogEnd(LogLevel.Debug, steptime, $"[RunDecoderAsync] Step: {i + 1}/{timesteps.Count}");
                        }
                    }
                }

                // Unload if required
                if (options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled)
                    await DecoderUnet.UnloadAsync();

                Logger.LogEnd(LogLevel.Debug, timestamp, "[RunDecoderAsync] Decoder Inference Complete");
                return latents;
            }
        }


        /// <summary>
        /// Create Prior latent input.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private Task<Tensor<float>> CreatePriorLatentInputAsync(IPipelineOptions options, IScheduler scheduler, CancellationToken cancellationToken = default)
        {
            var latents = scheduler.CreateRandomSample([1, 16, (int)Math.Ceiling(options.Height / _resolutionMultiple), (int)Math.Ceiling(options.Width / _resolutionMultiple)]);
            return Task.FromResult(latents.Multiply(scheduler.StartSigma));
        }


        /// <summary>
        /// Create Decoder latent input.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="priorLatents">The prior latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private Task<Tensor<float>> CreateDecoderLatentsAsync(GenerateOptions options, IScheduler scheduler, Tensor<float> priorLatents, CancellationToken cancellationToken = default)
        {
            var latents = scheduler.CreateRandomSample([1, 4, (int)(priorLatents.Dimensions[2] * _latentDimScale), (int)(priorLatents.Dimensions[3] * _latentDimScale)]);
            return Task.FromResult(latents.Multiply(scheduler.StartSigma));
        }


        /// <summary>
        /// Encode the image to model latents
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="image">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private Task<Tensor<float>> EncodeLatentsAsync(IPipelineOptions options, ImageTensor image, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(new Tensor<float>([1, 1, ImageEncoder.HiddenSize]));
        }


        /// <summary>
        /// Decode the model latents to image
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<ImageTensor> DecodeLatentsAsync(IPipelineOptions options, Tensor<float> latents, CancellationToken cancellationToken = default)
        {
            var decoderResult = await ImageDecoder.RunAsync(latents, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled)
                await ImageDecoder.UnloadAsync();

            return decoderResult.AsImageTensor();
        }


        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override async Task CheckPipelineState(IPipelineOptions options)
        {
            // Check LowMemory status
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled) && TextEncoder.IsLoaded())
                await TextEncoder.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && PriorUnet.IsLoaded())
                await PriorUnet.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && DecoderUnet.IsLoaded())
                await DecoderUnet.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled) && ImageDecoder.IsLoaded())
                await ImageDecoder.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled) && ImageEncoder.IsLoaded())
                await ImageEncoder.UnloadAsync();
        }


        /// <summary>
        /// Configures the supported schedulers.
        /// </summary>
        protected override IReadOnlyList<SchedulerType> ConfigureSchedulers()
        {
            return [SchedulerType.DDPMWuerstchen];
        }


        /// <summary>
        /// Configures the default SchedulerOptions.
        /// </summary>
        protected override GenerateOptions ConfigureDefaultOptions()
        {
            return new GenerateOptions
            {
                Steps = 20,
                Steps2 = 10,
                Width = 1024,
                Height = 1024,
                GuidanceScale = 4f,
                GuidanceScale2 = 0f,
                Scheduler = SchedulerType.DDPMWuerstchen
            };
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Tokenizer?.Dispose();
            TextEncoder?.Dispose();
            PriorUnet?.Dispose();
            DecoderUnet?.Dispose();
            ImageDecoder?.Dispose();
            ImageEncoder?.Dispose();
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// Create StableCascade pipeline from StableCascadeConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableCascadePipeline.</returns>
        public static StableCascadePipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableCascadePipeline(StableCascadeConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create StableCascade pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableCascadePipeline.</returns>
        public static StableCascadePipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableCascadePipeline(StableCascadeConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }
    }
}
