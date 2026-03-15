// Copyright (c) 2026 Joe Dluzen. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Schedulers;
using TensorStack.TextGeneration.Pipelines.Llama;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.Nitro
{
    public abstract class NitroARBase : PipelineBase
    {
        private const float MASK_TOKEN_VALUE = 0.0f;
        private const int LATENT_RESOLUTION = 16;
        private const int TOTAL_TOKENS = LATENT_RESOLUTION * LATENT_RESOLUTION; // 256

        public NitroARBase(TransformerNitroARModel transformer, LlamaPipeline textEncoder, AutoEncoderModel autoEncoder, int outputSize = 512, ILogger logger = default) : base(logger)
        {
            Transformer = transformer;
            AutoEncoder = autoEncoder;
            TextEncoder = textEncoder;
            OutputSize = outputSize;
            Initialize();
            Logger?.LogInformation("[NitroARPipeline] Name: {Name}", Name);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NitroBase"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public NitroARBase(NitroConfig configuration, ILogger logger = default) : this(
            new TransformerNitroARModel(configuration.Transformer),
            new LlamaPipeline(new LlamaConfig
            {
                OutputLastHiddenStates = true,
                DecoderConfig = configuration.TextEncoder,
                Tokenizer = new BPETokenizer(configuration.Tokenizer),
            }),
            new AutoEncoderModel(configuration.AutoEncoder),
            configuration.OutputSize,
            logger)
        {
            Name = configuration.Name;
        }

        public override PipelineType PipelineType => PipelineType.Nitro;
        public override string Name { get; init; } = "Nitro-AR";
        public LlamaPipeline TextEncoder { get; init; }
        public TransformerNitroARModel Transformer { get; init; }
        public AutoEncoderModel AutoEncoder { get; init; }
        public int OutputSize { get; }

        public Task LoadAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

        public async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await Task.WhenAll
            (
                Transformer.UnloadAsync(),
                TextEncoder.UnloadAsync(cancellationToken),
                AutoEncoder.EncoderUnloadAsync(),
                AutoEncoder.DecoderUnloadAsync()
            );
        }

        protected override void ValidateOptions(GenerateOptions options)
        {
            base.ValidateOptions(options);
            if (options.Width != 512 || options.Height != 512)
                throw new ArgumentException($"Nitro-AR mathematically requires a 512x512 output size (16x16 latents). Requested: {options.Width}x{options.Height}");
        }

        protected async Task<PromptResult> CreatePromptAsync(IPipelineOptions options, CancellationToken cancellationToken = default)
        {
            var cachedPrompt = GetPromptCache(options);
            if (cachedPrompt is not null) return cachedPrompt;

            var promptEmbeds = await TextEncoder.GetLastHiddenState(new TextGeneration.Common.GenerateOptions
            {
                Seed = options.Seed,
                Prompt = options.Prompt,
                MinLength = 128,
                MaxLength = 128
            }, cancellationToken);

            var negativePromptEmbeds = default(Tensor<float>);
            if (options.GuidanceScale > 1.0f)
            {
                negativePromptEmbeds = await TextEncoder.GetLastHiddenState(new TextGeneration.Common.GenerateOptions
                {
                    Seed = options.Seed,
                    Prompt = options.NegativePrompt,
                    MinLength = 128,
                    MaxLength = 128
                }, cancellationToken);
            }

            return SetPromptCache(options, new PromptResult(promptEmbeds, default, negativePromptEmbeds, default));
        }

        protected async Task<Tensor<float>> RunInferenceAsync(IPipelineOptions options, IScheduler scheduler, PromptResult prompt, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunInferenceAsync] Begin Nitro-AR Inference");

            var isGuidanceEnabled = options.GuidanceScale > 1.0f;
            var promptEmbedsCond = prompt.PromptEmbeds;
            var promptEmbedsUncond = prompt.NegativePromptEmbeds;

            // Initialize 16x16 Latent Canvas with Mask Tokens (0.0f)
            var latents = CreateMaskedLatents();

            // Generate the Random Unmasking Order (Indices 0 to 255 shuffled)
            var random = new Random(options.Seed);
            var tokenOrder = Enumerable.Range(0, TOTAL_TOKENS).OrderBy(x => random.Next()).ToArray();

            var isMasked = new bool[TOTAL_TOKENS];
            Array.Fill(isMasked, true);

            await LoadTransformerAsync(options, progressCallback, cancellationToken);

            // Default to 3 steps for GAN, 6 for Joint GAN if not specified
            int totalArSteps = options is GenerateOptions genOptions ? genOptions.Steps : 3;

            for (int step = 0; step < totalArSteps; step++)
            {
                var steptime = Stopwatch.GetTimestamp();
                cancellationToken.ThrowIfCancellationRequested();

                // Cosine Masking Schedule
                float maskRatio = (float)Math.Cos(Math.PI / 2.0 * (step + 1) / totalArSteps);
                int targetMaskCount = (int)Math.Floor(TOTAL_TOKENS * maskRatio);
                if (step == totalArSteps - 1) targetMaskCount = 0;

                // Run ONNX Transformer
                var predictedLatents = await Transformer.RunAsync(latents, promptEmbedsCond, cancellationToken: cancellationToken);

                // AMD's specific Linear CFG scaling for Autoregressive models
                if (isGuidanceEnabled)
                {
                    var unconditional = await Transformer.RunAsync(latents, promptEmbedsUncond, cancellationToken: cancellationToken);

                    // CFG scales linearly from 1.0 to options.GuidanceScale as the image resolves
                    float currentMaskLen = (float)Math.Floor(TOTAL_TOKENS * (float)Math.Cos(Math.PI / 2.0 * step / totalArSteps));
                    float cfgIter = 1.0f + (options.GuidanceScale - 1.0f) * (TOTAL_TOKENS - currentMaskLen) / TOTAL_TOKENS;

                    predictedLatents = ApplyGuidance(predictedLatents, unconditional, cfgIter);
                }

                // Unmask the confident tokens
                CommitPredictedTokens(latents, predictedLatents, isMasked, tokenOrder, targetMaskCount);

                progressCallback?.Notify(step + 1, totalArSteps, latents, steptime);
                Logger.LogEnd(LogLevel.Debug, steptime, $"[RunInferenceAsync] AR Step: {step + 1}/{totalArSteps} | Masked Remaining: {targetMaskCount}");
            }

            if (options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled)
                await Transformer.UnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[RunInferenceAsync] AR Inference Complete");
            return latents;
        }

        private Tensor<float> CreateMaskedLatents()
        {
            var dimensions = new int[] { 1, AutoEncoder.LatentChannels, LATENT_RESOLUTION, LATENT_RESOLUTION };
            var maskedLatentTensor = new Tensor<float>(dimensions);
            maskedLatentTensor.Fill(MASK_TOKEN_VALUE);
            return maskedLatentTensor;
        }

        private void CommitPredictedTokens(Tensor<float> currentLatents, Tensor<float> predictedLatents, bool[] isMasked, int[] tokenOrder, int targetMaskCount)
        {
            int channels = currentLatents.Dimensions[1];

            // Only unmask the tokens transitioning from MASKED to UNMASKED
            for (int i = targetMaskCount; i < TOTAL_TOKENS; i++)
            {
                int tokenIndex = tokenOrder[i];
                if (isMasked[tokenIndex])
                {
                    int h = tokenIndex / LATENT_RESOLUTION;
                    int w = tokenIndex % LATENT_RESOLUTION;

                    for (int c = 0; c < channels; c++)
                    {
                        currentLatents[0, c, h, w] = predictedLatents[0, c, h, w];
                    }

                    isMasked[tokenIndex] = false;
                }
            }
        }

        protected async Task<ImageTensor> DecodeLatentsAsync(IPipelineOptions options, Tensor<float> latents, CancellationToken cancellationToken = default)
        {
            // Unscale the latents back to the AutoEncoder's RGB target range before decoding. Was supposed to fix blurry images.
            float unscaleFactor = 1.0f / AutoEncoder.ScaleFactor; // (e.g., 1.0f / 0.41407f)
            var scaledLatents = new Tensor<float>(latents.Dimensions);

            for (int i = 0; i < latents.Length; i++)
            {
                scaledLatents.SetValue(i, latents.GetValue(i) * unscaleFactor);
            }

            var decoderResult = await AutoEncoder.DecodeAsync(latents, cancellationToken: cancellationToken);
            //var decoderResult = await AutoEncoder.DecodeAsync(scaledLatents, cancellationToken: cancellationToken);

            if (options.IsLowMemoryEnabled) await AutoEncoder.DecoderUnloadAsync();
            return decoderResult.AsImageTensor();
        }

        private Tensor<float> ApplyGuidance(Tensor<float> cond, Tensor<float> uncond, float scale)
        {
            var result = new Tensor<float>(cond.Dimensions);
            for (int i = 0; i < cond.Length; i++)
            {
                result.SetValue(i, uncond.GetValue(i) + scale * (cond.GetValue(i) - uncond.GetValue(i)));
            }
            return result;
        }

        private ModelOptimization GetOptimizations(IPipelineOptions generateOptions, IProgress<GenerateProgress> progressCallback = null)
        {
            var optimizations = new ModelOptimization(Optimization.None);
            if (Transformer.HasOptimizationsChanged(optimizations)) progressCallback?.Notify("Optimizing Pipeline...");
            return optimizations;
        }

        private async Task<ModelMetadata> LoadTransformerAsync(IPipelineOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var optimizations = GetOptimizations(options, progressCallback);
            return await Transformer.LoadAsync(optimizations, cancellationToken);
        }

        protected override async Task CheckPipelineState(IPipelineOptions options)
        {
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Transformer.IsLoaded())
                await Transformer.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled) && AutoEncoder.IsEncoderLoaded())
                await AutoEncoder.EncoderUnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled) && AutoEncoder.IsDecoderLoaded())
                await AutoEncoder.DecoderUnloadAsync();
        }

        protected override IReadOnlyList<SchedulerType> ConfigureSchedulers() => [SchedulerType.None];

        protected override GenerateOptions ConfigureDefaultOptions()
        {
            return new GenerateOptions
            {
                Steps = 3, // Base GAN default
                Width = 512,
                Height = 512,
                GuidanceScale = 1.0f,
                Scheduler = SchedulerType.None
            };
        }

        private bool _disposed;
        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;
            if (disposing)
            {
                TextEncoder?.Dispose();
                Transformer?.Dispose();
                AutoEncoder?.Dispose();
            }
            _disposed = true;
        }
    }
}