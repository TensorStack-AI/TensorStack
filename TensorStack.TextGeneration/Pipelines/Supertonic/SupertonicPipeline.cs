using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Pipelines.Supertonic
{
    /// <summary>
    /// Supertonic TTS Pipeline.
    /// </summary>
    public class SupertonicPipeline : IPipeline<AudioTensor, SupertonicOptions, GenerateProgress>
    {
        private readonly Random _random;

        /// <summary>
        /// Initializes a new instance of the <see cref="SupertonicPipeline"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        public SupertonicPipeline(SupertonicConfig configuration)
        {
            _random = new Random();
            Configuration = configuration;
            Processor = new SupertonicProcessor(configuration.IndexerPath, configuration.VoiceStylePath);
            Prediction = new ModelSession(configuration.PredictorConfig);
            Encoder = new ModelSession(configuration.EncoderConfig);
            Estimator = new ModelSession(configuration.EstimatorConfig);
            Decoder = new ModelSession(configuration.DecoderConfig);
        }

        public SupertonicConfig Configuration { get; init; }
        public SupertonicProcessor Processor { get; init; }
        public ModelSession Prediction { get; init; }
        public ModelSession Encoder { get; init; }
        public ModelSession Estimator { get; init; }
        public ModelSession Decoder { get; init; }
        public IEnumerable<string> VoiceStyles => Processor.VoiceStyles;


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            // SupertonicPipeline pipelines are lazy loaded on first run
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
                Prediction.UnloadAsync(),
                Encoder.UnloadAsync(),
                Estimator.UnloadAsync(),
                Decoder.UnloadAsync()
            );
        }


        /// <summary>
        /// Run as an asynchronous operation.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<AudioTensor> RunAsync(SupertonicOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var totalDuration = 0.0f;
            var audioBuffer = new List<float>();
            var voiceStyle = Processor.GetVoiceStyle(options.VoiceStyle);
            var silenceLength = (int)(options.SilenceDuration * Configuration.SampleRate);
            var silenceBuffer = new float[silenceLength];

            // Process text
            foreach (var textIds in Processor.GetTextIds(options.TextInput))
            {
                var result = await RunInferenceAsync(textIds, voiceStyle, options.Steps, options.Speed);
                if (audioBuffer.Count == 0)
                {
                    audioBuffer.AddRange(result.Audio.Memory.Span);
                    totalDuration = result.Duration;
                }
                else
                {
                    audioBuffer.AddRange(silenceBuffer);
                    audioBuffer.AddRange(result.Audio.Memory.Span);
                    totalDuration += result.Duration + options.SilenceDuration;
                }
            }

            var audioSpan = CollectionsMarshal.AsSpan(audioBuffer);
            var audioLength = (int)(Configuration.SampleRate * totalDuration);
            var audioTensor = new Tensor<float>([1, audioLength]);
            audioSpan[..Math.Min(audioLength, audioSpan.Length)].CopyTo(audioTensor.Memory.Span);
            return audioTensor.AsAudioTensor(Configuration.SampleRate);
        }


        /// <summary>
        /// Run inference as an asynchronous operation.
        /// </summary>
        /// <param name="textIds">The text ids.</param>
        /// <param name="style">The style.</param>
        /// <param name="totalStep">The total step.</param>
        /// <param name="speed">The speed.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<InferenceResult> RunInferenceAsync(Tensor<long> textIds, VoiceStyle style, int totalStep, float speed = 1.05f, CancellationToken cancellationToken = default)
        {
            var predictionResult = await PredictAsync(textIds, style.Dropout, cancellationToken);
            var duration = predictionResult.Memory.Span[0] / speed;
            var encoderResult = await EncodeAsync(textIds, style.Global, cancellationToken);
            var latents = PrepareLatents(duration);
            for (int step = 0; step < totalStep; step++)
            {
                latents = await EstimateAsync(latents, encoderResult, style.Global, step, totalStep, cancellationToken);
            }
            var decoderResult = await DecodeAsync(latents, cancellationToken);
            return new InferenceResult(decoderResult, duration);
        }


        /// <summary>
        /// Run duration prediction model
        /// </summary>
        /// <param name="textIds">The text ids.</param>
        /// <param name="styleDropout">The style dropout.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> PredictAsync(Tensor<long> textIds, Tensor<float> styleDropout, CancellationToken cancellationToken = default)
        {
            var metadata = await Prediction.LoadAsync();
            var textMask = new Tensor<float>([1, 1, textIds.Dimensions[1]], 1f);
            using (var parameters = new ModelParameters(metadata, cancellationToken))
            {
                parameters.AddInput(textIds);
                parameters.AddInput(styleDropout);
                parameters.AddInput(textMask);
                parameters.AddOutput([1]);
                using (var result = await Prediction.RunInferenceAsync(parameters))
                {
                    return result[0].ToTensor();
                }
            }
        }

        /// <summary>
        /// Run text encoder model
        /// </summary>
        /// <param name="textIds">The text ids.</param>
        /// <param name="styleGlobal">The style global.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> EncodeAsync(Tensor<long> textIds, Tensor<float> styleGlobal, CancellationToken cancellationToken = default)
        {
            var metadata = await Encoder.LoadAsync();
            var textMask = new Tensor<float>([1, 1, textIds.Dimensions[1]], 1f);
            using (var parameters = new ModelParameters(metadata, cancellationToken))
            {
                parameters.AddInput(textIds);
                parameters.AddInput(styleGlobal);
                parameters.AddInput(textMask);
                parameters.AddOutput([1, Configuration.TextEmbedSize, textIds.Dimensions[1]]);
                using (var result = await Encoder.RunInferenceAsync(parameters))
                {
                    return result[0].ToTensor();
                }
            }
        }

        /// <summary>
        /// Run vector estimate model
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="textEmbeds">The text embeds.</param>
        /// <param name="styleGlobal">The style global.</param>
        /// <param name="step">The step.</param>
        /// <param name="steps">The steps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> EstimateAsync(Tensor<float> latents, Tensor<float> textEmbeds, Tensor<float> styleGlobal, int step, int steps, CancellationToken cancellationToken = default)
        {
            var metadata = await Estimator.LoadAsync();
            var textMask = new Tensor<float>([1, 1, textEmbeds.Dimensions[2]], 1f);
            var latentMask = new Tensor<float>([1, 1, latents.Dimensions[2]], 1f);
            using (var parameters = new ModelParameters(metadata, cancellationToken))
            {
                parameters.AddInput(latents);
                parameters.AddInput(textEmbeds);
                parameters.AddInput(styleGlobal);
                parameters.AddInput(latentMask);
                parameters.AddInput(textMask);
                parameters.AddScalarInput(step);
                parameters.AddScalarInput(steps);
                parameters.AddOutput(latents.Dimensions);
                using (var vectorEstResult = await Estimator.RunInferenceAsync(parameters))
                {
                    return vectorEstResult[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Run decoder model
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> DecodeAsync(Tensor<float> latents, CancellationToken cancellationToken = default)
        {
            var metadata = await Decoder.LoadAsync();
            var bufferSize = Configuration.ScaleFactor * latents.Dimensions[2];
            using (var parameters = new ModelParameters(metadata, cancellationToken))
            {
                parameters.AddInput(latents);
                parameters.AddOutput([1, bufferSize]);
                using (var result = await Decoder.RunInferenceAsync(parameters))
                {
                    return result[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Prepares the latents.
        /// </summary>
        /// <param name="duration">The duration.</param>
        private Tensor<float> PrepareLatents(float duration)
        {
            var audioLength = duration * Configuration.SampleRate;
            var chunkSize = Configuration.BaseChunkSize * Configuration.ChunkCompressFactor;
            var latentLen = (int)((audioLength + chunkSize - 1) / chunkSize);
            var latentDim = Configuration.LatentDim * Configuration.ChunkCompressFactor;
            var latents = _random.NextTensor([1, latentDim, latentLen]);
            return latents;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Prediction.Dispose();
            Encoder.Dispose();
            Estimator.Dispose();
            Decoder.Dispose();
        }


        /// <summary>
        /// Creates the SupertonicPipeline
        /// </summary>
        /// <param name="modelPath">The model path.</param>
        /// <param name="provider">The provider.</param>
        /// <returns>SupertonicPipeline.</returns>
        public static SupertonicPipeline Create(string modelPath, ExecutionProvider provider)
        {
            var config = new SupertonicConfig
            {
                LatentDim = 24,
                SampleRate = 44100,
                ScaleFactor = 3072,
                BaseChunkSize = 512,
                TextEmbedSize = 256,
                ChunkCompressFactor = 6,
                VoiceStylePath = Path.Combine(modelPath, "voice_styles"),
                IndexerPath = Path.Combine(modelPath, "unicode_indexer.json"),
                PredictorConfig = new ModelConfig
                {
                    ExecutionProvider = provider,
                    Path = Path.Combine(modelPath, "duration_predictor.onnx")
                },
                EncoderConfig = new ModelConfig
                {
                    ExecutionProvider = provider,
                    Path = Path.Combine(modelPath, "text_encoder.onnx")
                },
                EstimatorConfig = new ModelConfig
                {
                    ExecutionProvider = provider,
                    Path = Path.Combine(modelPath, "vector_estimator.onnx")
                },
                DecoderConfig = new ModelConfig
                {
                    ExecutionProvider = provider,
                    Path = Path.Combine(modelPath, "vocoder.onnx"),
                }
            };
            return new SupertonicPipeline(config);
        }

        private record InferenceResult(Tensor<float> Audio, float Duration);
    }
}
