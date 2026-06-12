using Amuse.Common;
using Amuse.Common.Config;
using Amuse.Common.Message;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Providers;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Pipelines.Flux;
using TensorStack.StableDiffusion.Pipelines.LatentConsistency;
using TensorStack.StableDiffusion.Pipelines.StableCascade;
using TensorStack.StableDiffusion.Pipelines.StableDiffusion;
using TensorStack.StableDiffusion.Pipelines.StableDiffusion3;
using TensorStack.StableDiffusion.Pipelines.StableDiffusionXL;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Pipelines.Supertonic;
using TensorStack.TextGeneration.Pipelines.Whisper;
using GenerateOptions = TensorStack.StableDiffusion.Common.GenerateOptions;
using GenerateProgress = TensorStack.StableDiffusion.Common.GenerateProgress;
using GenerateTextProgress = TensorStack.TextGeneration.Common.GenerateProgress;
using GenerateTextResult = TensorStack.TextGeneration.Common.GenerateResult;

namespace Amuse.Host.Onnx
{
    public sealed class HostServer : PipelineServer
    {
        private readonly IProgress<RunProgress> _progressRelayRunCallback;
        private readonly IProgress<GenerateProgress> _progressRelayGenerateCallback;
        private readonly IProgress<GenerateTextProgress> _progressRelayGenerateTextCallback;

        private IPipeline _pipeline;
        private PipelineLoadOptions _pipelineOptions;
        private ExecutionProvider _executionProvider;
        private ExecutionProvider _executionProviderCPU;

        public HostServer(ServerConfig channelConfig, ILogger logger)
            : base(channelConfig, logger)
        {
            _progressRelayRunCallback = new Progress<RunProgress>(async (p) => await UpdateProgress(p));
            _progressRelayGenerateCallback = new Progress<GenerateProgress>(async (p) => await UpdateProgress(p));
            _progressRelayGenerateTextCallback = new Progress<GenerateTextProgress>(async (p) => await UpdateProgress(p));
        }


        /// <summary>
        /// Called when the Channel is opened.
        /// </summary>
        /// <returns>Task.</returns>
        protected override Task ChannelOpenedAsync()
        {
            return Task.CompletedTask;
        }


        /// <summary>
        /// Called when the Channel is closed.
        /// </summary>
        protected override Task ChannelClosedAsync()
        {
            _pipeline?.Dispose();
            return Task.CompletedTask;
        }


        protected override async Task CreatePipelineAsync(PipelineRequest request, CancellationToken cancellationToken)
        {
            try
            {
                var timestamp = Stopwatch.GetTimestamp();
                var environmentRequest = request.CreateOptions;

                Logger.LogInformation($"[PipelineServer] [CreatePipeline] Environment created, Elapsed: {Stopwatch.GetElapsedTime(timestamp)}");
                await SendResponse(cancellationToken);
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[PipelineServer] [CreatePipeline] An exception occurred creating environment.");
                await SendException(ex, cancellationToken);
            }
        }


        protected override async Task LoadPipelineAsync(PipelineRequest request, CancellationToken cancellationToken)
        {
            try
            {
                _pipelineOptions = request.LoadOptions;
                _executionProvider = Provider.GetProvider(DeviceType.GPU, _pipelineOptions.DeviceId, GraphOptimizationLevel.ORT_ENABLE_ALL);
                _executionProviderCPU = Provider.GetProvider(DeviceType.CPU, GraphOptimizationLevel.ORT_ENABLE_ALL); // TODO: DirectML not working with decoder

                Enum.TryParse<ModelType>(_pipelineOptions.ModelType, true, out var modelType);
                Enum.TryParse<WhisperType>(_pipelineOptions.ModelType, true, out var WhisperType);

                var onnxModelPath = _pipelineOptions.CheckpointConfig.Compute;

                _pipeline = _pipelineOptions.Pipeline switch
                {
                    "FluxPipeline" => FluxPipeline.FromFolder(onnxModelPath, modelType, _executionProvider, Logger),
                    "LatentConsistencyPipeline" => LatentConsistencyPipeline.FromFolder(onnxModelPath, modelType, _executionProvider, Logger),
                    "StableCascadePipeline" => StableCascadePipeline.FromFolder(onnxModelPath, modelType, _executionProvider, Logger),
                    "StableDiffusionPipeline" => StableDiffusionPipeline.FromFolder(onnxModelPath, modelType, _executionProvider, Logger),
                    "StableDiffusion3Pipeline" => StableDiffusion3Pipeline.FromFolder(onnxModelPath, modelType, _executionProvider, Logger),
                    "StableDiffusionXLPipeline" => StableDiffusionXLPipeline.FromFolder(onnxModelPath, modelType, _executionProvider, Logger),

                    "SupertonicPipeline" => SupertonicPipeline.Create(onnxModelPath, _executionProvider),
                    "WhisperPipeline" => WhisperPipeline.Create(_executionProvider, _executionProviderCPU, onnxModelPath, WhisperType),
                    _ => throw new NotImplementedException()
                };
                await _pipeline.LoadAsync(cancellationToken);
                await SendResponse(cancellationToken);

            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[PipelineServer] [LoadPipeline] An exception occurred loading pipeline.");
                await SendException(ex, cancellationToken);
            }
        }


        protected override async Task ReloadPipelineAsync(PipelineRequest request, CancellationToken cancellationToken)
        {
            try
            {
                var reloadOptions = request.ReloadOptions;
                _pipelineOptions.ProcessType = reloadOptions.ProcessType;
                _pipelineOptions.ControlNet = reloadOptions.ControlNet;
                _pipelineOptions.LoraAdapters = reloadOptions.LoraAdapters;

                // TODO: Reload?

                await SendResponse(cancellationToken);
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[PipelineServer] [ReloadPipeline] An exception occurred reloading pipeline.");
                await SendException(ex, cancellationToken);
            }
        }


        protected override async Task UnloadPipelineAsync(PipelineRequest request, CancellationToken cancellationToken)
        {
            try
            {
                await _pipeline.UnloadAsync();
                await SendResponse(cancellationToken);
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[PipelineServer] [UnloadPipeline] An exception occurred unloading pipeline.");
                await SendException(ex, cancellationToken);
            }
        }


        protected override async Task RunPipelineAsync(PipelineRequest request, CancellationToken cancellationToken)
        {
            try
            {
                request.UnpackTensors();
                if (_pipelineOptions.ProcessType == Common.ProcessType.AudioToText)
                {
                    var resultTensor = await GenerateTextAsync(request.RunOptions, cancellationToken);
                    await SendMessage(new PipelineResponse(resultTensor), cancellationToken);
                }
                else if (_pipelineOptions.ProcessType == Common.ProcessType.TextToAudio)
                {
                    var resultTensor = await GenerateAudioAsync(request.RunOptions, cancellationToken);
                    await SendMessage(new PipelineResponse(resultTensor), cancellationToken);
                }
                else
                {
                    var resultTensor = await GenerateImageAsync(request.RunOptions, cancellationToken);
                    await SendMessage(new PipelineResponse(resultTensor), cancellationToken);
                }
            }
            catch (OperationCanceledException ex)
            {
                Logger.LogError("[PipelineServer] [RunPipeline] {Message}", ex.Message);
                await SendException(ex, cancellationToken);
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, "[PipelineServer] [RunPipeline] An exception occurred running pipeline.");
                await SendException(ex, cancellationToken);
            }
        }


        private async Task<ImageTensor> GenerateImageAsync(PipelineRunOptions options, CancellationToken cancellationToken)
        {
            var onnxOptions = options.ToOnnxOptions(_pipelineOptions, _executionProvider);
            var diffusionPipeline = _pipeline as IPipeline<ImageTensor, GenerateOptions, GenerateProgress>;
            return await diffusionPipeline.RunAsync(onnxOptions, _progressRelayGenerateCallback, cancellationToken);
        }


        private async Task<AudioTensor> GenerateAudioAsync(PipelineRunOptions options, CancellationToken cancellationToken)
        {
            var supertonicPipeline = _pipeline as IPipeline<AudioTensor, SupertonicOptions, RunProgress>;
            var pipelineOptions = new SupertonicOptions
            {
                TextInput = options.Prompt,
                VoiceStyle = options.Task,
                Steps = options.Steps,
                Speed = options.Speed,
                SilenceDuration = options.SilenceDuration,
                Seed = options.Seed,
            };
            return await supertonicPipeline.RunAsync(pipelineOptions, _progressRelayRunCallback, cancellationToken);
        }


        public async Task<PipelineTextResult[]> GenerateTextAsync(PipelineRunOptions options, CancellationToken cancellationToken)
        {
            var pipelineOptions = new WhisperOptions
            {
                Seed = options.Seed,
                Beams = options.Beams,
                TopK = options.TopK,
                TopP = options.TopP,
                Temperature = options.Temperature,
                MaxLength = options.MaxLength,
                MinLength = options.MinLength,
                NoRepeatNgramSize = options.NoRepeatNgramSize,
                LengthPenalty = options.LengthPenalty,
                DiversityLength = options.DiversityLength,
                EarlyStopping = Enum.Parse<EarlyStopping>(options.EarlyStopping, true),
                Language = options.GetLanguageType(),
                Task = Enum.Parse<TaskType>(options.Task),
                ChunkSize = options.ChunkSize,
                AudioInput = options.InputAudios[0]
            };

            var pipelineResult = await Task.Run(async () =>
            {
                if (options.Beams == 0)
                {
                    // Greedy Search
                    var greedyPipeline = _pipeline as IPipeline<GenerateTextResult, WhisperOptions, GenerateTextProgress>;
                    return [await greedyPipeline.RunAsync(pipelineOptions, _progressRelayGenerateTextCallback, cancellationToken)];
                }

                // Beam Search
                var beamSearchPipeline = _pipeline as IPipeline<GenerateTextResult[], WhisperSearchOptions, GenerateTextProgress>;
                return await beamSearchPipeline.RunAsync(new WhisperSearchOptions(pipelineOptions), _progressRelayGenerateTextCallback, cancellationToken);
            });

            var results = new PipelineTextResult[pipelineResult.Length];
            for (int i = 0; i < pipelineResult.Length; i++)
            {
                var beamResult = pipelineResult[i];
                results[i] = new PipelineTextResult
                {
                    Beam = beamResult.Beam,
                    PenaltyScore = beamResult.PenaltyScore,
                    Score = beamResult.Score,
                    Text = beamResult.Result
                };
            }
            return results;
        }


        private async Task UpdateProgress(GenerateProgress progress)
        {
            var subkey = progress.Type == GenerateProgress.ProgressType.Step ? "Step" : null;
            await QueueProgress(new PipelineProgress
            {
                Key = "Generate",
                Subkey = subkey,
                Value = progress.Value,
                Maximum = progress.Max,
                Message = progress.Message,
                Elapsed = (float)progress.Elapsed.TotalMilliseconds
            });
        }


        private async Task UpdateProgress(GenerateTextProgress progress)
        {
            await QueueProgress(new PipelineProgress
            {
                Key = "Generate",
                Subkey = $"{progress.IsReset}",
                Value = progress.Value,
                Maximum = progress.Maximum,
                Message = progress.Result
            });
        }

        private async Task UpdateProgress(RunProgress progress)
        {
            await QueueProgress(new PipelineProgress
            {
                Key = "Generate",
                Subkey = "Step",
                Value = progress.Value,
                Maximum = progress.Maximum,
                Message = progress.Message,
                Elapsed = (float)progress.Elapsed.TotalMilliseconds
            });
        }

    }
}