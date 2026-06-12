using Amuse.App.Common;
using Amuse.App.Services;
using Amuse.Common;
using Amuse.Common.Config;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Audio;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Video;

namespace Amuse.App.Runtime
{
    public sealed class OnnxDiffusion : ServiceBase, IDiffusionRuntime
    {
        private readonly ILogger _logger;
        private readonly Settings _settings;
        private readonly IMediaService _mediaService;
        private CancellationTokenSource _cancellationTokenSource;

        private PipelineModel _pipelineConfig;
        private PipelineClient _pipeline;
        private DiffusionDefaultOptions _defaultOptions;
        private IProgress<PipelineProgress> _progressCallback;

        public OnnxDiffusion(Settings settings, IMediaService mediaService, ILogger logger)
        {
            _logger = logger;
            _settings = settings;
            _mediaService = mediaService;
        }

        public PipelineModel Pipeline => _pipelineConfig;

        public DiffusionDefaultOptions DefaultOptions => _defaultOptions;


        public async Task LoadAsync(PipelineModel pipeline, IProgress<PipelineProgress> progressCallback)
        {
            try
            {
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    await UnloadPythonPipeline();

                    _pipelineConfig = pipeline;
                    _progressCallback = progressCallback;
                    _defaultOptions = _pipelineConfig.DiffusionModel.DefaultOptions;

                    _pipeline = await CreateClientAsync(_cancellationTokenSource.Token);
                    _pipelineConfig.DiffusionModel.Status = ModelStatusType.Installed;
                    _settings.ScanModels();
                }
            }
            catch (OperationCanceledException)
            {
                _pipeline?.Dispose();
                _pipeline = null;
                _defaultOptions = null;
                _pipelineConfig = null;
                throw;
            }
            finally
            {
                _cancellationTokenSource = null;
            }
        }


        /// <summary>
        /// Reload the pipeline
        /// </summary>
        /// <param name="pipeline">The pipeline.</param>
        public async Task ReloadAsync(PipelineModel pipeline, IProgress<PipelineProgress> progressCallback)
        {
            try
            {
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    _pipelineConfig = pipeline;
                    _progressCallback = progressCallback;
                    var reloadOptions = new PipelineReloadOptions
                    {
                        ControlNet = pipeline.ControlNetModel.GetControlNet(_settings),
                        LoraAdapters = pipeline.LoraAdapterModel.GetLoraAdapters(_settings),
                        ProcessType = pipeline.ProcessType,
                    };

                    await _pipeline.ReloadAsync(reloadOptions, _cancellationTokenSource.Token);
                    _settings.ScanModels();
                }
            }
            catch (OperationCanceledException)
            {
                _pipeline?.Dispose();
                _pipeline = null;
                _defaultOptions = null;
                _pipelineConfig = null;
                throw;
            }
            finally
            {
                _cancellationTokenSource = null;
            }
        }


        public Task UpdateAsync(PipelineModel pipeline)
        {
            _pipelineConfig = pipeline;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Cancel the running task (Load or Execute)
        /// </summary>
        public async Task CancelAsync()
        {
            try
            {
                if (_pipeline is not null)
                    await _pipeline.CancelAsync();
            }
            catch (Exception) { }
            finally
            {
                await _cancellationTokenSource.SafeCancelAsync();
            }
        }


        /// <summary>
        /// Stop/Kill server
        /// </summary>
        public async Task StopAsync()
        {
            try
            {
                await _pipeline.KillServerAsync();
            }
            catch (Exception) { }
            finally
            {
                _pipeline = null;
            }
        }


        /// <summary>
        /// Unload the pipeline
        /// </summary>
        public async Task UnloadAsync()
        {
            await CancelAsync();
            await UnloadPythonPipeline();
            _pipelineConfig = null;
            _defaultOptions = null;
        }


        /// <summary>
        /// Generate image
        /// </summary>
        /// <param name="options">The options.</param>
        /// <exception cref="Exception">Pipeline Closed Unexpectedly</exception>
        public async Task<ImageTensor> GenerateImageAsync(DiffusionInputOptions options)
        {
            try
            {
                var imageFileName = _mediaService.GetTempFile(MediaType.Image);
                options.Seed = options.Seed > 0 ? options.Seed : Random.Shared.Next();
                options.NegativePrompt = options.GuidanceScale > 1f && string.IsNullOrEmpty(options.NegativePrompt) ? " " : options.NegativePrompt;
                var generateOptions = options.ToClientOptions(_defaultOptions, imageFileName);

                var tensorResult = await _pipeline.RunAsync(generateOptions);
                return tensorResult.AsImageTensor();
            }
            catch (IOException ex)
            {
                HandleServerError(ex);
                throw new Exception("Pipeline Closed Unexpectedly");
            }
        }


        /// <summary>
        /// Generate video
        /// </summary>
        /// <param name="options">The options.</param>
        /// <exception cref="Exception">Generated video result not found.</exception>
        /// <exception cref="Exception">Pipeline Closed Unexpectedly</exception>
        public async Task<VideoInputStream> GenerateVideoAsync(DiffusionInputOptions options)
        {
            try
            {
                var videoFileName = _mediaService.GetTempFile(MediaType.Video);
                options.Seed = options.Seed > 0 ? options.Seed : Random.Shared.Next();
                options.NegativePrompt = options.GuidanceScale > 1f && string.IsNullOrEmpty(options.NegativePrompt) ? " " : options.NegativePrompt;
                var generateOptions = options.ToClientOptions(_defaultOptions, videoFileName);

                var tensorResult = await _pipeline.RunAsync(generateOptions);
                if (tensorResult is null)
                {
                    if (!File.Exists(videoFileName))
                        throw new Exception("Generated video result not found.");

                    return new VideoInputStream(videoFileName);
                }

                var videoTensor = tensorResult.AsVideoTensor(generateOptions.FrameRate);
                await videoTensor.SaveAsync(videoFileName);
                return new VideoInputStream(videoFileName);
            }
            catch (IOException ex)
            {
                HandleServerError(ex);
                throw new Exception("Pipeline Closed Unexpectedly");
            }
        }


        /// <summary>
        /// Generate audio as an asynchronous operation.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <exception cref="Exception">Generated video result not found.</exception>
        /// <exception cref="Exception">Pipeline Closed Unexpectedly</exception>
        public async Task<AudioInputStream> GenerateAudioAsync(DiffusionInputOptions options)
        {
            try
            {
                var audioFileName = _mediaService.GetTempFile(MediaType.Audio);
                options.Seed = options.Seed > 0 ? options.Seed : Random.Shared.Next();
                var generateOptions = options.ToClientOptions(_defaultOptions, audioFileName);

                foreach (var inputAudios in options.InputAudios)
                {
                    generateOptions.InputAudios.Add(await inputAudios.GetAsync(_defaultOptions.SampleRate, _defaultOptions.Channels));
                }

                var audioTensor = await _pipeline.RunAsync(generateOptions);
                var audioInput = new AudioInput(audioTensor.AsAudioTensor(_defaultOptions.SampleRate));
                await audioInput.SaveAsync(audioFileName);
                if (!File.Exists(audioFileName))
                    throw new Exception("Generated audio result not found.");

                return await AudioInputStream.CreateAsync(audioFileName);
            }
            catch (IOException ex)
            {
                HandleServerError(ex);
                throw new Exception("Pipeline Closed Unexpectedly");
            }
        }


        public async Task<TextResult> GenerateTextAsync(DiffusionInputOptions options)
        {
            try
            {
                var textResult = new TextResult();
                var textFileName = _mediaService.GetTempFile(MediaType.Text);
                options.Seed = options.Seed > 0 ? options.Seed : Random.Shared.Next();
                var generateOptions = options.ToClientOptions(_defaultOptions, textFileName);
                foreach (var inputAudios in options.InputAudios)
                {
                    generateOptions.InputAudios.Add(await inputAudios.GetAsync(_defaultOptions.SampleRate, _defaultOptions.Channels));
                }

                var pipelineResult = await _pipeline.GenerateText(generateOptions);
                foreach (var beamResult in pipelineResult)
                {
                    textResult.Results.Add(new TextInput(beamResult.Text)
                    {
                        Beam = beamResult.Beam,
                        Score = beamResult.Score,
                        PenaltyScore = beamResult.PenaltyScore
                    });
                }
                return textResult;
            }
            catch (IOException ex)
            {
                HandleServerError(ex);
                throw new Exception("Pipeline Closed Unexpectedly");
            }
        }


        public void Dispose()
        {
            _pipeline?.Dispose();
            _pipeline = null;
            _cancellationTokenSource = null;
            _pipelineConfig = null;
            _defaultOptions = null;
            _progressCallback = null;
        }


        private async Task<PipelineClient> CreateClientAsync(CancellationToken cancellationToken = default)
        {
            var clientConfig = new ClientConfig
            {
                ServerPath = App.DirectoryServer,
                IsDebugMode = _settings.IsServerDebugEnabled,
                ServerType = ServerType.OnnxRuntime,
            };
          
            var progressCallback = new Progress<PipelineProgress>(progress => _progressCallback?.Report(progress));
            var pipelineClient = new PipelineClient(clientConfig, progressCallback, _logger);

            try
            {
                var createOptions = new PipelineCreateOptions();
                var loadOptions = _pipelineConfig.ToClientOptions(_settings);
                await pipelineClient.CreateAsync(createOptions, cancellationToken);
                await pipelineClient.LoadAsync(loadOptions, cancellationToken);
                return pipelineClient;
            }
            catch (Exception)
            {
                pipelineClient?.Dispose();
                throw;
            }
        }


        private async Task UnloadPythonPipeline()
        {
            try
            {
                if (_pipeline != null)
                    await _pipeline.UnloadAsync();
            }
            catch (Exception)
            {
            }
            finally
            {
                _pipeline?.Dispose();
                _pipeline = null;
            }
        }


        private void HandleServerError(Exception exception)
        {
            try
            {
                _pipeline?.Dispose();
            }
            catch (Exception) { }
            finally
            {
                _pipeline = null;
                _pipelineConfig = null;
                _defaultOptions = null;
            }
        }
    }
}
