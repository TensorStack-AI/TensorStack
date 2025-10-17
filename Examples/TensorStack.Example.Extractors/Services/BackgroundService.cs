using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Video;
using TensorStack.Example.Common;
using TensorStack.Extractors.Common;
using TensorStack.Extractors.Pipelines;
using TensorStack.Image;
using TensorStack.Providers;
using TensorStack.Video;

namespace TensorStack.Example.Services
{
    public class BackgroundService : ServiceBase, IBackgroundService
    {
        private readonly Settings _settings;
        private readonly IMediaService _mediaService;
        private BackgroundPipeline _currentPipeline;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isLoaded;
        private bool _isLoading;
        private bool _isExecuting;
        private ExtractorConfig _currentConfig;

        /// <summary>
        /// Initializes a new instance of the <see cref="BackgroundService"/> class.
        /// </summary>
        /// <param name="settings">The settings.</param>
        public BackgroundService(Settings settings, IMediaService mediaService)
        {
            _settings = settings;
            _mediaService = mediaService;
        }

        /// <summary>
        /// Gets a value indicating whether this instance is loaded.
        /// </summary>
        public bool IsLoaded
        {
            get { return _isLoaded; }
            private set { SetProperty(ref _isLoaded, value); }
        }

        /// <summary>
        /// Gets a value indicating whether this instance is loading.
        /// </summary>
        public bool IsLoading
        {
            get { return _isLoading; }
            private set { SetProperty(ref _isLoading, value); NotifyPropertyChanged(nameof(CanCancel)); }
        }

        /// <summary>
        /// Gets a value indicating whether this instance is executing.
        /// </summary>
        public bool IsExecuting
        {
            get { return _isExecuting; }
            private set { SetProperty(ref _isExecuting, value); NotifyPropertyChanged(nameof(CanCancel)); }
        }

        /// <summary>
        /// Gets a value indicating whether this instance can cancel.
        /// </summary>
        public bool CanCancel => _isLoading || _isExecuting;


        /// <summary>
        /// Load the pipeline
        /// </summary>
        /// <param name="config">The configuration.</param>
        public async Task LoadAsync(BackgroundModel model, Device device)
        {
            try
            {
                IsLoaded = false;
                IsLoading = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var cancellationToken = _cancellationTokenSource.Token;
                    if (_currentPipeline != null)
                    {
                        if (_currentConfig.Path == model.Path)
                            return; // Already loaded

                        await _currentPipeline.UnloadAsync(cancellationToken);
                    }

                    _currentConfig = new ExtractorConfig
                    {
                        Path = model.Path,
                        Channels = model.Channels,
                        Normalization = model.Normalization,
                        OutputChannels = model.OutputChannels,
                        OutputNormalization = model.OutputNormalization,
                        SampleSize = model.SampleSize
                    };

                    _currentConfig.SetProvider(device.GetProvider());
                    _currentPipeline = BackgroundPipeline.Create(_currentConfig);
                    await Task.Run(() => _currentPipeline.LoadAsync(cancellationToken), cancellationToken);
                }
            }
            catch (OperationCanceledException)
            {
                _currentPipeline?.Dispose();
                _currentPipeline = null;
                _currentConfig = null;
                throw;
            }
            finally
            {
                IsLoaded = true;
                IsLoading = false;
            }
        }


        /// <summary>
        /// Execute the image pipeline
        /// </summary>
        /// <param name="request">The request.</param>
        public async Task<ImageInput> ExecuteAsync(BackgroundImageRequest options)
        {
            try
            {
                IsExecuting = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var imageTensor = await Task.Run(() => _currentPipeline.RunAsync(new BackgroundImageOptions
                    {
                        Mode = options.Mode,
                        Input = options.Image
                    }, cancellationToken: _cancellationTokenSource.Token));

                    if (options.IsTransparentSupported)
                        return new ImageInput(imageTensor.ToImageTransparent());

                    return new ImageInput(imageTensor);
                }
            }
            finally
            {
                IsExecuting = false;
            }
        }


        /// <summary>
        /// Execute the video pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        public async Task<VideoInputStream> ExecuteAsync(BackgroundVideoRequest options, IProgress<RunProgress> progressCallback)
        {
            try
            {
                IsExecuting = true;
                var resultVideoFile = FileHelper.RandomFileName(_settings.DirectoryTemp, "mp4");
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var frameCount = options.VideoStream.FrameCount;
                    var cancellationToken = _cancellationTokenSource.Token;

                    async Task<VideoFrame> FrameProcessor(VideoFrame frame)
                    {
                        var processedFrame = await _currentPipeline.RunAsync(new BackgroundImageOptions
                        {
                            Input = frame.Frame,
                            Mode = options.Mode
                        }, cancellationToken: cancellationToken);

                        progressCallback.Report(new RunProgress(frame.Index, frameCount));
                        return new VideoFrame(frame.Index, processedFrame, frame.SourceFrameRate, frame.AuxFrame);
                    }

                    return await _mediaService.SaveWithAudioAsync(options.VideoStream, resultVideoFile, FrameProcessor, cancellationToken);
                }
            }
            finally
            {
                IsExecuting = false;
            }
        }


        /// <summary>
        /// Cancel the running task (Load or Execute)
        /// </summary>
        public async Task CancelAsync()
        {
            await _cancellationTokenSource.SafeCancelAsync();
        }


        /// <summary>
        /// Unload the pipeline
        /// </summary>
        public async Task UnloadAsync()
        {
            if (_currentPipeline != null)
            {
                await _cancellationTokenSource.SafeCancelAsync();
                await _currentPipeline.UnloadAsync();
                _currentPipeline.Dispose();
                _currentPipeline = null;
                _currentConfig = null;
            }

            IsLoaded = false;
            IsLoaded = false;
            IsExecuting = false;
        }
    }


    public interface IBackgroundService
    {
        bool IsLoaded { get; }
        bool IsLoading { get; }
        bool IsExecuting { get; }
        bool CanCancel { get; }
        Task LoadAsync(BackgroundModel model, Device device);
        Task UnloadAsync();
        Task CancelAsync();
        Task<ImageInput> ExecuteAsync(BackgroundImageRequest options);
        Task<VideoInputStream> ExecuteAsync(BackgroundVideoRequest options, IProgress<RunProgress> progressCallback);
    }


    public record BackgroundImageRequest
    {
        public BackgroundMode Mode { get; init; }
        public ImageInput Image { get; set; }
        public bool IsTransparentSupported { get; set; }
    }


    public record BackgroundVideoRequest
    {
        public BackgroundMode Mode { get; init; }
        public VideoInputStream VideoStream { get; set; }
    }
}
