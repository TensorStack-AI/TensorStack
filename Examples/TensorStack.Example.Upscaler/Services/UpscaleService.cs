using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Video;
using TensorStack.Example.Common;
using TensorStack.Image;
using TensorStack.Providers;
using TensorStack.Upscaler.Common;
using TensorStack.Upscaler.Pipelines;
using TensorStack.Video;

namespace TensorStack.Example.Services
{
    public class UpscaleService : ServiceBase, IUpscaleService
    {
        private readonly Settings _settings;
        private readonly IMediaService _mediaService;
        private UpscalePipeline _currentPipeline;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isLoaded;
        private bool _isLoading;
        private bool _isExecuting;
        private UpscalerConfig _currentConfig;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleService"/> class.
        /// </summary>
        /// <param name="settings">The settings.</param>
        public UpscaleService(Settings settings, IMediaService mediaService)
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
        /// Load the upscale pipeline
        /// </summary>
        /// <param name="config">The configuration.</param>
        public async Task LoadAsync(UpscaleModel model, Device device)
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

                    _currentConfig = new UpscalerConfig
                    {
                        Channels = model.Channels,
                        Normalization = model.Normalization,
                        OutputNormalization = model.OutputNormalization,
                        SampleSize = model.SampleSize,
                        ScaleFactor = model.ScaleFactor,
                        Path = model.Path
                    };
                    _currentConfig.SetProvider(device.GetProvider());
                    _currentPipeline = UpscalePipeline.Create(_currentConfig);
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
        /// Execute the upscaler
        /// </summary>
        /// <param name="request">The request.</param>
        public async Task<ImageInput> ExecuteAsync(UpscaleImageRequest options)
        {
            try
            {
                IsExecuting = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var imageOptions = new UpscaleImageOptions
                    {
                        Image = options.Image,
                        MaxTileSize = options.MaxTileSize,
                        TileMode = options.TileMode,
                        TileOverlap = options.TileOverlap
                    };

                    var imageTensor = await Task.Run(() => _currentPipeline.RunAsync(imageOptions, cancellationToken: _cancellationTokenSource.Token));
                    return new ImageInput(imageTensor);
                }
            }
            finally
            {
                IsExecuting = false;
            }
        }


        /// <summary>
        /// Execute as an asynchronous operation.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <returns>A Task&lt;VideoInputStream&gt; representing the asynchronous operation.</returns>
        public async Task<VideoInputStream> ExecuteAsync(UpscaleVideoRequest options, IProgress<RunProgress> progressCallback)
        {
            try
            {
                IsExecuting = true;
                var resultVideoFile = _mediaService.GetTempVideoFile();
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var frameCount = options.VideoStream.FrameCount;
                    var cancellationToken = _cancellationTokenSource.Token;

                    async Task<VideoFrame> FrameProcessor(VideoFrame frame)
                    {
                        var processedFrame = await _currentPipeline.RunAsync(new UpscaleImageOptions
                        {
                            Image = frame.Frame,
                            MaxTileSize = options.MaxTileSize,
                            TileMode = options.TileMode,
                            TileOverlap = options.TileOverlap
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
            IsLoading = false;
            IsExecuting = false;
        }
    }


    public interface IUpscaleService
    {
        bool IsLoaded { get; }
        bool IsLoading { get; }
        bool IsExecuting { get; }
        bool CanCancel { get; }
        Task LoadAsync(UpscaleModel model, Device device);
        Task UnloadAsync();
        Task CancelAsync();
        Task<ImageInput> ExecuteAsync(UpscaleImageRequest options);
        Task<VideoInputStream> ExecuteAsync(UpscaleVideoRequest options, IProgress<RunProgress> progressCallback);
    }


    public record UpscaleImageRequest
    {
        public TileMode TileMode { get; init; }
        public int MaxTileSize { get; init; }
        public int TileOverlap { get; init; }
        public ImageInput Image { get; set; }
    }


    public record UpscaleVideoRequest
    {
        public TileMode TileMode { get; init; }
        public int MaxTileSize { get; init; }
        public int TileOverlap { get; init; }
        public VideoInputStream VideoStream { get; set; }
    }

}
