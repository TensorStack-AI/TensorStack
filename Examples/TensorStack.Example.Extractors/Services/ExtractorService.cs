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
    public class ExtractorService : ServiceBase, IExtractorService
    {
        private readonly Settings _settings;
        private readonly IMediaService _mediaService;
        private IPipeline _currentPipeline;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isLoaded;
        private bool _isLoading;
        private bool _isExecuting;
        private ExtractorConfig _currentConfig;

        /// <summary>
        /// Initializes a new instance of the <see cref="ExtractorService"/> class.
        /// </summary>
        /// <param name="settings">The settings.</param>
        public ExtractorService(Settings settings, IMediaService mediaService)
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
        public async Task LoadAsync(ExtractorModel model, Device device)
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
                        IsDynamicOutput = model.IsDynamicOutput,
                        SampleSize = model.SampleSize
                    };

                    _currentConfig.SetProvider(device.GetProvider());
                    _currentPipeline = model.Type switch
                    {
                        ExtractorType.Pose => PosePipeline.Create(_currentConfig),
                        ExtractorType.Background => BackgroundPipeline.Create(_currentConfig),
                        _ => ExtractorPipeline.Create(_currentConfig)
                    };
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
        /// Execute the image ExtractorPipeline
        /// </summary>
        /// <param name="request">The request.</param>
        public async Task<ImageInput> ExecuteAsync(ExtractorImageRequest options)
        {
            try
            {
                IsExecuting = true;
                var pipeline = _currentPipeline as ExtractorPipeline;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var imageTensor = await Task.Run(() => pipeline.RunAsync(new ExtractorImageOptions
                    {
                        Image = options.Image,
                        IsInverted = options.IsInverted,
                        MaxTileSize = options.MaxTileSize,
                        TileMode = options.TileMode,
                        TileOverlap = options.TileOverlap,
                        MergeInput = options.MergeInput
                    }, cancellationToken: _cancellationTokenSource.Token));

                    return new ImageInput(imageTensor);
                }
            }
            finally
            {
                IsExecuting = false;
            }
        }


        /// <summary>
        /// Execute the image BackgroundPipeline
        /// </summary>
        /// <param name="request">The request.</param>
        public async Task<ImageInput> ExecuteAsync(BackgroundImageRequest options)
        {
            try
            {
                IsExecuting = true;
                var pipeline = _currentPipeline as BackgroundPipeline;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var imageTensor = await Task.Run(() => pipeline.RunAsync(new BackgroundImageOptions
                    {
                        Mode = options.Mode,
                        Image = options.Image
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
        /// Execute the image PosePipeline
        /// </summary>
        /// <param name="request">The request.</param>
        public async Task<ImageInput> ExecuteAsync(PoseImageRequest options)
        {
            try
            {
                IsExecuting = true;
                var pipeline = _currentPipeline as PosePipeline;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var imageTensor = await Task.Run(() => pipeline.RunAsync(new PoseImageOptions
                    {
                        Image = options.Image,
                        BodyConfidence = options.BodyConfidence,
                        BoneRadius = options.BoneRadius,
                        BoneThickness = options.BoneThickness,
                        ColorAlpha = options.ColorAlpha,
                        Detections = options.Detections,
                        IsTransparent = options.IsTransparent,
                        JointConfidence = options.JointConfidence,
                        JointRadius = options.JointRadius,
                    }, cancellationToken: _cancellationTokenSource.Token));

                    return new ImageInput(imageTensor);
                }
            }
            finally
            {
                IsExecuting = false;
            }
        }


        /// <summary>
        /// Execute the video ExtractorPipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <returns>A Task&lt;VideoInputStream&gt; representing the asynchronous operation.</returns>
        public async Task<VideoInputStream> ExecuteAsync(ExtractorVideoRequest options, IProgress<RunProgress> progressCallback)
        {
            try
            {
                IsExecuting = true;
                var pipeline = _currentPipeline as ExtractorPipeline;
                var resultVideoFile = FileHelper.RandomFileName(_settings.DirectoryTemp, "mp4");
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var frameCount = options.VideoStream.FrameCount;
                    var cancellationToken = _cancellationTokenSource.Token;

                    async Task<VideoFrame> FrameProcessor(VideoFrame frame)
                    {
                        var processedFrame = await pipeline.RunAsync(new ExtractorImageOptions
                        {
                            Image = frame.Frame,
                            IsInverted = options.IsInverted,
                            MaxTileSize = options.MaxTileSize,
                            TileMode = options.TileMode,
                            TileOverlap = options.TileOverlap,
                            MergeInput = options.MergeInput
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
        /// Execute the video BackgroundPipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <returns>A Task&lt;VideoInputStream&gt; representing the asynchronous operation.</returns>
        public async Task<VideoInputStream> ExecuteAsync(BackgroundVideoRequest options, IProgress<RunProgress> progressCallback)
        {
            try
            {
                IsExecuting = true;
                var pipeline = _currentPipeline as BackgroundPipeline;
                var resultVideoFile = FileHelper.RandomFileName(_settings.DirectoryTemp, "mp4");
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var frameCount = options.VideoStream.FrameCount;
                    var cancellationToken = _cancellationTokenSource.Token;

                    async Task<VideoFrame> FrameProcessor(VideoFrame frame)
                    {
                        var processedFrame = await pipeline.RunAsync(new BackgroundImageOptions
                        {
                            Image = frame.Frame,
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
        /// Execute the video PosePipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <returns>A Task&lt;VideoInputStream&gt; representing the asynchronous operation.</returns>
        public async Task<VideoInputStream> ExecuteAsync(PoseVideoRequest options, IProgress<RunProgress> progressCallback)
        {
            try
            {
                IsExecuting = true;
                var pipeline = _currentPipeline as PosePipeline;
                var resultVideoFile = FileHelper.RandomFileName(_settings.DirectoryTemp, "mp4");
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var frameCount = options.VideoStream.FrameCount;
                    var cancellationToken = _cancellationTokenSource.Token;

                    async Task<VideoFrame> FrameProcessor(VideoFrame frame)
                    {
                        var processedFrame = await pipeline.RunAsync(new PoseImageOptions
                        {
                            Image = frame.Frame,
                            BodyConfidence = options.BodyConfidence,
                            BoneRadius = options.BoneRadius,
                            BoneThickness = options.BoneThickness,
                            ColorAlpha = options.ColorAlpha,
                            Detections = options.Detections,
                            IsTransparent = options.IsTransparent,
                            JointConfidence = options.JointConfidence,
                            JointRadius = options.JointRadius,
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


    public interface IExtractorService
    {
        bool IsLoaded { get; }
        bool IsLoading { get; }
        bool IsExecuting { get; }
        bool CanCancel { get; }
        Task LoadAsync(ExtractorModel model, Device device);
        Task UnloadAsync();
        Task CancelAsync();
        Task<ImageInput> ExecuteAsync(ExtractorImageRequest options);
        Task<ImageInput> ExecuteAsync(BackgroundImageRequest options);
        Task<ImageInput> ExecuteAsync(PoseImageRequest options);
        Task<VideoInputStream> ExecuteAsync(ExtractorVideoRequest options, IProgress<RunProgress> progressCallback);
        Task<VideoInputStream> ExecuteAsync(BackgroundVideoRequest options, IProgress<RunProgress> progressCallback);
        Task<VideoInputStream> ExecuteAsync(PoseVideoRequest options, IProgress<RunProgress> progressCallback);
    }


    public record ExtractorImageRequest
    {
        public TileMode TileMode { get; init; }
        public int MaxTileSize { get; init; }
        public int TileOverlap { get; init; }
        public bool IsInverted { get; init; }
        public bool MergeInput { get; init; }
        public ImageInput Image { get; init; }
    }


    public record ExtractorVideoRequest
    {
        public TileMode TileMode { get; init; }
        public int MaxTileSize { get; init; }
        public int TileOverlap { get; init; }
        public bool IsInverted { get; init; }
        public bool MergeInput { get; init; }
        public VideoInputStream VideoStream { get; init; }
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


    public record PoseImageRequest
    {
        public int Detections { get; set; } = 0;
        public float BodyConfidence { get; init; } = 0.4f;
        public float JointConfidence { get; init; } = 0.1f;
        public float ColorAlpha { get; init; } = 0.8f;
        public float JointRadius { get; init; } = 7f;
        public float BoneRadius { get; init; } = 8f;
        public float BoneThickness { get; init; } = 1f;
        public bool IsTransparent { get; set; }
        public ImageInput Image { get; init; }
    }


    public record PoseVideoRequest
    {
        public int Detections { get; set; } = 0;
        public float BodyConfidence { get; init; } = 0.4f;
        public float JointConfidence { get; init; } = 0.1f;
        public float ColorAlpha { get; init; } = 0.8f;
        public float JointRadius { get; init; } = 7f;
        public float BoneRadius { get; init; } = 8f;
        public float BoneThickness { get; init; } = 1f;
        public bool IsTransparent { get; set; }
        public VideoInputStream VideoStream { get; init; }
    }
}
