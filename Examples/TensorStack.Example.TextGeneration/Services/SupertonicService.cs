using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Example.Common;
using TensorStack.Providers;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Pipelines.Supertonic;

namespace TensorStack.Example.Services
{
    public class SupertonicService : ServiceBase, ISupertonicService
    {
        private readonly Settings _settings;
        private IPipeline _currentPipeline;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isLoaded;
        private bool _isLoading;
        private bool _isExecuting;

        /// <summary>
        /// Initializes a new instance of the <see cref="SupertonicService"/> class.
        /// </summary>
        /// <param name="settings">The settings.</param>
        public SupertonicService(Settings settings)
        {
            _settings = settings;
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
        public async Task LoadAsync(TextModel model, Device device)
        {
            try
            {
                IsLoaded = false;
                IsLoading = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var cancellationToken = _cancellationTokenSource.Token;
                    if (_currentPipeline != null)
                        await _currentPipeline.UnloadAsync(cancellationToken);

                    var provider = device.GetProvider();
                    _currentPipeline = SupertonicPipeline.Create(model.Path, provider);
                    await Task.Run(() => _currentPipeline.LoadAsync(cancellationToken), cancellationToken);

                }
            }
            catch (OperationCanceledException)
            {
                _currentPipeline?.Dispose();
                _currentPipeline = null;
                throw;
            }
            finally
            {
                IsLoaded = true;
                IsLoading = false;
            }
        }


        /// <summary>
        /// Execute the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        public async Task<AudioTensor> ExecuteAsync(SupertonicRequest options)
        {
            try
            {
                IsExecuting = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var pipeline = _currentPipeline as IPipeline<AudioTensor, SupertonicOptions, GenerateProgress>;
                    var pipelineOptions = new SupertonicOptions
                    {
                        TextInput = options.InputText,
                        VoiceStyle = options.VoiceStyle,
                        Steps = options.Steps,
                        Speed = options.Speed,
                        SilenceDuration = options.SilenceDuration,
                        Seed = options.Seed,
                    };

                    return await pipeline.RunAsync(pipelineOptions, cancellationToken: _cancellationTokenSource.Token);
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
            }

            IsLoaded = false;
            IsLoading = false;
            IsExecuting = false;
        }
    }


    public interface ISupertonicService
    {
        bool IsLoaded { get; }
        bool IsLoading { get; }
        bool IsExecuting { get; }
        bool CanCancel { get; }
        Task LoadAsync(TextModel model, Device device);
        Task UnloadAsync();
        Task CancelAsync();
        Task<AudioTensor> ExecuteAsync(SupertonicRequest options);
    }


    public record SupertonicRequest
    {
        public string InputText { get; set; }
        public string VoiceStyle { get; set; }
        public int Steps { get; set; } = 5;
        public float Speed { get; set; } = 1f;
        public float SilenceDuration { get; set; } = 0.3f;
        public int Seed { get; set; }
    }

}
