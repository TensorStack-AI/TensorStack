using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Example.Common;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Pipelines.Other;
using TensorStack.TextGeneration.Pipelines.Phi;
using TensorStack.Providers;
using TensorStack.TextGeneration.Pipelines.Whisper;
using TensorStack.Common.Tensor;

namespace TensorStack.Example.Services
{
    public class TextService : ServiceBase, ITextService
    {
        private readonly Settings _settings;
        private IPipeline _currentPipeline;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isLoaded;
        private bool _isLoading;
        private bool _isExecuting;
        private TransformerConfig _currentConfig;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextService"/> class.
        /// </summary>
        /// <param name="settings">The settings.</param>
        public TextService(Settings settings)
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
                    var providerCPU = Provider.GetProvider(DeviceType.CPU); // TODO: DirectML not working with decoder
                    if (model.Type == TextModelType.Phi3)
                    {
                        if (!Enum.TryParse<PhiType>(model.Version, true, out var phiType))
                            throw new ArgumentException("Invalid Phi Version");

                        _currentPipeline = Phi3Pipeline.Create(providerCPU, model.Path, phiType);
                    }
                    else if (model.Type == TextModelType.Summary)
                    {
                        _currentPipeline = SummaryPipeline.Create(provider, providerCPU, model.Path);
                    }
                    else if (model.Type == TextModelType.Whisper)
                    {
                        if (!Enum.TryParse<WhisperType>(model.Version, true, out var whisperType))
                            throw new ArgumentException("Invalid Whisper Version");

                        _currentPipeline = WhisperPipeline.Create(provider, providerCPU, model.Path, whisperType);
                    }
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
        public async Task<GenerateResult[]> ExecuteAsync(TextRequest options)
        {
            try
            {
                IsExecuting = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var pipelineOptions = new GenerateOptions
                    {
                        Prompt = options.Prompt,
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
                        EarlyStopping = options.EarlyStopping
                    };

                    var pipelineResult = await Task.Run(async () =>
                    {
                        if (options.Beams == 0)
                        {
                            // Greedy Search
                            var greedyPipeline = _currentPipeline as IPipeline<GenerateResult, GenerateOptions, GenerateProgress>;
                            return [await greedyPipeline.RunAsync(pipelineOptions, cancellationToken: _cancellationTokenSource.Token)];
                        }

                        // Beam Search
                        var beamSearchPipeline = _currentPipeline as IPipeline<GenerateResult[], SearchOptions, GenerateProgress>;
                        return await beamSearchPipeline.RunAsync(new SearchOptions(pipelineOptions), cancellationToken: _cancellationTokenSource.Token);
                    });

                    return pipelineResult;
                }
            }
            finally
            {
                IsExecuting = false;
            }
        }


        public async Task<GenerateResult[]> ExecuteAsync(WhisperRequest options)
        {
            try
            {
                IsExecuting = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var pipelineOptions = new WhisperOptions
                    {
                        Prompt = options.Prompt,
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
                        EarlyStopping = options.EarlyStopping,
                        AudioInput = options.AudioInput,
                        Language = options.Language,
                        Task = options.Task
                    };

                    var pipelineResult = await Task.Run(async () =>
                    {
                        if (options.Beams == 0)
                        {
                            // Greedy Search
                            var greedyPipeline = _currentPipeline as IPipeline<GenerateResult, WhisperOptions, GenerateProgress>;
                            return [await greedyPipeline.RunAsync(pipelineOptions, cancellationToken: _cancellationTokenSource.Token)];
                        }

                        // Beam Search
                        var beamSearchPipeline = _currentPipeline as IPipeline<GenerateResult[], WhisperSearchOptions, GenerateProgress>;
                        return await beamSearchPipeline.RunAsync(new WhisperSearchOptions(pipelineOptions), cancellationToken: _cancellationTokenSource.Token);
                    });

                    return pipelineResult;
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


    public interface ITextService
    {
        bool IsLoaded { get; }
        bool IsLoading { get; }
        bool IsExecuting { get; }
        bool CanCancel { get; }
        Task LoadAsync(TextModel model, Device device);
        Task UnloadAsync();
        Task CancelAsync();
        Task<GenerateResult[]> ExecuteAsync(TextRequest options);
        Task<GenerateResult[]> ExecuteAsync(WhisperRequest options);
    }


    public record TextRequest : ITransformerRequest
    {
        public string Prompt { get; set; }
        public int MinLength { get; set; } = 20;
        public int MaxLength { get; set; } = 200;
        public int NoRepeatNgramSize { get; set; } = 3;
        public int Seed { get; set; }
        public int Beams { get; set; } = 1;
        public int TopK { get; set; } = 1;
        public float TopP { get; set; } = 0.9f;
        public float Temperature { get; set; } = 1.0f;
        public float LengthPenalty { get; set; } = 1.0f;
        public EarlyStopping EarlyStopping { get; set; }
        public int DiversityLength { get; set; } = 5;
    }

    public record WhisperRequest : TextRequest
    {
        public AudioTensor AudioInput { get; set; }
        public LanguageType Language { get; set; } = LanguageType.EN;
        public TaskType Task { get; set; } = TaskType.Transcribe;
    }

}
