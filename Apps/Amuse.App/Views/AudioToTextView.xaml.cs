using Amuse.App.Common;
using Amuse.App.Services;
using Amuse.Common;
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Audio;
using TensorStack.Common;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace Amuse.App.Views
{
    /// <summary>
    /// Interaction logic for AudioToTextView.xaml
    /// </summary>
    public partial class AudioToTextView : ViewBaseDiffusion
    {
        private AudioInputStream _sourceAudio;
        private bool _isMultipleResult;
        private int _selectedBeam;
        private string _previewResult;

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioToTextView"/> class.
        /// </summary>
        public AudioToTextView(Settings settings, NavigationService navigationService, IModelDownloadService downloadService, IDiffusionService diffusionService, IExtractService extractService, IUpscaleService upscaleService, IHistoryService historyService, ILogger<AudioToTextView> logger)
            : base(settings, navigationService, downloadService, diffusionService, extractService, upscaleService, historyService, logger)
        {
            InitializeComponent();
        }

        /// <summary>
        /// Gets the view.
        /// </summary>
        public override View View => View.AudioToText;

        /// <summary>
        /// Gets or sets the source audio.
        /// </summary>
        public AudioInputStream SourceAudio
        {
            get { return _sourceAudio; }
            set { SetProperty(ref _sourceAudio, value); }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to show multiple results.
        /// </summary>
        public bool IsMultipleResult
        {
            get { return _isMultipleResult; }
            set { SetProperty(ref _isMultipleResult, value); }
        }

        /// <summary>
        /// Gets or sets the selected beam result.
        /// </summary>
        public int SelectedBeam
        {
            get { return _selectedBeam; }
            set { SetProperty(ref _selectedBeam, value); }
        }

        /// <summary>
        /// Gets or sets the preview result.
        /// </summary>
        public string PreviewResult
        {
            get { return _previewResult; }
            set { SetProperty(ref _previewResult, value); }
        }


        /// <summary>
        /// On View Open
        /// </summary>
        public override async Task OpenAsync(OpenViewArgs args = null)
        {
            await base.OpenAsync(args);
            if (!IsPipelineLoaded)
                ModelControl.SetPipeline(DiffusionService.Pipeline);
        }


        /// <summary>
        /// Execute thge pipeline.
        /// </summary>
        protected override async Task ExecuteAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            Logger.LogInformation($"[AudioToText] [Execute] Executing pipeline...");

            try
            {
                Progress.Clear();
                Statistics.Clear();
                ResultText = default;
                PreviewResult = default;
                Statistics.Start();

                // Options
                var options = Options with { };
                if (_sourceAudio != null)
                {
                    options.InputAudios = [_sourceAudio];
                }

                // Execute
                var textResult = await ExecuteTextDiffusionAsync(options);

                // Result
                Statistics.Stop();

                ResultText = textResult;
                SelectedBeam = 0;

                // History
                await SaveHistoryAsync(options);
                Logger.LogInformation("[AudioToText] [Execute] Executing pipeline complete, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (OperationCanceledException)
            {
                Statistics.Clear();
                Logger.LogInformation("[AudioToText] [Execute] Executing pipeline cancelled, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (Exception ex)
            {
                Statistics.Clear();
                IsPipelineLoaded = DiffusionService.IsLoaded;
                Logger.LogError(ex, "[AudioToText] [Execute] An exception occurred executing pipeline, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
                await DialogService.ShowErrorAsync("Execute Pipeline", ex.Message);
            }
            finally
            {
                Progress.Clear();
            }
        }


        /// <summary>
        /// Executes the pipeline automation.
        /// </summary>
        protected override async Task ExecuteAutomationAsync()
        {
            IsAutomating = true;
            var timestamp = Stopwatch.GetTimestamp();
            Logger.LogInformation($"[AudioToText] [ExecuteAutomation] Executing pipeline...");

            try
            {
                Progress.Clear();
                AutomationProgress.Clear();
                Statistics.Clear();
                ResultText = default;
                PreviewResult = default;
                Statistics.Start();
                CancellationTokenSource = new CancellationTokenSource();

                AutomationProgress.Indeterminate($"Automation Started");
                var cancellationToken = CancellationTokenSource.Token;
                await foreach (var automationJob in AutomationManager.CreateJobsAsync(AutomationOptions, Options, MediaType.Text, MediaType.Audio))
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    PreviewResult = default;

                    // Source
                    if (!automationJob.AudioStreams.IsNullOrEmpty())
                        SourceAudio = automationJob.AudioStreams[0];

                    // Diffusion
                    automationJob.DiffusionOptions.InputAudios = [_sourceAudio];
                    var textResult = await ExecuteTextDiffusionAsync(automationJob.DiffusionOptions);

                    // Result
                    ResultText = textResult;
                    SelectedBeam = 0;

                    // History
                    if (AutomationOptions.IsHistoryEnabled)
                    {
                        await SaveHistoryAsync(automationJob.DiffusionOptions);
                    }

                    await automationJob.SaveAsync(ResultAudio);
                    AutomationProgress.Update(automationJob.Id, automationJob.Count, $"Automation: {automationJob.Id}/{automationJob.Count}");
                }

                Statistics.Stop();
                Logger.LogInformation("[AudioToText] [ExecuteAutomation] Executing pipeline complete, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (OperationCanceledException)
            {
                Statistics.Clear();
                Logger.LogInformation("[AudioToText] [ExecuteAutomation] Executing pipeline cancelled, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (Exception ex)
            {
                Statistics.Clear();
                IsPipelineLoaded = DiffusionService.IsLoaded;
                Logger.LogError(ex, "[AudioToText] [ExecuteAutomation] An exception occurred executing pipeline, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
                await DialogService.ShowErrorAsync("Execute Automation", ex.Message);
            }
            finally
            {
                Progress.Clear();
                AutomationProgress.Clear();
                IsAutomating = false;
                CancellationTokenSource?.Dispose();
                CancellationTokenSource = null;
            }
        }


        /// <summary>
        /// Save history
        /// </summary>
        /// <param name="options">The options.</param>
        private async Task<TextInput> SaveHistoryAsync(DiffusionInputOptions options)
        {
            Logger.LogInformation($"[TextToImage] [SaveHistory] Saving history...");
            var result = await HistoryService.AddAsync(ResultText.Result, new DiffusionHistory
            {
                Options = options,
                Model = CurrentPipeline.DiffusionModel.Name,
                LoraModels = CurrentPipeline.LoraAdapterModel?.Select(x => x.Name).ToArray(),
                Source = View.AudioToText,
            });
            Logger.LogInformation($"[TextToImage] [SaveHistory] History saved.");
            return result;
        }


        protected override void OnProgress(PipelineProgress progress)
        {
            base.OnProgress(progress);
            if (bool.TryParse(progress.Subkey, out var isReset))
            {
                if (isReset)
                {
                    PreviewResult = progress.Message;
                    return;
                }
                PreviewResult += progress.Message;
            }
        }
    }
}
