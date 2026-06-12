using Amuse.App.Common;
using Amuse.App.Services;
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
    /// Interaction logic for TextToAudioView.xaml
    /// </summary>
    public partial class TextToAudioView : ViewBaseDiffusion
    {
        private TextInput _sourceText;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextToAudioView"/> class.
        /// </summary>
        public TextToAudioView(Settings settings, NavigationService navigationService, IModelDownloadService downloadService, IDiffusionService diffusionService, IExtractService extractService, IUpscaleService upscaleService, IHistoryService historyService, ILogger<TextToAudioView> logger)
            : base(settings, navigationService, downloadService, diffusionService, extractService, upscaleService, historyService, logger)
        {
            _sourceText = new TextInput(string.Empty);
            InitializeComponent();
        }

        /// <summary>
        /// Gets the view.
        /// </summary>
        public override View View => View.TextToAudio;


        public TextInput SourceText
        {
            get { return _sourceText; }
            set { SetProperty(ref _sourceText, value); }
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
            Logger.LogInformation($"[TextToAudio] [Execute] Executing pipeline...");

            try
            {
                Progress.Clear();
                Statistics.Clear();
                ResultAudio = default;
                Statistics.Start();

                // Options
                var options = Options with { Prompt = _sourceText.Text };

                // Execute
                var resultTensor = await ExecuteAudioDiffusionAsync(options);

                // Result
                Statistics.Stop();
                ResultAudio = await SaveHistoryAsync(resultTensor, options);

                Logger.LogInformation("[TextToAudio] [Execute] Executing pipeline complete, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (OperationCanceledException)
            {
                Statistics.Clear();
                Logger.LogInformation("[TextToAudio] [Execute] Executing pipeline cancelled, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (Exception ex)
            {
                Statistics.Clear();
                IsPipelineLoaded = DiffusionService.IsLoaded;
                Logger.LogError(ex, "[TextToAudio] [Execute] An exception occurred executing pipeline, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
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
            Logger.LogInformation($"[TextToAudio] [ExecuteAutomation] Executing pipeline...");

            try
            {
                Progress.Clear();
                AutomationProgress.Clear();
                Statistics.Clear();
                ResultAudio = default;
                Statistics.Start();
                CancellationTokenSource = new CancellationTokenSource();

                AutomationProgress.Indeterminate($"Automation Started");
                var cancellationToken = CancellationTokenSource.Token;
                await foreach (var automationJob in AutomationManager.CreateJobsAsync(AutomationOptions, Options, MediaType.Audio, MediaType.Text))
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Source
                    if (!automationJob.InputTexts.IsNullOrEmpty())
                        SourceText = automationJob.InputTexts[0];

                    // Diffusion
                    var options = automationJob.DiffusionOptions with { Prompt = _sourceText.Text };
                    var resultTensor = await ExecuteAudioDiffusionAsync(options);

                    // Result
                    ResultAudio = resultTensor;

                    // History
                    if (AutomationOptions.IsHistoryEnabled)
                    {
                        await SaveHistoryAsync(ResultAudio, automationJob.DiffusionOptions);
                    }

                    await automationJob.SaveAsync(ResultAudio);
                    AutomationProgress.Update(automationJob.Id, automationJob.Count, $"Automation: {automationJob.Id}/{automationJob.Count}");
                }

                Statistics.Stop();
                Logger.LogInformation("[TextToAudio] [ExecuteAutomation] Executing pipeline complete, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (OperationCanceledException)
            {
                Statistics.Clear();
                Logger.LogInformation("[TextToAudio] [ExecuteAutomation] Executing pipeline cancelled, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
            }
            catch (Exception ex)
            {
                Statistics.Clear();
                IsPipelineLoaded = DiffusionService.IsLoaded;
                Logger.LogError(ex, "[TextToAudio] [ExecuteAutomation] An exception occurred executing pipeline, Elapsed: {Elapsed:c}", Stopwatch.GetElapsedTime(timestamp));
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
        private async Task<AudioInputStream> SaveHistoryAsync(AudioInputStream audioStream, DiffusionInputOptions options)
        {
            Logger.LogInformation($"[TextToAudio] [SaveHistory] Saving history...");
            var result = await HistoryService.AddAsync(audioStream, new DiffusionHistory
            {
                Options = options,
                Model = CurrentPipeline.DiffusionModel.Name,
                LoraModels = CurrentPipeline.LoraAdapterModel?.Select(x => x.Name).ToArray(),
                Source = View.TextToAudio,
            });
            Logger.LogInformation($"[TextToAudio] [SaveHistory] History saved.");
            return result;
        }

    }
}
