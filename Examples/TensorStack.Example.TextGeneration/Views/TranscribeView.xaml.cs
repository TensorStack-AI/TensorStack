using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Audio;
using TensorStack.Common;
using TensorStack.Common.Common;
using TensorStack.Example.Common;
using TensorStack.Example.Services;
using TensorStack.TextGeneration.Common;
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for TranscribeView.xaml
    /// </summary>
    public partial class TranscribeView : ViewBase
    {
        private Device _selectedDevice;
        private TextModel _selectedModel;
        private int _topK = 50;
        private int _beams = 4;
        private int _seed = -1;
        private float _topP = 0.9f;
        private float _temperature = 1f;
        private float _lengthPenalty = 1f;
        private int _diversityLength = 512;
        private int _minLength = 20;
        private int _maxLength = 512;
        private EarlyStopping _earlyStopping = EarlyStopping.None;
        private string _selectedPrefix;
        private bool _isMultipleResult;
        private int _selectedBeam;
        private AudioInput _audioInput;

        public TranscribeView(Settings settings, NavigationService navigationService, ITextService textService)
            : base(settings, navigationService)
        {
            TextService = textService;
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            UnloadCommand = new AsyncRelayCommand(UnloadAsync, CanUnload);
            ExecuteCommand = new AsyncRelayCommand(ExecuteAsync, CanExecute);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanCancel);
            Progress = new ProgressInfo();
            SelectedModel = settings.TextModels.First(x => x.IsDefault);
            SelectedDevice = settings.DefaultDevice;
            Prefixes = new ObservableCollection<string>();
            TranscribeResults = new ObservableCollection<TranscribeResult>();
            InitializeComponent();
        }

        public override int Id => (int)View.Transcribe;
        public ITextService TextService { get; }
        public AsyncRelayCommand LoadCommand { get; }
        public AsyncRelayCommand UnloadCommand { get; }
        public AsyncRelayCommand ExecuteCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public ProgressInfo Progress { get; set; }
        public ObservableCollection<string> Prefixes { get; }
        public ObservableCollection<TranscribeResult> TranscribeResults { get; }

        public Device SelectedDevice
        {
            get { return _selectedDevice; }
            set { SetProperty(ref _selectedDevice, value); }
        }

        public TextModel SelectedModel
        {
            get { return _selectedModel; }
            set { SetProperty(ref _selectedModel, value); }
        }

        public int TopK
        {
            get { return _topK; }
            set { SetProperty(ref _topK, value); }
        }

        public int Beams
        {
            get { return _beams; }
            set { SetProperty(ref _beams, value); }
        }

        public int Seed
        {
            get { return _seed; }
            set { SetProperty(ref _seed, value); }
        }

        public float TopP
        {
            get { return _topP; }
            set { SetProperty(ref _topP, value); }
        }

        public float Temperature
        {
            get { return _temperature; }
            set { SetProperty(ref _temperature, value); }
        }

        public float LengthPenalty
        {
            get { return _lengthPenalty; }
            set { SetProperty(ref _lengthPenalty, value); }
        }

        public int DiversityLength
        {
            get { return _diversityLength; }
            set { SetProperty(ref _diversityLength, value); }
        }

        public int MinLength
        {
            get { return _minLength; }
            set { SetProperty(ref _minLength, value); }
        }

        public int MaxLength
        {
            get { return _maxLength; }
            set { SetProperty(ref _maxLength, value); }
        }

        public EarlyStopping EarlyStopping
        {
            get { return _earlyStopping; }
            set { SetProperty(ref _earlyStopping, value); }
        }

        public AudioInput AudioInput
        {
            get { return _audioInput; }
            set { SetProperty(ref _audioInput, value); }
        }

        public string SelectedPrefix
        {
            get { return _selectedPrefix; }
            set { SetProperty(ref _selectedPrefix, value); }
        }

        public bool IsMultipleResult
        {
            get { return _isMultipleResult; }
            set { SetProperty(ref _isMultipleResult, value); }
        }

        public int SelectedBeam
        {
            get { return _selectedBeam; }
            set { SetProperty(ref _selectedBeam, value); }
        }


        private async Task LoadAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            if (!await IsModelValidAsync())
                return;

            Progress.Indeterminate("Loading Model...");

            var device = _selectedDevice;
            if (_selectedDevice is null)
                device = Settings.DefaultDevice;

            Prefixes.Clear();
            SelectedPrefix = null;

            // Load Model
            await TextService.LoadAsync(_selectedModel, device);

            MinLength = SelectedModel.MinLength;
            MaxLength = SelectedModel.MaxLength;
            DiversityLength = SelectedModel.MinLength;
            if (SelectedModel.Prefixes != null)
            {
                foreach (var prefix in SelectedModel.Prefixes)
                {
                    Prefixes.Add(prefix);
                }
                SelectedPrefix = Prefixes.FirstOrDefault();
            }

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [LoadAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanLoad()
        {
            return SelectedModel is not null
                && !TextService.IsLoaded;
        }


        private async Task UnloadAsync()
        {
            await TextService.UnloadAsync();
        }


        private bool CanUnload()
        {
            return TextService.IsLoaded;
        }


        private async Task ExecuteAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            Progress.Indeterminate("Generating Results...");

            // Run Transcribe
            var transcribeResults = await TextService.ExecuteAsync(new TextRequest
            {
                //Prompt = promptText,
                Beams = _beams,
                TopK = _topK,
                Seed = _seed,
                TopP = _topP,
                Temperature = _temperature,
                LengthPenalty = _lengthPenalty,
                MinLength = _minLength,
                MaxLength = _maxLength,
                NoRepeatNgramSize = 4,
                DiversityLength = _diversityLength,
                EarlyStopping = _earlyStopping,
            });

            TranscribeResults.Clear();
            foreach (var transcribeResult in transcribeResults)
            {
                TranscribeResults.Add(new TranscribeResult($"Beam {transcribeResult.Beam}", transcribeResult.Result, transcribeResult.PenaltyScore));
            }
            SelectedBeam = 0;

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return AudioInput is not null && TextService.IsLoaded && !TextService.IsExecuting;
        }


        private async Task CancelAsync()
        {
            await TextService.CancelAsync();
        }


        private bool CanCancel()
        {
            return TextService.CanCancel;
        }


        private async Task<bool> IsModelValidAsync()
        {
            var modelFiles = FileHelper.GetUrlFileMapping(SelectedModel.UrlPaths, SelectedModel.Path);
            if (modelFiles.Values.All(File.Exists))
                return true;

            return await DialogService.DownloadAsync($"Download '{SelectedModel.Name}' model?", SelectedModel.UrlPaths, SelectedModel.Path);
        }
    }

    public record TranscribeResult(string Header, string Content, float Score);
}