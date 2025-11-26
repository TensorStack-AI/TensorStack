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
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for SupertonicView.xaml
    /// </summary>
    public partial class SupertonicView : ViewBase
    {
        private Device _selectedDevice;
        private TextModel _selectedModel;
        private string _selectedVoice;
        private AudioInput _audioResult;
        private int _steps = 10;
        private float _speed = 1f;
        private string _inputText;
        private int _seed;

        public SupertonicView(Settings settings, NavigationService navigationService, ISupertonicService supertonicService)
            : base(settings, navigationService)
        {
            TextService = supertonicService;
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            UnloadCommand = new AsyncRelayCommand(UnloadAsync, CanUnload);
            ExecuteCommand = new AsyncRelayCommand(ExecuteAsync, CanExecute);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanCancel);
            Progress = new ProgressInfo();
            SelectedModel = settings.TextToAudioModels.First(x => x.IsDefault);
            SelectedDevice = settings.DefaultDevice;
            Voices = new ObservableCollection<string>();
            InitializeComponent();
        }

        public override int Id => (int)View.Supertonic;
        public ISupertonicService TextService { get; }
        public AsyncRelayCommand LoadCommand { get; }
        public AsyncRelayCommand UnloadCommand { get; }
        public AsyncRelayCommand ExecuteCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }
        public ProgressInfo Progress { get; set; }
        public ObservableCollection<string> Voices { get; }

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

        public string SelectedVoice
        {
            get { return _selectedVoice; }
            set { SetProperty(ref _selectedVoice, value); }
        }

        public string InputText
        {
            get { return _inputText; }
            set { SetProperty(ref _inputText, value); }
        }

        public float Speed
        {
            get { return _speed; }
            set { SetProperty(ref _speed, value); }
        }

        public int Steps
        {
            get { return _steps; }
            set { SetProperty(ref _steps, value); }
        }

        public int Seed
        {
            get { return _seed; }
            set { SetProperty(ref _seed, value); }
        }

        public AudioInput AudioResult
        {
            get { return _audioResult; }
            set { SetProperty(ref _audioResult, value); }
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

            Voices.Clear();
            SelectedVoice = null;

            // Load Model
            await TextService.LoadAsync(_selectedModel, device);

            if (SelectedModel.Prefixes != null)
            {
                foreach (var prefix in SelectedModel.Prefixes)
                {
                    Voices.Add(prefix);
                }
                SelectedVoice = Voices.FirstOrDefault();
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
            AudioResult = null;

            // Run Transcribe
            var result = await TextService.ExecuteAsync(new SupertonicRequest
            {
                InputText = _inputText,
                VoiceStyle = _selectedVoice,
                Speed = _speed,
                Steps = _steps,
                Seed = _seed
            });

            AudioResult = new AudioInput( result);
         
            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return !string.IsNullOrEmpty(_inputText) && TextService.IsLoaded && !TextService.IsExecuting;
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
}