using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Example.Common;
using TensorStack.Example.Services;
using TensorStack.Video;
using TensorStack.WPF;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for VideoExtractorView.xaml
    /// </summary>
    public partial class VideoExtractorView : ViewBase
    {
        private Device _selectedDevice;
        private ExtractorModel _selectedModel;
        private VideoInputStream _sourceVideo;
        private VideoInputStream _resultVideo;
        private VideoInputStream _compareVideo;
        private TileMode _tileMode;
        private int _tileSize = 512;
        private int _tileOverlap = 16;
        private bool _invertOutput;
        private bool _mergeOutput;
        private IProgress<RunProgress> _progressCallback;

        public VideoExtractorView(Settings settings, NavigationService navigationService, IExtractorService extractorService)
            : base(settings, navigationService)
        {
            ExtractorService = extractorService;
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            UnloadCommand = new AsyncRelayCommand(UnloadAsync, CanUnload);
            ExecuteCommand = new AsyncRelayCommand(ExecuteAsync, CanExecute);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanCancel);
            SelectedModel = settings.ExtractorModels.First(x => x.IsDefault);
            SelectedDevice = settings.DefaultDevice;
            _progressCallback = new Progress<RunProgress>(OnProgress);
            InitializeComponent();
        }

        public override int Id => (int)View.VideoExtractor;
        public IExtractorService ExtractorService { get; }
        public AsyncRelayCommand LoadCommand { get; set; }
        public AsyncRelayCommand UnloadCommand { get; set; }
        public AsyncRelayCommand ExecuteCommand { get; set; }
        public AsyncRelayCommand CancelCommand { get; set; }

        public Device SelectedDevice
        {
            get { return _selectedDevice; }
            set { SetProperty(ref _selectedDevice, value); }
        }

        public ExtractorModel SelectedModel
        {
            get { return _selectedModel; }
            set { SetProperty(ref _selectedModel, value); }
        }

        public VideoInputStream SourceVideo
        {
            get { return _sourceVideo; }
            set { SetProperty(ref _sourceVideo, value); }
        }

        public VideoInputStream ResultVideo
        {
            get { return _resultVideo; }
            set { SetProperty(ref _resultVideo, value); }
        }

        public VideoInputStream CompareVideo
        {
            get { return _compareVideo; }
            set { SetProperty(ref _compareVideo, value); }
        }

        public TileMode TileMode
        {
            get { return _tileMode; }
            set { SetProperty(ref _tileMode, value); }
        }

        public int TileSize
        {
            get { return _tileSize; }
            set { SetProperty(ref _tileSize, value); }
        }

        public int TileOverlap
        {
            get { return _tileOverlap; }
            set { SetProperty(ref _tileOverlap, value); }
        }

        public bool InvertOutput
        {
            get { return _invertOutput; }
            set { SetProperty(ref _invertOutput, value); }
        }

        public bool MergeOutput
        {
            get { return _mergeOutput; }
            set { SetProperty(ref _mergeOutput, value); }
        }


        private async Task LoadAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            if (!await IsModelValidAsync())
                return;

            Progress.Indeterminate();
            var device = _selectedDevice;
            if (_selectedDevice is null)
                device = Settings.DefaultDevice;

            await ExtractorService.LoadAsync(SelectedModel, device);

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [LoadAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanLoad()
        {
            return SelectedModel is not null
                && !ExtractorService.IsLoaded;
        }


        private async Task UnloadAsync()
        {
            await ExtractorService.UnloadAsync();
        }


        private bool CanUnload()
        {
            return ExtractorService.IsLoaded;
        }


        private async Task ExecuteAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            Progress.Clear();
            ResultVideo = default;
            CompareVideo = default;

            // Run Extractor
            var resultVideo = await ExtractorService.ExecuteAsync(new ExtractorVideoRequest
            {
                VideoStream = _sourceVideo,
                TileMode = _tileMode,
                MaxTileSize = _tileSize,
                TileOverlap = _tileOverlap,
                IsInverted = _invertOutput,
                MergeInput = _mergeOutput
            }, _progressCallback);

            // Set Result
            ResultVideo = resultVideo;
            CompareVideo = SourceVideo;

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return _sourceVideo is not null && ExtractorService.IsLoaded && !ExtractorService.IsExecuting;
        }


        private async Task CancelAsync()
        {
            await ExtractorService.CancelAsync();
        }


        private bool CanCancel()
        {
            return ExtractorService.CanCancel;
        }


        private void OnProgress(RunProgress progress)
        {
            Progress.Update(progress.Value + 1, progress.Maximum, progress.Message);
        }


        private async Task<bool> IsModelValidAsync()
        {
            if (File.Exists(SelectedModel.Path))
                return true;

            return await DialogService.DownloadAsync($"Download '{SelectedModel.Name}' extractor model?", SelectedModel.UrlPath, SelectedModel.Path);
        }
    }
}