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
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for VideoUpscaleView.xaml
    /// </summary>
    public partial class VideoUpscaleView : ViewBase
    {
        private Device _selectedDevice;
        private UpscaleModel _selectedModel;
        private VideoInputStream _sourceVideo;
        private VideoInputStream _resultVideo;
        private VideoInputStream _compareVideo;
        private TileMode _tileMode;
        private int _tileSize = 512;
        private int _tileOverlap = 16;
        private IProgress<RunProgress> _progressCallback;

        public VideoUpscaleView(Settings settings, NavigationService navigationService, IUpscaleService upscaleService)
            : base(settings, navigationService)
        {
            UpscaleService = upscaleService;
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            UnloadCommand = new AsyncRelayCommand(UnloadAsync, CanUnload);
            ExecuteCommand = new AsyncRelayCommand(ExecuteAsync, CanExecute);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanCancel);
            SelectedModel = settings.UpscaleModels.First(x => x.IsDefault);
            SelectedDevice = settings.DefaultDevice;
            Progress = new ProgressInfo();
            _progressCallback = new Progress<RunProgress>(OnProgress);
            InitializeComponent();
        }



        public override int Id => (int)View.VideoUpscale;
        public IUpscaleService UpscaleService { get; }
        public AsyncRelayCommand LoadCommand { get; set; }
        public AsyncRelayCommand UnloadCommand { get; set; }
        public AsyncRelayCommand ExecuteCommand { get; set; }
        public AsyncRelayCommand CancelCommand { get; set; }
        public ProgressInfo Progress { get; set; }

        public Device SelectedDevice
        {
            get { return _selectedDevice; }
            set { SetProperty(ref _selectedDevice, value); }
        }

        public UpscaleModel SelectedModel
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


        public override Task OpenAsync(OpenViewArgs args = null)
        {
            if (UpscaleService.IsLoaded)
            {
                SelectedModel = UpscaleService.Model;
            }
            return base.OpenAsync(args);
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

            await UpscaleService.LoadAsync(SelectedModel, device);

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [LoadAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanLoad()
        {
            return SelectedModel is not null
                && !UpscaleService.IsLoaded;
        }


        private async Task UnloadAsync()
        {
            await UpscaleService.UnloadAsync();
        }


        private bool CanUnload()
        {
            return UpscaleService.IsLoaded;
        }


        private async Task ExecuteAsync()
        {
            await ResultControl.ClearAsync();
            var timestamp = Stopwatch.GetTimestamp();

            // Run Upscaler
            var upscaledVideo = await UpscaleService.ExecuteAsync(new UpscaleVideoRequest
            {
                VideoStream = _sourceVideo,
                TileMode = _tileMode,
                MaxTileSize = _tileSize,
                TileOverlap = _tileOverlap,
            }, _progressCallback);

            // Set Result
            ResultVideo = upscaledVideo;
            CompareVideo = SourceVideo;

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return _sourceVideo is not null && UpscaleService.IsLoaded && !UpscaleService.IsExecuting;
        }


        private async Task CancelAsync()
        {
            await UpscaleService.CancelAsync();
        }


        private bool CanCancel()
        {
            return UpscaleService.CanCancel;
        }


        private void OnProgress(RunProgress progress)
        {
            Progress.Update(progress.Value + 1, progress.Maximum, progress.Message);
        }


        private async Task<bool> IsModelValidAsync()
        {
            if (File.Exists(SelectedModel.Path))
                return true;

            return await DialogService.DownloadAsync($"Download '{SelectedModel.Name}' upscale model?", SelectedModel.UrlPath, SelectedModel.Path);
        }
    }
}