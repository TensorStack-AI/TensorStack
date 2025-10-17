using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Example.Common;
using TensorStack.Example.Services;
using TensorStack.Extractors.Common;
using TensorStack.Video;
using TensorStack.WPF;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for VideoBackgroundView.xaml
    /// </summary>
    public partial class VideoBackgroundView : ViewBase
    {
        private VideoInputStream _sourceVideo;
        private VideoInputStream _resultVideo;
        private VideoInputStream _compareVideo;
        private Device _selectedDevice;
        private BackgroundModel _selectedModel;
        private IProgress<RunProgress> _progressCallback;

        public VideoBackgroundView(Settings settings, NavigationService navigationService, IBackgroundService backgroundService)
            : base(settings, navigationService)
        {
            BackgroundService = backgroundService;
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            UnloadCommand = new AsyncRelayCommand(UnloadAsync, CanUnload);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanCancel);
            MaskBackgroundCommand = new AsyncRelayCommand(MaskBackgroundAsync, CanExecute);
            MaskForegroundCommand = new AsyncRelayCommand(MaskForegroundAsync, CanExecute);
            RemoveBackgroundCommand = new AsyncRelayCommand(RemoveBackgroundAsync, CanExecute);
            RemoveForegroundCommand = new AsyncRelayCommand(RemoveForegroundAsync, CanExecute);
            SelectedModel = Settings.BackgroundModels.First(x => x.IsDefault);
            SelectedDevice = settings.DefaultDevice;
            _progressCallback = new Progress<RunProgress>(OnProgress);
            InitializeComponent();
        }

        public override int Id => (int)View.VideoBackground;
        public IBackgroundService BackgroundService { get; }
        public AsyncRelayCommand LoadCommand { get; set; }
        public AsyncRelayCommand UnloadCommand { get; set; }
        public AsyncRelayCommand CancelCommand { get; set; }
        public AsyncRelayCommand MaskBackgroundCommand { get; set; }
        public AsyncRelayCommand MaskForegroundCommand { get; set; }
        public AsyncRelayCommand RemoveBackgroundCommand { get; set; }
        public AsyncRelayCommand RemoveForegroundCommand { get; set; }

        public Device SelectedDevice
        {
            get { return _selectedDevice; }
            set { SetProperty(ref _selectedDevice, value); }
        }

        public BackgroundModel SelectedModel
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


        private async Task LoadAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            if (!await IsModelValidAsync())
                return;

            Progress.Indeterminate();
            var device = _selectedDevice;
            if (_selectedDevice is null)
                device = Settings.DefaultDevice;

            await BackgroundService.LoadAsync(SelectedModel, device);

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [LoadAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanLoad()
        {
            return SelectedModel is not null && !BackgroundService.IsLoaded;
        }


        private async Task UnloadAsync()
        {
            await BackgroundService.UnloadAsync();
        }


        private bool CanUnload()
        {
            return BackgroundService.IsLoaded;
        }


        private async Task MaskBackgroundAsync()
        {
            await ExecuteAsync(new BackgroundVideoRequest
            {
                VideoStream = _sourceVideo,
                Mode = BackgroundMode.MaskBackground
            });
        }


        private async Task MaskForegroundAsync()
        {
            await ExecuteAsync(new BackgroundVideoRequest
            {
                VideoStream = _sourceVideo,
                Mode = BackgroundMode.MaskForeground
            });
        }


        private async Task RemoveBackgroundAsync()
        {
            await ExecuteAsync(new BackgroundVideoRequest
            {
                VideoStream = _sourceVideo,
                Mode = BackgroundMode.RemoveBackground
            });
        }


        private async Task RemoveForegroundAsync()
        {
            await ExecuteAsync(new BackgroundVideoRequest
            {
                VideoStream = _sourceVideo,
                Mode = BackgroundMode.RemoveForeground
            });
        }


        private async Task ExecuteAsync(BackgroundVideoRequest options)
        {
            var timestamp = Stopwatch.GetTimestamp();
            Progress.Clear();
            ResultVideo = default;
            CompareVideo = default;

            // Run Background Service
            var resultVideo = await BackgroundService.ExecuteAsync(options, _progressCallback);

            // Set Result
            ResultVideo = resultVideo;
            CompareVideo = SourceVideo;

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return _sourceVideo is not null && BackgroundService.IsLoaded;
        }


        private async Task CancelAsync()
        {
            await BackgroundService.CancelAsync();
        }


        private bool CanCancel()
        {
            return BackgroundService.CanCancel;
        }


        private void OnProgress(RunProgress progress)
        {
            Progress.Update(progress.Value + 1, progress.Maximum, progress.Message);
        }


        private async Task<bool> IsModelValidAsync()
        {
            if (File.Exists(SelectedModel.Path))
                return true;

            return await DialogService.DownloadAsync($"Download '{SelectedModel.Name}' background model?", SelectedModel.UrlPath, SelectedModel.Path);
        }
    }
}