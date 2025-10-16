using System;
using System.Diagnostics;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Example.Services;
using TensorStack.Video;
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for InterpolationView.xaml
    /// </summary>
    public partial class InterpolationView : ViewBase
    {
        private Device _selectedDevice;
        private VideoInputStream _sourceVideo;
        private VideoInputStream _resultVideo;
        private VideoInputStream _compareVideo;
        private int _multiplier = 2;
        private IProgress<RunProgress> _progressCallback;

        public InterpolationView(Settings settings, NavigationService navigationService, IInterpolationService interpolationService)
            : base(settings, navigationService)
        {
            InterpolationService = interpolationService;
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            UnloadCommand = new AsyncRelayCommand(UnloadAsync, CanUnload);
            ExecuteCommand = new AsyncRelayCommand(ExecuteAsync, CanExecute);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanCancel);
            SelectedDevice = settings.DefaultDevice;
            Progress = new ProgressInfo();
            _progressCallback = new Progress<RunProgress>(OnProgress);
            InitializeComponent();
        }

        public override int Id => (int)View.Interpolation;
        public IInterpolationService InterpolationService { get; }
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

        public int Multiplier
        {
            get { return _multiplier; }
            set { SetProperty(ref _multiplier, value); }
        }


        private async Task LoadAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            Progress.Indeterminate();

            var device = _selectedDevice;
            if (_selectedDevice is null)
                device = Settings.DefaultDevice;

            await InterpolationService.LoadAsync(device);

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [LoadAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanLoad()
        {
            return !InterpolationService.IsLoaded;
        }


        private async Task UnloadAsync()
        {
            await InterpolationService.UnloadAsync();
        }


        private bool CanUnload()
        {
            return InterpolationService.IsLoaded;
        }


        private async Task ExecuteAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
            Progress.Clear();
            ResultVideo = default;
            CompareVideo = default;

            // Run Interpolation
            var resultVideo = await InterpolationService.ExecuteAsync(new InterpolationRequest
            {
                VideoStream = _sourceVideo,
                Frames = _sourceVideo.FrameCount,
                FrameRate = _sourceVideo.FrameRate,
                Multiplier = _multiplier
            }, _progressCallback);

            // Set Result
            ResultVideo = resultVideo;
            CompareVideo = SourceVideo;

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return _sourceVideo is not null && InterpolationService.IsLoaded && !InterpolationService.IsExecuting;
        }


        private async Task CancelAsync()
        {
            await InterpolationService.CancelAsync();
        }


        private bool CanCancel()
        {
            return InterpolationService.CanCancel;
        }


        private void OnProgress(RunProgress progress)
        {
            Progress.Update(progress.Value + 1, progress.Maximum, progress.Message);
        }

    }
}