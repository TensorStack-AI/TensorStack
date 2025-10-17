using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Example.Common;
using TensorStack.Example.Services;
using TensorStack.Extractors.Common;
using TensorStack.Image;
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for ImageBackgroundView.xaml
    /// </summary>
    public partial class ImageBackgroundView : ViewBase
    {
        private ImageInput _sourceImage;
        private ImageInput _resultImage;
        private ImageInput _compareImage;
        private Device _selectedDevice;
        private BackgroundModel _selectedModel;

        public ImageBackgroundView(Settings settings, NavigationService navigationService, IBackgroundService backgroundService)
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
            Progress = new ProgressInfo();
            InitializeComponent();
        }

        public override int Id => (int)View.ImageBackground;
        public IBackgroundService BackgroundService { get; }
        public AsyncRelayCommand LoadCommand { get; set; }
        public AsyncRelayCommand UnloadCommand { get; set; }
        public AsyncRelayCommand CancelCommand { get; set; }
        public AsyncRelayCommand MaskBackgroundCommand { get; set; }
        public AsyncRelayCommand MaskForegroundCommand { get; set; }
        public AsyncRelayCommand RemoveBackgroundCommand { get; set; }
        public AsyncRelayCommand RemoveForegroundCommand { get; set; }
        public ProgressInfo Progress { get; set; }

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

        public ImageInput SourceImage
        {
            get { return _sourceImage; }
            set { SetProperty(ref _sourceImage, value); }
        }

        public ImageInput ResultImage
        {
            get { return _resultImage; }
            set { SetProperty(ref _resultImage, value); }
        }

        public ImageInput CompareImage
        {
            get { return _compareImage; }
            set { SetProperty(ref _compareImage, value); }
        }


        private async Task LoadAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
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
            await ExecuteAsync(new BackgroundImageRequest
            {
                Image = _sourceImage,
                Mode = BackgroundMode.MaskBackground,
                IsTransparentSupported = true
            });
        }


        private async Task MaskForegroundAsync()
        {
            await ExecuteAsync(new BackgroundImageRequest
            {
                Image = _sourceImage,
                Mode = BackgroundMode.MaskForeground,
                IsTransparentSupported = true
            });
        }


        private async Task RemoveBackgroundAsync()
        {
            await ExecuteAsync(new BackgroundImageRequest
            {
                Image = _sourceImage,
                Mode = BackgroundMode.RemoveBackground,
                IsTransparentSupported = true
            });
        }


        private async Task RemoveForegroundAsync()
        {
            await ExecuteAsync(new BackgroundImageRequest
            {
                Image = _sourceImage,
                Mode = BackgroundMode.RemoveForeground,
                IsTransparentSupported = false
            });
        }


        private async Task ExecuteAsync(BackgroundImageRequest options)
        {
            var timestamp = Stopwatch.GetTimestamp();
            Progress.Indeterminate();
            CompareImage = default;

            // Run Extractor
            var extractorImage = await BackgroundService.ExecuteAsync(options);

            // Set Result
            ResultImage = extractorImage;
            CompareImage = SourceImage;

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return _sourceImage is not null && BackgroundService.IsLoaded;
        }


        private async Task CancelAsync()
        {
            await BackgroundService.CancelAsync();
        }


        private bool CanCancel()
        {
            return BackgroundService.CanCancel;
        }

    }
}