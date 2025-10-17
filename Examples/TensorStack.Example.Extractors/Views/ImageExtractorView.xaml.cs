using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Example.Common;
using TensorStack.Example.Services;
using TensorStack.Image;
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    /// <summary>
    /// Interaction logic for ImageExtractorView.xaml
    /// </summary>
    public partial class ImageExtractorView : ViewBase
    {
        private Device _selectedDevice;
        private ExtractorModel _selectedModel;
        private ImageInput _sourceImage;
        private ImageInput _resultImage;
        private ImageInput _compareImage;
        private TileMode _tileMode;
        private int _tileSize = 512;
        private int _tileOverlap = 16;

        public ImageExtractorView(Settings settings, NavigationService navigationService, IExtractorService extractorService)
            : base(settings, navigationService)
        {
            ExtractorService = extractorService;
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            UnloadCommand = new AsyncRelayCommand(UnloadAsync, CanUnload);
            ExecuteCommand = new AsyncRelayCommand(ExecuteAsync, CanExecute);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanCancel);
            SelectedModel = settings.ExtractorModels.First(x => x.IsDefault);
            SelectedDevice = settings.DefaultDevice;
            Progress = new ProgressInfo();
            InitializeComponent();
        }

        public override int Id => (int)View.ImageExtractor;
        public IExtractorService ExtractorService { get; }
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

        public ExtractorModel SelectedModel
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


        private async Task LoadAsync()
        {
            var timestamp = Stopwatch.GetTimestamp();
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
            Progress.Indeterminate();
            CompareImage = default;

            // Run Extractor
            var resultImage = await ExtractorService.ExecuteAsync(new ExtractorImageRequest
            {
                Image = _sourceImage,
                TileMode = _tileMode,
                MaxTileSize = _tileSize,
                TileOverlap = _tileOverlap
            });

            // Set Result
            ResultImage = resultImage;
            CompareImage = SourceImage;

            Progress.Clear();
            Debug.WriteLine($"[{GetType().Name}] [ExecuteAsync] - {Stopwatch.GetElapsedTime(timestamp)}");
        }


        private bool CanExecute()
        {
            return _sourceImage is not null && ExtractorService.IsLoaded && !ExtractorService.IsExecuting;
        }


        private async Task CancelAsync()
        {
            await ExtractorService.CancelAsync();
        }


        private bool CanCancel()
        {
            return ExtractorService.CanCancel;
        }

    }
}