using System;
using System.Diagnostics;
using System.IO;
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
        private bool _invertOutput;
        private bool _mergeOutput;
        private BackgroundMode _selectedBackgroundMode = BackgroundMode.RemoveBackground;
        private int _detections = 0;
        private float _bodyConfidence = 0.4f;
        private float _jointConfidence = 0.1f;
        private float _colorAlpha = 0.8f;
        private float _jointRadius = 7f;
        private float _boneRadius = 8f;
        private float _boneThickness = 1f;
        private bool _isTransparent = false;

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
            InitializeComponent();
        }

        public override int Id => (int)View.ImageExtractor;
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

        public BackgroundMode SelectedBackgroundMode
        {
            get { return _selectedBackgroundMode; }
            set { SetProperty(ref _selectedBackgroundMode, value); }
        }

        public int Detections
        {
            get { return _detections; }
            set { SetProperty(ref _detections, value); }
        }

        public float BodyConfidence
        {
            get { return _bodyConfidence; }
            set { SetProperty(ref _bodyConfidence, value); }
        }

        public float JointConfidence
        {
            get { return _jointConfidence; }
            set { SetProperty(ref _jointConfidence, value); }
        }

        public float ColorAlpha
        {
            get { return _colorAlpha; }
            set { SetProperty(ref _colorAlpha, value); }
        }

        public float JointRadius
        {
            get { return _jointRadius; }
            set { SetProperty(ref _jointRadius, value); }
        }

        public float BoneRadius
        {
            get { return _boneRadius; }
            set { SetProperty(ref _boneRadius, value); }
        }

        public float BoneThickness
        {
            get { return _boneThickness; }
            set { SetProperty(ref _boneThickness, value); }
        }

        public bool IsTransparent
        {
            get { return _isTransparent; }
            set { SetProperty(ref _isTransparent, value); }
        }


        public override Task OpenAsync(OpenViewArgs args = null)
        {
            if (ExtractorService.IsLoaded)
            {
                SelectedModel = ExtractorService.Model;
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
            var resultImage = _selectedModel.Type switch
            {
                ExtractorType.Default => await ExecuteDefaultAsync(),
                ExtractorType.Background => await ExecuteBackgroundAsync(),
                ExtractorType.Pose => await ExecutePoseAsync(),
                _ => throw new NotImplementedException()
            };

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


        private async Task<bool> IsModelValidAsync()
        {
            if (File.Exists(SelectedModel.Path))
                return true;

            return await DialogService.DownloadAsync($"Download '{SelectedModel.Name}' extractor model?", SelectedModel.UrlPath, SelectedModel.Path);
        }


        private async Task<ImageInput> ExecuteDefaultAsync()
        {
            return await ExtractorService.ExecuteAsync(new ExtractorImageRequest
            {
                Image = _sourceImage,
                TileMode = _tileMode,
                MaxTileSize = _tileSize,
                TileOverlap = _tileOverlap,
                IsInverted = _invertOutput,
                MergeInput = _mergeOutput
            });
        }


        private async Task<ImageInput> ExecuteBackgroundAsync()
        {
            return await ExtractorService.ExecuteAsync(new BackgroundImageRequest
            {
                Image = _sourceImage,
                Mode = _selectedBackgroundMode,
                IsTransparentSupported = true
            });
        }


        private async Task<ImageInput> ExecutePoseAsync()
        {
            return await ExtractorService.ExecuteAsync(new PoseImageRequest
            {
                Image = _sourceImage,
                Detections = _detections,
                BodyConfidence = _bodyConfidence,
                BoneRadius = _boneRadius,
                BoneThickness = _boneThickness,
                ColorAlpha = _colorAlpha,
                IsTransparent = _isTransparent,
                JointConfidence = _jointConfidence,
                JointRadius = _jointRadius
            });
        }
    }
}