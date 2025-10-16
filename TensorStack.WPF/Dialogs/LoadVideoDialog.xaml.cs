// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Input;
using TensorStack.Common;
using TensorStack.Common.Common;
using TensorStack.Video;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.WPF.Dialogs
{
    /// <summary>
    /// Interaction logic for LoadVideoDialog.xaml
    /// </summary>
    public partial class LoadVideoDialog : DialogControl
    {
        private readonly IUIConfiguration _configuration;
        private readonly IVideoService _videoService;
        private int _cropWidth;
        private int _cropHeight;
        private string _sourceVideo;
        private VideoInfo _sourceInfo;
        private bool _isFixedSizeEnabled;
        private int _selectedWidth;
        private int _selectedHeight;
        private float _selectedFrameRate;
        private int _selectedFrameCount;
        private ResizeMode _selectedResizeMode;
        private VideoInput _result;
        private bool _isGenerating;
        private CancellationTokenSource _cancellationTokenSource;

        /// <summary>
        /// Initializes a new instance of the <see cref="LoadVideoDialog"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public LoadVideoDialog(IUIConfiguration configuration, IVideoService videoService)
        {
            _configuration = configuration;
            _videoService = videoService;
            SaveCommand = new AsyncRelayCommand(SaveAsync, CanExecuteSave);
            CancelCommand = new AsyncRelayCommand(CancelAsync, CanExecuteCancel);
            InitializeComponent();
        }

        public VideoInput Result => _result;
        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CancelCommand { get; }

        public string SourceVideo
        {
            get { return _sourceVideo; }
            set { _sourceVideo = value; NotifyPropertyChanged(); }
        }

        public VideoInfo SourceInfo
        {
            get { return _sourceInfo; }
            set { _sourceInfo = value; NotifyPropertyChanged(); }
        }

        public int CropWidth
        {
            get { return _cropWidth; }
            set { _cropWidth = value; NotifyPropertyChanged(); }
        }

        public int CropHeight
        {
            get { return _cropHeight; }
            set { _cropHeight = value; NotifyPropertyChanged(); }
        }

        public bool IsFixedSizeEnabled
        {
            get { return _isFixedSizeEnabled; }
            set { _isFixedSizeEnabled = value; NotifyPropertyChanged(); }
        }

        public int SelectedWidth
        {
            get { return _selectedWidth; }
            set
            {
                if (IsFixedSizeEnabled)
                {
                    _selectedWidth = _cropWidth;
                    NotifyPropertyChanged();
                }
                else
                {
                    if (_selectedResizeMode == TensorStack.Common.ResizeMode.Crop)
                    {
                        var oldValue = _selectedWidth == 0 ? value : _selectedWidth;
                        _selectedHeight = (_selectedHeight * value) / oldValue;
                        CropHeight = _selectedHeight;
                    }
                    _selectedWidth = value;
                    CropWidth = _selectedWidth;
                    NotifyPropertyChanged();
                    NotifyPropertyChanged(nameof(SelectedHeight));
                }
            }
        }

        public int SelectedHeight
        {
            get { return _selectedHeight; }
            set
            {
                if (IsFixedSizeEnabled)
                {
                    _selectedHeight = _cropHeight;
                    NotifyPropertyChanged();
                }
                else
                {
                    if (_selectedResizeMode == TensorStack.Common.ResizeMode.Crop)
                    {
                        var oldValue = _selectedHeight == 0 ? value : _selectedHeight;
                        _selectedWidth = (_selectedWidth * value) / oldValue;
                        CropWidth = _selectedWidth;
                    }
                    _selectedHeight = value;
                    CropHeight = _selectedHeight;
                    NotifyPropertyChanged();
                    NotifyPropertyChanged(nameof(SelectedWidth));
                }
            }
        }

        public float SelectedFrameRate
        {
            get { return _selectedFrameRate; }
            set
            {
                _selectedFrameRate = Math.Max(0.001f, Math.Min(value, SourceInfo.FrameRate));
                _selectedFrameCount = (int)(_selectedFrameRate * SourceInfo.Duration.TotalSeconds);
                NotifyPropertyChanged();
                NotifyPropertyChanged(nameof(SelectedFrameCount));
            }
        }

        public int SelectedFrameCount
        {
            get { return _selectedFrameCount; }
            set
            {
                _selectedFrameCount = Math.Max(1, Math.Min(value, SourceInfo.FrameCount));
                _selectedFrameRate = _selectedFrameCount / (float)SourceInfo.Duration.TotalSeconds;
                NotifyPropertyChanged();
                NotifyPropertyChanged(nameof(SelectedFrameRate));
            }
        }

        public ResizeMode SelectedResizeMode
        {
            get { return _selectedResizeMode; }
            set
            {
                _selectedResizeMode = value;
                NotifyPropertyChanged();
                if (!IsFixedSizeEnabled)
                {
                    CropWidth = _sourceInfo.Width;
                    CropHeight = _sourceInfo.Height;
                    _selectedWidth = _sourceInfo.Width;
                    _selectedHeight = _sourceInfo.Height;
                    NotifyPropertyChanged(nameof(SelectedWidth));
                    NotifyPropertyChanged(nameof(SelectedHeight));
                }
            }
        }

        public bool IsGenerating
        {
            get { return _isGenerating; }
            set { _isGenerating = value; NotifyPropertyChanged(); }
        }


        /// <summary>
        /// Show LoadVideoDialog
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public async Task<bool> ShowDialogAsync(int cropWidth, int cropHeight, string filename = default)
        {
            filename = filename ?? await DialogService.OpenFileAsync("Open Video", filter: "Videos|*.mp4;*.gif;|All Files|*.*;");
            if (string.IsNullOrEmpty(filename))
                return false;

            SourceVideo = filename;
            SourceInfo = await _videoService.GetVideoInfoAsync(filename);
            IsFixedSizeEnabled = cropWidth > 0 && cropHeight > 0;

            if (IsFixedSizeEnabled)
            {
                CropWidth = cropWidth;
                CropHeight = cropHeight;
                SelectedWidth = cropWidth;
                SelectedHeight = cropHeight;
                SelectedFrameRate = SourceInfo.FrameRate;
                SelectedFrameCount = SourceInfo.FrameCount;
            }
            else
            {
                CropWidth = SourceInfo.Width;
                CropHeight = SourceInfo.Height;
                SelectedWidth = SourceInfo.Width;
                SelectedHeight = SourceInfo.Height;
                SelectedFrameRate = SourceInfo.FrameRate;
                SelectedFrameCount = SourceInfo.FrameCount;
            }
            return await ShowDialogAsync();
        }


        /// <summary>
        /// Saves the cropped image.
        /// </summary>
        protected override async Task SaveAsync()
        {
            try
            {
                IsGenerating = true;
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var tempFilename = FileHelper.RandomFileName(_configuration.DirectoryTemp, "mp4");
                    var videoTensor = await VideoManager.LoadVideoTensorAsync(SourceVideo, CropWidth, CropHeight, SelectedFrameRate, SelectedResizeMode, _cancellationTokenSource.Token);
                    _result = new VideoInput(tempFilename, videoTensor);
                    await _result.SaveAsync(tempFilename);
                    await base.SaveAsync();
                }
            }
            catch (OperationCanceledException)
            {
                // Canceled
            }
            catch (Exception ex)
            {
                await DialogService.ShowErrorAsync("Generate Error", ex.Message);
            }
            IsGenerating = false;
        }


        /// <summary>
        /// Determines whether this instance can execute Done.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if this instance can execute Done; otherwise, <c>false</c>.
        /// </returns>
        protected override bool CanExecuteSave()
        {
            return true;
        }


        /// <summary>
        /// Cancel as an asynchronous operation.
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        protected override async Task CancelAsync()
        {
            if (_cancellationTokenSource is not null && !_cancellationTokenSource.IsCancellationRequested)
            {
                try
                {
                    await _cancellationTokenSource?.CancelAsync();
                    return;
                }
                catch (ObjectDisposedException) { }
            }
            await base.CancelAsync();
        }


        /// <summary>
        /// Close as an asynchronous operation.
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        protected override async Task CloseAsync()
        {
            await CancelAsync();
            await base.CloseAsync();
        }


        /// <summary>
        /// Handles the Loaded event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void VideoControl_Loaded(object sender, System.Windows.RoutedEventArgs e)
        {
            VideoControl.LoadedBehavior = MediaState.Play;
        }


        /// <summary>
        /// Handles the MediaOpened event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void VideoControl_MediaOpened(object sender, System.Windows.RoutedEventArgs e)
        {
            if (VideoControl.HasVideo)
            {
                //_progressTimer.Start();
                //ProgressMax = Source.Duration.TotalMilliseconds;
            }
        }


        /// <summary>
        /// Handles the MediaEnded event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void VideoControl_MediaEnded(object sender, System.Windows.RoutedEventArgs e)
        {
            VideoControl.LoadedBehavior = MediaState.Stop;
            VideoControl.Position = TimeSpan.FromMilliseconds(1);
            VideoControl.LoadedBehavior = MediaState.Play;
        }


        /// <summary>
        /// Handles the MouseDown event of the VideoControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private void VideoControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            VideoControl.LoadedBehavior = VideoControl.LoadedBehavior == MediaState.Pause || VideoControl.LoadedBehavior == MediaState.Stop
                 ? MediaState.Play
                 : MediaState.Pause;
        }
    }

    public readonly record struct LoadVideoResult(string FileName, int Width, int Height, float FrameRate, ResizeMode ResizeMode);
}
