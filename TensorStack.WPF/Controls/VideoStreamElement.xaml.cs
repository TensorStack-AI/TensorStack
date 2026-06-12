
using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using TensorStack.Common;
using TensorStack.Common.Video;
using TensorStack.Image;
using TensorStack.Video;
using TensorStack.WPF.Services;
using TensorStack.WPF.Utils;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for VideoStreamElement.xaml
    /// </summary>
    public partial class VideoStreamElement : BaseControl
    {
        private readonly DispatcherTimer _progressTimer;
        private CancellationTokenSource _cancellationTokenSource;
        private MediaState _mediaState = MediaState.Stop;
        private TimeSpan _progressPosition;
        private bool _isMediaControlsEnabled = true;
        private bool _isContextMenuEnabled = true;
        private bool _isToolbarEnabled = true;
        private Point _videoDragStart;
        private bool _isVideoDragging;
        private bool _isVideoFrameView;
        private bool _isFrameViewerEnabled;
        private VideoFrame _selectedVideoFrame;
        private bool _isLoadingVideoFrames;
        private Point _videoFrameDragStart;
        private bool _isVideoFrameDragging;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoStreamElement"/> class.
        /// </summary>
        public VideoStreamElement()
        {
            _progressTimer = new DispatcherTimer(TimeSpan.FromMilliseconds(50), DispatcherPriority.Normal, UpdateProgress, Dispatcher);
            VideoFrames = new ObservableCollection<VideoFrame>();
            ClearCommand = new AsyncRelayCommand(ClearAsync, CanClear);
            PlayCommand = new AsyncRelayCommand(PlayAsync, CanSaveSource);
            PauseCommand = new AsyncRelayCommand(PauseAsync, CanSaveSource);
            StopCommand = new AsyncRelayCommand(StopAsync, CanSaveSource);
            SaveSourceCommand = new AsyncRelayCommand(SaveSourceAsync, CanSaveSource);
            SaveOverlayCommand = new AsyncRelayCommand(SaveOverlayAsync, CanSaveOverlay);
            LoadSourceCommand = new AsyncRelayCommand(LoadSourceAsync, CanLoadSource);
            LoadOverlayCommand = new AsyncRelayCommand(LoadOverlayAsync, CanLoadOverlay);
            CopySourceCommand = new AsyncRelayCommand(CopySourceAsync, CanCopySource);
            CopyOverlayCommand = new AsyncRelayCommand(CopyOverlayAsync, CanCopyOverlay);
            PasteSourceCommand = new AsyncRelayCommand(PasteSourceAsync, CanPasteSource);
            ChangeViewCommand = new AsyncRelayCommand<bool>(ChangeViewAsync, CanChangeView);
            SaveVideoFrameCommand = new AsyncRelayCommand<int>(SaveVideoFrameAsync);
            CopyVideoFrameCommand = new AsyncRelayCommand<int>(CopyVideoFrameAsync);
            VideoProgress = new ProgressInfo();
            InitializeComponent();
        }

        public static readonly DependencyProperty ConfigurationProperty = DependencyProperty.Register(nameof(Configuration), typeof(IUIConfiguration), typeof(VideoStreamElement));
        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(Source), typeof(VideoInputStream), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty OverlaySourceProperty = DependencyProperty.Register(nameof(OverlaySource), typeof(VideoInputStream), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterPositionProperty = DependencyProperty.Register(nameof(SplitterPosition), typeof(SplitterPosition), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterVisibilityProperty = DependencyProperty.Register(nameof(SplitterVisibility), typeof(SplitterVisibility), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterDirectionProperty = DependencyProperty.Register(nameof(SplitterDirection), typeof(SplitterDirection), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty IsLoadEnabledProperty = DependencyProperty.Register(nameof(IsLoadEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsSaveEnabledProperty = DependencyProperty.Register(nameof(IsSaveEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsRemoveEnabledProperty = DependencyProperty.Register(nameof(IsRemoveEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadOverlayEnabledProperty = DependencyProperty.Register(nameof(IsLoadOverlayEnabled), typeof(bool), typeof(VideoStreamElement));
        public static readonly DependencyProperty IsSaveOverlayEnabledProperty = DependencyProperty.Register(nameof(IsSaveOverlayEnabled), typeof(bool), typeof(VideoStreamElement));
        public static readonly DependencyProperty IsReplayEnabledProperty = DependencyProperty.Register(nameof(IsReplayEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsAutoPlayEnabledProperty = DependencyProperty.Register(nameof(IsAutoPlayEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress), typeof(ProgressInfo), typeof(VideoStreamElement), new PropertyMetadata(new ProgressInfo()));
        public static readonly DependencyProperty PlaceholderProperty = DependencyProperty.Register(nameof(Placeholder), typeof(BitmapSource), typeof(VideoStreamElement));
        public static readonly DependencyProperty VolumeProperty = DependencyProperty.Register(nameof(Volume), typeof(double), typeof(VideoStreamElement), new PropertyMetadata(0.10));
        public static readonly DependencyProperty IsVolumeMuteProperty = DependencyProperty.Register(nameof(IsVolumeMute), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(false));
        public static readonly DependencyProperty ImageFrameProperty = DependencyProperty.Register(nameof(ImageFrame), typeof(BitmapSource), typeof(VideoStreamElement));
        public static readonly DependencyProperty IsInputEnabledProperty = DependencyProperty.Register(nameof(IsInputEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public event EventHandler<VideoInputStream> OnSourceChanged;
        public event EventHandler<MediaImportEventArgs> OnMediaImport;

        public AsyncRelayCommand ClearCommand { get; }
        public AsyncRelayCommand LoadSourceCommand { get; }
        public AsyncRelayCommand SaveSourceCommand { get; }
        public AsyncRelayCommand CopySourceCommand { get; }
        public AsyncRelayCommand PasteSourceCommand { get; }
        public AsyncRelayCommand<bool> ChangeViewCommand { get; }
        public AsyncRelayCommand<int> SaveVideoFrameCommand { get; }
        public AsyncRelayCommand<int> CopyVideoFrameCommand { get; }
        public AsyncRelayCommand LoadOverlayCommand { get; }
        public AsyncRelayCommand SaveOverlayCommand { get; }
        public AsyncRelayCommand CopyOverlayCommand { get; }
        public AsyncRelayCommand PlayCommand { get; set; }
        public AsyncRelayCommand PauseCommand { get; set; }
        public AsyncRelayCommand StopCommand { get; set; }
        public ObservableCollection<VideoFrame> VideoFrames { get; }
        public ProgressInfo VideoProgress { get; }
        public bool HasSourceVideo => Source != null;
        public bool HasOverlayVideo => OverlaySource != null;

        public IUIConfiguration Configuration
        {
            get { return (IUIConfiguration)GetValue(ConfigurationProperty); }
            set { SetValue(ConfigurationProperty, value); }
        }

        public VideoInputStream Source
        {
            get { return (VideoInputStream)GetValue(SourceProperty); }
            set { SetValue(SourceProperty, value); }
        }

        public VideoInputStream OverlaySource
        {
            get { return (VideoInputStream)GetValue(OverlaySourceProperty); }
            set { SetValue(OverlaySourceProperty, value); }
        }

        public SplitterPosition SplitterPosition
        {
            get { return (SplitterPosition)GetValue(SplitterPositionProperty); }
            set { SetValue(SplitterPositionProperty, value); }
        }

        public SplitterVisibility SplitterVisibility
        {
            get { return (SplitterVisibility)GetValue(SplitterVisibilityProperty); }
            set { SetValue(SplitterVisibilityProperty, value); }
        }

        public SplitterDirection SplitterDirection
        {
            get { return (SplitterDirection)GetValue(SplitterDirectionProperty); }
            set { SetValue(SplitterDirectionProperty, value); }
        }

        public bool IsLoadEnabled
        {
            get { return (bool)GetValue(IsLoadEnabledProperty); }
            set { SetValue(IsLoadEnabledProperty, value); }
        }

        public bool IsSaveEnabled
        {
            get { return (bool)GetValue(IsSaveEnabledProperty); }
            set { SetValue(IsSaveEnabledProperty, value); }
        }

        public bool IsRemoveEnabled
        {
            get { return (bool)GetValue(IsRemoveEnabledProperty); }
            set { SetValue(IsRemoveEnabledProperty, value); }
        }

        public bool IsLoadOverlayEnabled
        {
            get { return (bool)GetValue(IsLoadOverlayEnabledProperty); }
            set { SetValue(IsLoadOverlayEnabledProperty, value); }
        }

        public bool IsSaveOverlayEnabled
        {
            get { return (bool)GetValue(IsSaveOverlayEnabledProperty); }
            set { SetValue(IsSaveOverlayEnabledProperty, value); }
        }

        public bool IsReplayEnabled
        {
            get { return (bool)GetValue(IsReplayEnabledProperty); }
            set { SetValue(IsReplayEnabledProperty, value); }
        }

        public bool IsAutoPlayEnabled
        {
            get { return (bool)GetValue(IsAutoPlayEnabledProperty); }
            set { SetValue(IsAutoPlayEnabledProperty, value); }
        }

        public ProgressInfo Progress
        {
            get { return (ProgressInfo)GetValue(ProgressProperty); }
            set { SetValue(ProgressProperty, value); }
        }

        public BitmapSource Placeholder
        {
            get { return (BitmapSource)GetValue(PlaceholderProperty); }
            set { SetValue(PlaceholderProperty, value); }
        }

        public BitmapSource ImageFrame
        {
            get { return (BitmapSource)GetValue(ImageFrameProperty); }
            set { SetValue(ImageFrameProperty, value); }
        }

        public bool IsInputEnabled
        {
            get { return (bool)GetValue(IsInputEnabledProperty); }
            set { SetValue(IsInputEnabledProperty, value); }
        }

        public double Volume
        {
            get { return (double)GetValue(VolumeProperty); }
            set { SetValue(VolumeProperty, value); }
        }

        public bool IsVolumeMute
        {
            get { return (bool)GetValue(IsVolumeMuteProperty); }
            set { SetValue(IsVolumeMuteProperty, value); }
        }

        public MediaState MediaState
        {
            get { return _mediaState; }
            set { SetProperty(ref _mediaState, value); }
        }

        public TimeSpan ProgressPosition
        {
            get { return _progressPosition; }
            set { SetProperty(ref _progressPosition, value); }
        }

        public bool IsVideoFrameView
        {
            get { return _isVideoFrameView; }
            set { SetProperty(ref _isVideoFrameView, value); }
        }

        public VideoFrame SelectedVideoFrame
        {
            get { return _selectedVideoFrame; }
            set { SetProperty(ref _selectedVideoFrame, value); }
        }

        public bool IsLoadingVideoFrames
        {
            get { return _isLoadingVideoFrames; }
            set { SetProperty(ref _isLoadingVideoFrames, value); }
        }

        public bool IsFrameViewerEnabled
        {
            get { return _isFrameViewerEnabled; }
            set { SetProperty(ref _isFrameViewerEnabled, value); }
        }

        public bool IsToolbarEnabled
        {
            get { return _isToolbarEnabled; }
            set { SetProperty(ref _isToolbarEnabled, value); }
        }

        public bool IsMediaControlsEnabled
        {
            get { return _isMediaControlsEnabled; }
            set { SetProperty(ref _isMediaControlsEnabled, value); }
        }

        public bool IsContextMenuEnabled
        {
            get { return _isContextMenuEnabled; }
            set { SetProperty(ref _isContextMenuEnabled, value); }
        }


        /// <summary>
        /// Called when DependencyProperty changeded.
        /// </summary>
        private async Task OnValueChanged()
        {
            await StopAsync();

            VideoControl.Source = default;
            VideoOverlayControl.Source = default;
            GridSplitterContainer.Visibility = Visibility.Hidden;

            if (HasSourceVideo)
            {
                await ClearVideoFrames();
                VideoControl.Source = new Uri(Source.SourceFile);
                OnSourceChanged?.Invoke(this, Source);
                await LoadVideoFramesAsync();
            }

            if (HasOverlayVideo)
            {
                VideoOverlayControl.Source = new Uri(OverlaySource.SourceFile);

                AutoHideSplitter();
                if (SplitterPosition == SplitterPosition.Source)
                {
                    GridSplitterColumn.Width = SplitterDirection == SplitterDirection.LeftToRight
                        ? new GridLength(0)
                        : new GridLength(GridSplitterContainer.ActualWidth);
                }
                else if (SplitterPosition == SplitterPosition.Center)
                {
                    GridSplitterColumn.Width = new GridLength(0);
                    GridSplitterColumn.Width = new GridLength(GridSplitterContainer.ActualWidth / 2);
                }
                else if (SplitterPosition == SplitterPosition.Overlay)
                {
                    GridSplitterColumn.Width = SplitterDirection == SplitterDirection.RightToLeft
                        ? new GridLength(0)
                        : new GridLength(GridSplitterContainer.ActualWidth);
                }
            }

            ChangeViewCommand.RaiseCanExecuteChanged();
            if (IsAutoPlayEnabled)
            {
                await PlayAsync();
            }
        }


        /// <summary>
        /// Clears thes control
        /// </summary>
        public async Task ClearAsync()
        {
            await CloseAsync();
            await ClearVideoFrames();

            Source = null;
            OverlaySource = null;
            IsVideoFrameView = false;
            GridSplitterContainer.Visibility = Visibility.Hidden;
        }


        /// <summary>
        /// Determines whether this instance can clear.
        /// </summary>
        /// <returns><c>true</c> if this instance can clear; otherwise, <c>false</c>.</returns>
        private bool CanClear()
        {
            return (HasSourceVideo || HasOverlayVideo) && IsRemoveEnabled;
        }


        /// <summary>
        /// Load source
        /// </summary>
        private async Task LoadSourceAsync()
        {
            var source = await LoadVideoAsync();
            if (source != null)
                Source = source;
        }


        /// <summary>
        /// Determines whether this instance can load source.
        /// </summary>
        /// <returns><c>true</c> if this instance can load source; otherwise, <c>false</c>.</returns>
        private bool CanLoadSource()
        {
            return IsLoadEnabled && IsInputEnabled;
        }


        /// <summary>
        /// Load overlay
        /// </summary>
        private async Task LoadOverlayAsync()
        {
            var source = await LoadVideoAsync();
            if (source != null)
                OverlaySource = source;
        }


        /// <summary>
        /// Determines whether this instance can load overlay.
        /// </summary>
        /// <returns><c>true</c> if this instance can load overlay; otherwise, <c>false</c>.</returns>
        private bool CanLoadOverlay()
        {
            return IsLoadOverlayEnabled && IsInputEnabled;
        }


        /// <summary>
        /// Saves the source
        /// </summary>
        private async Task SaveSourceAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Video", "Video", filter: "mp4 files (*.mp4)|*.mp4", defualtExt: "mp4");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                File.Copy(Source.SourceFile, saveFilename, true);
            }
        }


        /// <summary>
        /// Determines whether this instance can save source.
        /// </summary>
        /// <returns><c>true</c> if this instance can save source; otherwise, <c>false</c>.</returns>
        private bool CanSaveSource()
        {
            return HasSourceVideo;
        }


        /// <summary>
        /// Save the overlay
        /// </summary>
        private async Task SaveOverlayAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Video", "Overlay", filter: "mp4 files (*.mp4)|*.mp4", defualtExt: "mp4");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                File.Copy(OverlaySource.SourceFile, saveFilename, true);
            }
        }


        /// <summary>
        /// Determines whether this instance can save overlay.
        /// </summary>
        /// <returns><c>true</c> if this instance can save overlay; otherwise, <c>false</c>.</returns>
        private bool CanSaveOverlay()
        {
            return HasOverlayVideo;
        }


        /// <summary>
        /// Copies the source.
        /// </summary>
        private Task CopySourceAsync()
        {
            Clipboard.SetFileDropList(new StringCollection
            {
                Source.SourceFile
            });
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can copy source.
        /// </summary>
        /// <returns><c>true</c> if this instance can copy source; otherwise, <c>false</c>.</returns>
        private bool CanCopySource()
        {
            return HasSourceVideo;
        }


        /// <summary>
        /// Copies the overlay.
        /// </summary>
        private Task CopyOverlayAsync()
        {
            Clipboard.SetFileDropList(new StringCollection
            {
                OverlaySource.SourceFile
            });
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can copy overlay.
        /// </summary>
        /// <returns><c>true</c> if this instance can copy overlay; otherwise, <c>false</c>.</returns>
        private bool CanCopyOverlay()
        {
            return HasOverlayVideo;
        }


        /// <summary>
        /// Paste source
        /// </summary>
        private async Task PasteSourceAsync()
        {
            if (!IsLoadEnabled)
                return;

            if (Clipboard.ContainsFileDropList())
            {
                var sourceFilename = Clipboard.GetFileDropList()
                    .OfType<string>()
                    .FirstOrDefault();
                var source = await LoadVideoAsync(sourceFilename);
                if (source != null)
                    Source = source;
            }
        }


        /// <summary>
        /// Determines whether this instance can paste source.
        /// </summary>
        /// <returns><c>true</c> if this instance can paste source; otherwise, <c>false</c>.</returns>
        private bool CanPasteSource()
        {
            return IsLoadEnabled && IsInputEnabled;
        }


        /// <summary>
        /// On Drop
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.DragEventArgs" /> that contains the event data.</param>
        protected override async void OnDrop(DragEventArgs e)
        {
            if (!IsLoadEnabled || !IsInputEnabled)
                return;

            var fileNames = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (!fileNames.IsNullOrEmpty())
            {
                var source = await LoadVideoAsync(fileNames.FirstOrDefault());
                if (source != null)
                    Source = source;
            }

            base.OnDrop(e);
        }


        /// <summary>
        /// Plays the Video.
        /// </summary>
        private Task PlayAsync()
        {
            _progressTimer.Start();
            VideoControl.Play();
            VideoOverlayControl.Play();
            MediaState = MediaState.Play;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Pauses the Video.
        /// </summary>
        private Task PauseAsync()
        {
            VideoControl.Pause();
            VideoOverlayControl.Pause();
            MediaState = MediaState.Pause;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Stops the Video.
        /// </summary>
        private async Task StopAsync()
        {
            _progressTimer.Stop();
            VideoControl.Stop();
            VideoOverlayControl.Stop();
            MediaState = MediaState.Stop;
            Progress.Clear();
            VideoProgress.Clear();

            // Bugfix: Set position to non-zero to keep GIFs looping
            VideoControl.Position = TimeSpan.FromMilliseconds(1);
            VideoOverlayControl.Position = TimeSpan.FromMilliseconds(1);

            // Bugfix: Pause to apply GPU queue or MediaElements may freeze globally
            await Task.Delay(50);
        }


        /// <summary>
        /// Close the Video.
        /// </summary>
        private async Task CloseAsync()
        {
            await StopAsync();
            VideoControl.Close();
            VideoOverlayControl.Close();
            VideoControl.Source = null;
            VideoOverlayControl.Source = null;
        }


        /// <summary>
        /// Load video as an VideoInput from file
        /// </summary>
        /// <param name="initialFilename">The initial filename.</param>
        /// <returns>VideoInput</returns>
        private async Task<VideoInputStream> LoadVideoAsync(string initialFilename = null)
        {
            var sourceFilename = initialFilename ?? await DialogService.OpenFileAsync("Open Video", filter: "Videos|*.mp4;*.gif;|All Files|*.*;");
            if (string.IsNullOrEmpty(sourceFilename))
                return default;

            var videoInput = await VideoInputStream.CreateAsync(sourceFilename);
            OnMediaImport?.Invoke(this, new MediaImportEventArgs(sourceFilename, videoInput));
            return videoInput;
        }


        /// <summary>
        /// Handles the MediaEnded event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private async void VideoControl_MediaEnded(object sender, RoutedEventArgs e)
        {
            await StopAsync();
            if (IsReplayEnabled)
                await PlayAsync();
        }


        /// <summary>
        /// Handles the MouseDown event of the VideoControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private async void VideoControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (MediaState == MediaState.Play)
                await PauseAsync();
            else if (MediaState == MediaState.Pause)
                await PlayAsync();
        }


        /// <summary>
        /// Update progress
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="EventArgs"/> instance containing the event data.</param>
        private void UpdateProgress(object sender, EventArgs e)
        {
            if (VideoControl.HasVideo)
            {
                ProgressPosition = VideoControl.Position;
                var duration = VideoControl.NaturalDuration.HasTimeSpan ? VideoControl.NaturalDuration.TimeSpan : Source?.Duration ?? TimeSpan.Zero;
                VideoProgress.Update((int)(ProgressPosition.TotalMilliseconds / 10), (int)(duration.TotalMilliseconds / 10));
            }
        }


        /// <summary>
        /// Change Video/VideoFrame view.
        /// </summary>
        /// <param name="isChecked">if set to <c>true</c> show VideoFrame view, otherwise VideoPlayer view.</param>
        private async Task ChangeViewAsync(bool isChecked)
        {
            await LoadVideoFramesAsync();
        }


        /// <summary>
        /// Determines if Video/VideoFrame view can be changed
        /// </summary>
        /// <param name="isChecked">if set to <c>true</c> show VideoFrame view, otherwise VideoPlayer view.</param>
        private bool CanChangeView(bool isChecked)
        {
            return HasSourceVideo;
        }


        /// <summary>
        /// Load video frames.
        /// </summary>
        private async Task LoadVideoFramesAsync()
        {
            try
            {
                if (!_isVideoFrameView)
                    return;

                if (VideoFrames.Count > 0)
                    return;

                IsLoadingVideoFrames = true;
                var frameCount = Source.FrameCount;
                await _cancellationTokenSource.SafeCancelAsync();
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    var index = 0;
                    await foreach (var videoFrame in Source.GetAsync(heightOverride: 80, resizeMode: TensorStack.Common.ResizeMode.LetterBox, cancellationToken: _cancellationTokenSource.Token))
                    {
                        if (_cancellationTokenSource.IsCancellationRequested)
                            break;

                        VideoFrames.Add(videoFrame);
                        if (index % 10 == 0)
                            await Dispatcher.Yield();

                        index++;
                        Progress.Update(index, frameCount);
                    }
                }
            }
            catch (OperationCanceledException) { }
            finally
            {
                IsLoadingVideoFrames = false;
                Progress.Clear();
            }
        }


        /// <summary>
        /// Clears the video frames.
        /// </summary>
        private async Task ClearVideoFrames()
        {
            await _cancellationTokenSource.SafeCancelAsync();
            VideoFrames.Clear();
        }


        /// <summary>
        /// Get the video frame at the specified index.
        /// </summary>
        /// <param name="frameIndex">Index of the frame.</param>
        private async Task<VideoFrame> GetVideoFrameAsync(int frameIndex)
        {
            try
            {
                IsLoadingVideoFrames = true;
                if (Source == null)
                    return null;

                await _cancellationTokenSource.SafeCancelAsync();
                using (_cancellationTokenSource = new CancellationTokenSource())
                {
                    return await Source.GetFrameAsync(frameIndex);
                }
            }
            catch (Exception) { }
            finally
            {
                IsLoadingVideoFrames = false;
            }
            return null;
        }


        /// <summary>
        /// Save video frame at the specified frame index.
        /// </summary>
        /// <param name="frameIndex">Index of the frame.</param>
        private async Task SaveVideoFrameAsync(int frameIndex)
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Image", $"VideoFrame_{frameIndex:0000}", filter: "png files (*.png)|*.png", defualtExt: "png");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                var videoFrame = await GetVideoFrameAsync(frameIndex);
                if (videoFrame == null)
                    return;

                var inputImage = await videoFrame.Frame.ToImageInputAsync();
                await inputImage.SaveAsync(saveFilename);
            }
        }


        /// <summary>
        /// Copy video frame at the specified frame index.
        /// </summary>
        /// <param name="frameIndex">Index of the frame.</param>
        private async Task CopyVideoFrameAsync(int frameIndex)
        {
            var videoFrame = await GetVideoFrameAsync(frameIndex);
            if (videoFrame == null)
                return;

            Clipboard.SetImage(videoFrame.Frame.ToImage());
        }


        /// <summary>
        /// Auto hide splitter.
        /// </summary>
        private async void AutoHideSplitter()
        {
            GridSplitterContainer.Visibility = Visibility.Visible;
            if (SplitterVisibility == SplitterVisibility.Auto)
            {
                await Task.Delay(3000);
            }

            if (!IsMouseOver || SplitterVisibility == SplitterVisibility.Manual)
            {
                GridSplitterContainer.Visibility = Visibility.Hidden;
            }
        }


        /// <summary>
        /// Handles the PreviewMouseDown event of the SplitterControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private void SplitterControl_PreviewMouseDown(object sender, MouseButtonEventArgs e)
        {
            SplitterControl.CaptureMouse();
        }


        /// <summary>
        /// Handles the PreviewMouseUp event of the SplitterControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private void SplitterControl_PreviewMouseUp(object sender, MouseButtonEventArgs e)
        {
            SplitterControl.ReleaseMouseCapture();
        }


        /// <summary>
        /// GridSplitter SizeChanged event, Update Overlay Clip
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="SizeChangedEventArgs"/> instance containing the event data.</param>
        private void GridSplitter_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            VideoOverlayControl.Clip = SplitterDirection == SplitterDirection.LeftToRight
                ? new RectangleGeometry(new Rect(0, 0, e.NewSize.Width, e.NewSize.Height))
                : new RectangleGeometry(new Rect(e.NewSize.Width, 0, VideoOverlayControl.ActualWidth, VideoOverlayControl.ActualHeight));
        }


        /// <summary>
        /// MouseEnter
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseEnter(MouseEventArgs e)
        {
            base.OnMouseEnter(e);
            if (HasOverlayVideo && SplitterVisibility != SplitterVisibility.Manual)
                GridSplitterContainer.Visibility = Visibility.Visible;

            if (!IsKeyboardFocusWithin)
                Focus();
        }


        /// <summary>
        /// MouseLeave
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseLeave(MouseEventArgs e)
        {
            base.OnMouseLeave(e);
            if (HasOverlayVideo && SplitterVisibility != SplitterVisibility.Manual)
                GridSplitterContainer.Visibility = Visibility.Hidden;
        }


        /// <summary>
        /// MouseLeftButtonDown
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseButtonEventArgs" /> that contains the event data. The event data reports that the left mouse button was pressed.</param>
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            _videoDragStart = e.GetPosition(this);
            _isVideoDragging = false;
        }


        /// <summary>
        /// Invoked when an unhandled <see cref="E:System.Windows.Input.Mouse.MouseMove" /> attached event reaches an element in its route that is derived from this class. Implement this method to add class handling for this event.
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (e.LeftButton == MouseButtonState.Pressed && !_isVideoDragging && !IsVideoFrameView)
            {
                if (e.OriginalSource is not MediaElement)
                    return;

                var currentPosition = e.GetPosition(this);
                var diffPosition = _videoDragStart - currentPosition;
                if (Math.Abs(diffPosition.X) > Common.DragDistance || Math.Abs(diffPosition.Y) > Common.DragDistance)
                {
                    if (Source is VideoInputStream videoStream)
                    {
                        _isVideoDragging = true;
                        DragDropHelper.DoDragDropFile(this, videoStream.SourceFile, DragDropType.Video, VideoControl, 4);
                        _isVideoDragging = false;
                    }
                }
            }
        }


        /// <summary>
        /// Handles the MouseWheel event of the VideoFrameListBox control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseWheelEventArgs"/> instance containing the event data.</param>
        private void VideoFrameListBox_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            var scrollViewer = (ScrollViewer)sender;
            scrollViewer.ScrollToVerticalOffset(scrollViewer.VerticalOffset - e.Delta);
            e.Handled = true;
        }


        /// <summary>
        /// MouseLeftButtonDown
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseButtonEventArgs" /> that contains the event data. The event data reports that the left mouse button was pressed.</param>
        protected void VideoFrameListBox_MouseLeftButtonDown(object sender, MouseEventArgs e)
        {
            _isVideoFrameDragging = false;
            _videoFrameDragStart = e.GetPosition(this);
        }


        /// <summary>
        /// Invoked when an unhandled <see cref="E:System.Windows.Input.Mouse.MouseMove" /> attached event reaches an element in its route that is derived from this class. Implement this method to add class handling for this event.
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected async void VideoFrameListBox_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed && !_isVideoFrameDragging)
            {
                var currentPosition = e.GetPosition(this);
                var diffPosition = _videoFrameDragStart - currentPosition;
                if (Math.Abs(diffPosition.X) > Common.DragDistance || Math.Abs(diffPosition.Y) > Common.DragDistance)
                {
                    _isVideoFrameDragging = true;
                    if (_selectedVideoFrame is null)
                        return;

                    var listBoxItem = (ListBoxItem)VideoFrameListBox.ItemContainerGenerator.ContainerFromItem(_selectedVideoFrame);
                    if (listBoxItem == null)
                        return;

                    var videoFrame = await GetVideoFrameAsync(_selectedVideoFrame.Index);
                    if (videoFrame is null)
                        return;

                    DragDropHelper.DoDragDropObject(this, videoFrame.Frame.ToImage(), DragDropType.Image, listBoxItem);
                    _isVideoFrameDragging = false;
                }
            }
        }
    }
}
