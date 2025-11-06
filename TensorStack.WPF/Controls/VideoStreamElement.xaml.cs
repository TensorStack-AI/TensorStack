
using System;
using System.Collections.Specialized;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using TensorStack.Common;
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
        private string _fileSource;
        private string _fileOverlaySource;
        private MediaState _mediaState;
        private TimeSpan _progressPosition;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoStreamElement"/> class.
        /// </summary>
        public VideoStreamElement()
        {
            _progressTimer = new DispatcherTimer(TimeSpan.FromMilliseconds(50), DispatcherPriority.Normal, UpdateProgress, Dispatcher);
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
            InitializeComponent();
        }

        public static readonly DependencyProperty ConfigurationProperty = DependencyProperty.Register(nameof(Configuration), typeof(IUIConfiguration), typeof(VideoStreamElement));
        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(Source), typeof(VideoInputStream), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty OverlaySourceProperty = DependencyProperty.Register(nameof(OverlaySource), typeof(VideoInputStream), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterPositionProperty = DependencyProperty.Register(nameof(SplitterPosition), typeof(SplitterPosition), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterVisibilityProperty = DependencyProperty.Register(nameof(SplitterVisibility), typeof(SplitterVisibility), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterDirectionProperty = DependencyProperty.Register(nameof(SplitterDirection), typeof(SplitterDirection), typeof(VideoStreamElement), new PropertyMetadata<VideoStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty IsToolbarEnabledProperty = DependencyProperty.Register(nameof(IsToolbarEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadEnabledProperty = DependencyProperty.Register(nameof(IsLoadEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsSaveEnabledProperty = DependencyProperty.Register(nameof(IsSaveEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadOverlayEnabledProperty = DependencyProperty.Register(nameof(IsLoadOverlayEnabled), typeof(bool), typeof(VideoStreamElement));
        public static readonly DependencyProperty IsSaveOverlayEnabledProperty = DependencyProperty.Register(nameof(IsSaveOverlayEnabled), typeof(bool), typeof(VideoStreamElement));
        public static readonly DependencyProperty IsReplayEnabledProperty = DependencyProperty.Register(nameof(IsReplayEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsAutoPlayEnabledProperty = DependencyProperty.Register(nameof(IsAutoPlayEnabled), typeof(bool), typeof(VideoStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress), typeof(ProgressInfo), typeof(VideoStreamElement), new PropertyMetadata(new ProgressInfo()));
        public static readonly DependencyProperty PlaceholderProperty = DependencyProperty.Register(nameof(Placeholder), typeof(BitmapSource), typeof(VideoStreamElement));

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

        public bool IsToolbarEnabled
        {
            get { return (bool)GetValue(IsToolbarEnabledProperty); }
            set { SetValue(IsToolbarEnabledProperty, value); }
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

        public AsyncRelayCommand ClearCommand { get; }
        public AsyncRelayCommand LoadSourceCommand { get; }
        public AsyncRelayCommand SaveSourceCommand { get; }
        public AsyncRelayCommand CopySourceCommand { get; }
        public AsyncRelayCommand PasteSourceCommand { get; }
        public AsyncRelayCommand LoadOverlayCommand { get; }
        public AsyncRelayCommand SaveOverlayCommand { get; }
        public AsyncRelayCommand CopyOverlayCommand { get; }
        public AsyncRelayCommand PlayCommand { get; set; }
        public AsyncRelayCommand PauseCommand { get; set; }
        public AsyncRelayCommand StopCommand { get; set; }
        public bool HasSourceVideo => Source != null;
        public bool HasOverlayVideo => OverlaySource != null;

        public string FileSource
        {
            get { return _fileSource; }
            set { SetProperty(ref _fileSource, value); }
        }

        public string FileOverlaySource
        {
            get { return _fileOverlaySource; }
            set { SetProperty(ref _fileOverlaySource, value); }
        }

        public MediaState MediaState
        {
            get { return _mediaState; }
            set
            {
                _mediaState = value;
                VideoControl.LoadedBehavior = _mediaState;
                if (HasOverlayVideo)
                    VideoOverlayControl.LoadedBehavior = _mediaState;
                SetProperty(ref _mediaState, value);
            }
        }

        public TimeSpan ProgressPosition
        {
            get { return _progressPosition; }
            set { SetProperty(ref _progressPosition, value); }
        }


        /// <summary>
        /// Called when DependencyProperty changeded.
        /// </summary>
        private Task OnValueChanged()
        {
            FileSource = default;
            FileOverlaySource = default;
            GridSplitterContainer.Visibility = Visibility.Hidden;
            if (HasSourceVideo)
            {
                FileSource = Source.SourceFile;
            }

            if (HasOverlayVideo)
            {
                FileOverlaySource = OverlaySource.SourceFile;

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
            return Task.CompletedTask;
        }


        /// <summary>
        /// Clears thes control
        /// </summary>
        public Task ClearAsync()
        {
            MediaState = MediaState.Close;
            FileSource = null;
            FileOverlaySource = null;
            Source = null;
            OverlaySource = null;
            GridSplitterContainer.Visibility = Visibility.Hidden;
            Progress.Clear();
            ProgressPosition = TimeSpan.Zero;
            _progressTimer.Stop();
            MediaState = IsAutoPlayEnabled
                ? MediaState.Play
                : MediaState.Stop;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can clear.
        /// </summary>
        /// <returns><c>true</c> if this instance can clear; otherwise, <c>false</c>.</returns>
        private bool CanClear()
        {
            return HasSourceVideo || HasOverlayVideo;
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
            return IsLoadEnabled;
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
            return IsLoadOverlayEnabled;
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
            return IsLoadEnabled;
        }


        /// <summary>
        /// Plays the Video.
        /// </summary>
        private Task PlayAsync()
        {
            if (MediaState == MediaState.Close || MediaState == MediaState.Play)
                return Task.CompletedTask;

            MediaState = MediaState.Play;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Pauses the Video.
        /// </summary>
        private Task PauseAsync()
        {
            if (MediaState != MediaState.Play)
                return Task.CompletedTask;

            MediaState = MediaState.Pause;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Stops the Video.
        /// </summary>
        private Task StopAsync()
        {
            if (MediaState == MediaState.Close || MediaState == MediaState.Stop)
                return Task.CompletedTask;

            Progress.Clear();
            MediaState = MediaState.Stop;
            ProgressPosition = TimeSpan.Zero;
            VideoControl.Position = TimeSpan.FromMilliseconds(1);
            VideoOverlayControl.Position = TimeSpan.FromMilliseconds(1);

            return Task.CompletedTask;
        }


        /// <summary>
        /// Load video as an VideoInput from file
        /// </summary>
        /// <param name="initialFilename">The initial filename.</param>
        /// <returns>VideoInput</returns>
        private async Task<VideoInputStream> LoadVideoAsync(string initialFilename = null)
        {
            var videoFilename = initialFilename ?? await DialogService.OpenFileAsync("Open Video", filter: "Videos|*.mp4;*.gif;|All Files|*.*;");
            if (string.IsNullOrEmpty(videoFilename))
                return default;

            return await VideoInputStream.CreateAsync(videoFilename);
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
            if (HasOverlayVideo && SplitterVisibility != SplitterVisibility.Manual)
            {
                GridSplitterContainer.Visibility = Visibility.Visible;
            }

            if (!IsKeyboardFocusWithin)
                Keyboard.Focus(this);

            base.OnMouseEnter(e);
        }


        /// <summary>
        /// MouseLeave
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseLeave(MouseEventArgs e)
        {
            if (HasOverlayVideo && SplitterVisibility != SplitterVisibility.Manual)
            {
                GridSplitterContainer.Visibility = Visibility.Hidden;
            }
            base.OnMouseLeave(e);
        }


        /// <summary>
        /// MouseLeftButtonDown
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseButtonEventArgs" /> that contains the event data. The event data reports that the left mouse button was pressed.</param>
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            if (e.OriginalSource is MediaElement videoElement)
            {
                if (Source is VideoInputStream videoStream)
                {
                    DragDropHelper.DoDragDropFile(this, videoStream.SourceFile, DragDropType.Video, VideoControl, 4);
                }
            }
            base.OnMouseLeftButtonDown(e);
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
        /// On Drop
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.DragEventArgs" /> that contains the event data.</param>
        protected override async void OnDrop(DragEventArgs e)
        {
            if (!IsLoadEnabled)
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
        /// Handles the Loaded event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void VideoControl_Loaded(object sender, RoutedEventArgs e)
        {
            if (IsAutoPlayEnabled)
            {
                MediaState = MediaState.Play;
            }
        }


        /// <summary>
        /// Handles the MediaOpened event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void VideoControl_MediaOpened(object sender, RoutedEventArgs e)
        {
            if (HasSourceVideo)
            {
                _progressTimer.Start();
            }
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
            {
                MediaState = MediaState.Play;
            }
        }


        /// <summary>
        /// Handles the MouseDown event of the VideoControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private void VideoControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            MediaState = MediaState == MediaState.Pause || MediaState == MediaState.Stop
                 ? MediaState.Play
                 : MediaState.Pause;
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
            }
        }

    }
}
