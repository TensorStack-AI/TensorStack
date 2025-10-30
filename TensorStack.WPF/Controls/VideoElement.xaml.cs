using System;
using System.Collections.Specialized;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using TensorStack.Common;
using TensorStack.Common.Common;
using TensorStack.Video;
using TensorStack.WPF.Dialogs;
using TensorStack.WPF.Services;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for VideoElement.xaml
    /// </summary>
    public partial class VideoElement : BaseControl
    {
        private readonly DispatcherTimer _progressTimer;
        private string _fileSource;
        private string _fileOverlaySource;
        private MediaState _mediaState;
        private TimeSpan _progressPosition;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoElement"/> class.
        /// </summary>
        public VideoElement()
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

        public static readonly DependencyProperty ConfigurationProperty = DependencyProperty.Register(nameof(Configuration), typeof(IUIConfiguration), typeof(VideoElement));
        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(Source), typeof(VideoInput), typeof(VideoElement), new PropertyMetadata<VideoElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty OverlaySourceProperty = DependencyProperty.Register(nameof(OverlaySource), typeof(VideoInput), typeof(VideoElement), new PropertyMetadata<VideoElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterPositionProperty = DependencyProperty.Register(nameof(SplitterPosition), typeof(SplitterPosition), typeof(VideoElement), new PropertyMetadata<VideoElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterVisibilityProperty = DependencyProperty.Register(nameof(SplitterVisibility), typeof(SplitterVisibility), typeof(VideoElement), new PropertyMetadata<VideoElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty SplitterDirectionProperty = DependencyProperty.Register(nameof(SplitterDirection), typeof(SplitterDirection), typeof(VideoElement), new PropertyMetadata<VideoElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty CropWidthProperty = DependencyProperty.Register(nameof(CropWidth), typeof(int), typeof(VideoElement));
        public static readonly DependencyProperty CropHeightProperty = DependencyProperty.Register(nameof(CropHeight), typeof(int), typeof(VideoElement));
        public static readonly DependencyProperty IsToolbarEnabledProperty = DependencyProperty.Register(nameof(IsToolbarEnabled), typeof(bool), typeof(VideoElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadEnabledProperty = DependencyProperty.Register(nameof(IsLoadEnabled), typeof(bool), typeof(VideoElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsSaveEnabledProperty = DependencyProperty.Register(nameof(IsSaveEnabled), typeof(bool), typeof(VideoElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadOverlayEnabledProperty = DependencyProperty.Register(nameof(IsLoadOverlayEnabled), typeof(bool), typeof(VideoElement));
        public static readonly DependencyProperty IsSaveOverlayEnabledProperty = DependencyProperty.Register(nameof(IsSaveOverlayEnabled), typeof(bool), typeof(VideoElement));
        public static readonly DependencyProperty ProgressValueProperty = DependencyProperty.Register(nameof(ProgressValue), typeof(double), typeof(VideoElement), new PropertyMetadata(0d));
        public static readonly DependencyProperty ProgressMaxProperty = DependencyProperty.Register(nameof(ProgressMax), typeof(double), typeof(VideoElement), new PropertyMetadata(1d));
        public static readonly DependencyProperty IsReplayEnabledProperty = DependencyProperty.Register(nameof(IsReplayEnabled), typeof(bool), typeof(VideoElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsAutoPlayEnabledProperty = DependencyProperty.Register(nameof(IsAutoPlayEnabled), typeof(bool), typeof(VideoElement), new PropertyMetadata(true));
        public static readonly DependencyProperty PlaceholderProperty = DependencyProperty.Register(nameof(Placeholder), typeof(BitmapSource), typeof(VideoElement));

        public IUIConfiguration Configuration
        {
            get { return (IUIConfiguration)GetValue(ConfigurationProperty); }
            set { SetValue(ConfigurationProperty, value); }
        }

        public VideoInput Source
        {
            get { return (VideoInput)GetValue(SourceProperty); }
            set { SetValue(SourceProperty, value); }
        }

        public VideoInput OverlaySource
        {
            get { return (VideoInput)GetValue(OverlaySourceProperty); }
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

        public int CropWidth
        {
            get { return (int)GetValue(CropWidthProperty); }
            set { SetValue(CropWidthProperty, value); }
        }

        public int CropHeight
        {
            get { return (int)GetValue(CropHeightProperty); }
            set { SetValue(CropHeightProperty, value); }
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

        public double ProgressValue
        {
            get { return (double)GetValue(ProgressValueProperty); }
            set { SetValue(ProgressValueProperty, value); }
        }

        public double ProgressMax
        {
            get { return (double)GetValue(ProgressMaxProperty); }
            set { SetValue(ProgressMaxProperty, value); }
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
            set { _fileSource = value; NotifyPropertyChanged(); }
        }

        public string FileOverlaySource
        {
            get { return _fileOverlaySource; }
            set { _fileOverlaySource = value; NotifyPropertyChanged(); }
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
                NotifyPropertyChanged();
            }
        }

        public TimeSpan ProgressPosition
        {
            get { return _progressPosition; }
            set { _progressPosition = value; NotifyPropertyChanged(); }
        }


        /// <summary>
        /// Called when DependencyProperty changeded.
        /// </summary>
        private async Task OnValueChanged()
        {
            if (HasSourceVideo)
            {
                // If filename is null, create and save temp file
                FileSource = await GetOrCreateFileSource(Source);
            }

            if (HasOverlayVideo)
            {
                // If filename is null, create and save temp file
                FileOverlaySource = await GetOrCreateFileSource(OverlaySource);

                AutoHideSplitter();
                if (SplitterPosition == SplitterPosition.Source)
                {
                    GridSplitterColumn.Width = SplitterDirection == SplitterDirection.LeftToRight
                        ? new GridLength(0)
                        : new GridLength(OverlaySource.Width + 45);
                }
                else if (SplitterPosition == SplitterPosition.Center)
                {
                    GridSplitterColumn.Width = new GridLength(0);
                    GridSplitterColumn.Width = new GridLength(OverlaySource.Width / 2 + 30);
                }
                else if (SplitterPosition == SplitterPosition.Overlay)
                {
                    GridSplitterColumn.Width = SplitterDirection == SplitterDirection.RightToLeft
                        ? new GridLength(0)
                        : new GridLength(OverlaySource.Width + 45);
                }
            }
        }


        /// <summary>
        /// Clears the control
        /// </summary>
        private Task ClearAsync()
        {
            MediaState = MediaState.Close;
            FileSource = null;
            FileOverlaySource = null;
            Source = null;
            OverlaySource = null;
            GridSplitterContainer.Visibility = Visibility.Hidden;
            ProgressMax = 1;
            ProgressValue = 0;
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
            return true;
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
            return true;
        }


        /// <summary>
        /// Saves the source
        /// </summary>
        private async Task SaveSourceAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Video", "Video", filter: "mp4 files (*.mp4)|*.mp4", defualtExt: "mp4");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                await Source.SaveAsync(saveFilename);
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
                await OverlaySource.SaveAsync(saveFilename);
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
            Clipboard.SetFileDropList([Source.SourceFile]);
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

            ProgressValue = 0;
            ProgressPosition = TimeSpan.Zero;
            VideoControl.Position = TimeSpan.Zero;
            VideoOverlayControl.Position = VideoControl.Position;
            MediaState = MediaState.Stop;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Load video as an VideoInput from file
        /// </summary>
        /// <param name="initialFilename">The initial filename.</param>
        /// <returns>VideoInput</returns>
        private async Task<VideoInput> LoadVideoAsync(string initialFilename = null)
        {
            var loadDialog = DialogService.GetDialog<LoadVideoDialog>();
            if (await loadDialog.ShowDialogAsync(CropWidth, CropHeight, initialFilename))
            {
                return loadDialog.Result;
            }
            return null;
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
                AllowDrop = false;
                DragDrop.DoDragDrop(videoElement, new DataObject(typeof(Uri), videoElement.Source), DragDropEffects.Copy);
                AllowDrop = true;
            }
            base.OnMouseLeftButtonDown(e);
        }


        /// <summary>Handles the PreviewMouseDown event of the SplitterControl control.</summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs" /> instance containing the event data.</param>
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
                ProgressMax = Source.Duration.TotalMilliseconds;
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
                ProgressValue = ProgressPosition.TotalMilliseconds;
            }
        }


        /// <summary>
        /// Gets or create the temp video file.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <returns>Path of the temp video file.</returns>
        private async Task<string> GetOrCreateFileSource(VideoInput source)
        {
            if (!string.IsNullOrEmpty(source.SourceFile))
                return source.SourceFile;

            var tempFilename = FileHelper.RandomFileName(Configuration.DirectoryTemp, "mp4");
            await source.SaveAsync(tempFilename);
            return tempFilename;
        }
    }
}
