using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using TensorStack.Audio;
using TensorStack.Common;
using TensorStack.WPF.Services;
using TensorStack.WPF.Utils;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for AudioStreamElement.xaml
    /// </summary>
    public partial class AudioStreamElement : BaseControl
    {
        private readonly DispatcherTimer _progressTimer;
        private MediaState _mediaState = MediaState.Stop;
        private TimeSpan _progressPosition;
        private bool _isMediaControlsEnabled = true;
        private bool _isContextMenuEnabled = true;
        private bool _isToolbarEnabled = true;
        private Point _audioDragStart;
        private bool _isAudioDragging;

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioStreamElement"/> class.
        /// </summary>
        public AudioStreamElement()
        {
            _progressTimer = new DispatcherTimer(TimeSpan.FromMilliseconds(50), DispatcherPriority.Normal, UpdateProgress, Dispatcher);
            AudioProgress = new ProgressInfo();
            ClearCommand = new AsyncRelayCommand(ClearAsync, CanClear);
            PlayCommand = new AsyncRelayCommand(PlayAsync, CanSaveSource);
            PauseCommand = new AsyncRelayCommand(PauseAsync, CanSaveSource);
            StopCommand = new AsyncRelayCommand(StopAsync, CanSaveSource);
            SaveCommand = new AsyncRelayCommand(SaveAsync, CanSaveSource);
            LoadCommand = new AsyncRelayCommand(LoadSourceAsync, CanLoadSource);
            CopyCommand = new AsyncRelayCommand(CopyAsync, CanCopySource);
            PasteCommand = new AsyncRelayCommand(PasteAsync, CanPasteSource);
            InitializeComponent();
        }

        public static readonly DependencyProperty ConfigurationProperty = DependencyProperty.Register(nameof(Configuration), typeof(IUIConfiguration), typeof(AudioStreamElement));
        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(Source), typeof(AudioInputStream), typeof(AudioStreamElement), new PropertyMetadata<AudioStreamElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty IsReplayEnabledProperty = DependencyProperty.Register(nameof(IsReplayEnabled), typeof(bool), typeof(AudioStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsAutoPlayEnabledProperty = DependencyProperty.Register(nameof(IsAutoPlayEnabled), typeof(bool), typeof(AudioStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadEnabledProperty = DependencyProperty.Register(nameof(IsLoadEnabled), typeof(bool), typeof(AudioStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsSaveEnabledProperty = DependencyProperty.Register(nameof(IsSaveEnabled), typeof(bool), typeof(AudioStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsRemoveEnabledProperty = DependencyProperty.Register(nameof(IsRemoveEnabled), typeof(bool), typeof(AudioStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty VolumeProperty = DependencyProperty.Register(nameof(Volume), typeof(double), typeof(AudioStreamElement), new PropertyMetadata(0.10));
        public static readonly DependencyProperty IsVolumeMuteProperty = DependencyProperty.Register(nameof(IsVolumeMute), typeof(bool), typeof(AudioStreamElement), new PropertyMetadata(false));
        public static readonly DependencyProperty IsInputEnabledProperty = DependencyProperty.Register(nameof(IsInputEnabled), typeof(bool), typeof(AudioStreamElement), new PropertyMetadata(true));
        public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress), typeof(ProgressInfo), typeof(AudioStreamElement), new PropertyMetadata(new ProgressInfo()));
        public static readonly DependencyProperty PlaceholderProperty = DependencyProperty.Register(nameof(Placeholder), typeof(BitmapSource), typeof(AudioStreamElement));
        public event EventHandler<AudioInputStream> OnSourceChanged;
        public event EventHandler<MediaImportEventArgs> OnMediaImport;

        public AsyncRelayCommand ClearCommand { get; }
        public AsyncRelayCommand LoadCommand { get; }
        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CopyCommand { get; }
        public AsyncRelayCommand PasteCommand { get; }
        public AsyncRelayCommand PlayCommand { get; }
        public AsyncRelayCommand PauseCommand { get; }
        public AsyncRelayCommand StopCommand { get; }
        public ProgressInfo AudioProgress { get; }
        public bool HasAudio => Source != null;

        public IUIConfiguration Configuration
        {
            get { return (IUIConfiguration)GetValue(ConfigurationProperty); }
            set { SetValue(ConfigurationProperty, value); }
        }

        public AudioInputStream Source
        {
            get { return (AudioInputStream)GetValue(SourceProperty); }
            set { SetValue(SourceProperty, value); }
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
            AudioControl.Source = default;

            if (HasAudio)
            {
                AudioControl.Source = new Uri(Source.SourceFile);
                OnSourceChanged?.Invoke(this, Source);
            }

            CommandManager.InvalidateRequerySuggested();
            if (IsAutoPlayEnabled)
            {
                await PlayAsync();
            }
        }


        /// <summary>
        /// Clears thes control
        /// </summary>
        private async Task ClearAsync()
        {
            await CloseAsync();
            Source = null;
        }


        /// <summary>
        /// Determines whether this instance can clear.
        /// </summary>
        /// <returns><c>true</c> if this instance can clear; otherwise, <c>false</c>.</returns>
        private bool CanClear()
        {
            return HasAudio && IsRemoveEnabled;
        }


        /// <summary>
        /// Load source
        /// </summary>
        private async Task LoadSourceAsync()
        {
            var source = await LoadAudioAsync();
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
        /// Saves the source
        /// </summary>
        private async Task SaveAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Audio", "Audio", filter: "Audio Files (*.mp3;*.wav;*.m4a;*.aac;*.ogg;)|*.mp3;*.wav;*.m4a;*.aac;*.ogg;", defualtExt: "wav");
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
            return HasAudio;
        }


        /// <summary>
        /// Copies the source.
        /// </summary>
        private Task CopyAsync()
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
            return HasAudio;
        }


        /// <summary>
        /// Paste source
        /// </summary>
        private async Task PasteAsync()
        {
            if (!IsLoadEnabled)
                return;

            if (Clipboard.ContainsFileDropList())
            {
                var sourceFilename = Clipboard.GetFileDropList()
                    .OfType<string>()
                    .FirstOrDefault();
                var source = await LoadAudioAsync(sourceFilename);
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
                var source = await LoadAudioAsync(fileNames.FirstOrDefault());
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
            AudioControl.Play();
            MediaState = MediaState.Play;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Pauses the Video.
        /// </summary>
        private Task PauseAsync()
        {
            AudioControl.Pause();
            MediaState = MediaState.Pause;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Stops the Video.
        /// </summary>
        private async Task StopAsync()
        {
            _progressTimer.Stop();
            AudioControl.Stop();
            MediaState = MediaState.Stop;
            Progress.Clear();
            AudioProgress.Clear();
            AudioControl.Position = TimeSpan.FromMilliseconds(1);
            await Task.Delay(50);
        }


        /// <summary>
        /// Close the Audio.
        /// </summary>
        private async Task CloseAsync()
        {
            await StopAsync();
            AudioControl.Close();
            AudioControl.Source = null;
        }


        /// <summary>
        /// Load audio stream from file.
        /// </summary>
        /// <param name="initialFilename">The initial filename.</param>
        /// <returns>A Task&lt;AudioInputStream&gt; representing the asynchronous operation.</returns>
        private async Task<AudioInputStream> LoadAudioAsync(string initialFilename = null)
        {
            var sourceFilename = initialFilename ?? await DialogService.OpenFileAsync("Load Audio", "Audio", filter: "Audio/Video files (*.mp3;*.wav;*.flac;*.m4a;*.aac;*.ogg;*.mp4;*.mov;*.mkv;*.webm)|*.mp3;*.wav;*.flac;*.m4a;*.aac;*.ogg;*.mp4;*.mov;*.mkv;*.webm", defualtExt: "wav");
            if (string.IsNullOrEmpty(sourceFilename))
                return default;

            var audioInput = await AudioInputStream.CreateAsync(sourceFilename);
            OnMediaImport?.Invoke(this, new MediaImportEventArgs(sourceFilename, audioInput));
            return audioInput;
        }


        /// <summary>
        /// Updates the audio progress.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="EventArgs"/> instance containing the event data.</param>
        private void UpdateProgress(object sender, EventArgs e)
        {
            if (AudioControl.HasAudio)
            {
                ProgressPosition = AudioControl.Position;
                var duration = AudioControl.NaturalDuration.HasTimeSpan ? AudioControl.NaturalDuration.TimeSpan : Source?.Duration ?? TimeSpan.Zero;
                AudioProgress.Update((int)(ProgressPosition.TotalMilliseconds / 10), (int)(duration.TotalMilliseconds / 10));
            }
        }


        /// <summary>
        /// Handles the MediaEnded event of the MediaElement control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private async void AudioControl_MediaEnded(object sender, RoutedEventArgs e)
        {
            await StopAsync();
            if (IsReplayEnabled)
                await PlayAsync();
        }


        /// <summary>
        /// Handles the MouseDown event of the AudioControl control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseButtonEventArgs"/> instance containing the event data.</param>
        private async void AudioControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (MediaState == MediaState.Play)
                await PauseAsync();
            else if (MediaState == MediaState.Pause)
                await PlayAsync();
        }


        /// <summary>
        /// Invoked when an unhandled <see cref="E:System.Windows.Input.Mouse.MouseEnter" /> attached event is raised on this element. Implement this method to add class handling for this event.
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseEnter(MouseEventArgs e)
        {
            if (!IsKeyboardFocusWithin)
                Focus();
            base.OnMouseEnter(e);
        }


        /// <summary>
        /// Invoked when an unhandled <see cref="E:System.Windows.UIElement.MouseLeftButtonDown" /> routed event is raised on this element. Implement this method to add class handling for this event.
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseButtonEventArgs" /> that contains the event data. The event data reports that the left mouse button was pressed.</param>
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            _audioDragStart = e.GetPosition(this);
            _isAudioDragging = false;
        }


        /// <summary>
        /// Invoked when an unhandled <see cref="E:System.Windows.Input.Mouse.MouseMove" /> attached event reaches an element in its route that is derived from this class. Implement this method to add class handling for this event.
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (e.LeftButton == MouseButtonState.Pressed && !_isAudioDragging)
            {
                if (e.OriginalSource is not MediaElement)
                    return;

                var currentPosition = e.GetPosition(this);
                var diffPosition = _audioDragStart - currentPosition;
                if (Math.Abs(diffPosition.X) > Common.DragDistance || Math.Abs(diffPosition.Y) > Common.DragDistance)
                {
                    if (Source is AudioInputStream audioStream)
                    {
                        _isAudioDragging = true;
                        DragDropHelper.DoDragDropFile(this, audioStream.SourceFile, DragDropType.Audio, AudioControl, 4);
                        _isAudioDragging = false;
                    }
                }
            }
        }

    }
}
