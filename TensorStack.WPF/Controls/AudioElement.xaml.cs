using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;
using TensorStack.Audio;
using TensorStack.Common;
using TensorStack.Common.Common;
using TensorStack.WPF.Services;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for AudioElement.xaml
    /// </summary>
    public partial class AudioElement : BaseControl
    {
        private readonly DispatcherTimer _progressTimer;
        private string _fileSource;
        private MediaState _mediaState;
        private TimeSpan _progressPosition;
        private ProgressInfo _progress;

        public AudioElement()
        {
            _progressTimer = new DispatcherTimer(TimeSpan.FromMilliseconds(50), DispatcherPriority.Normal, UpdateProgress, Dispatcher);
            Progress = new ProgressInfo();
            ClearCommand = new AsyncRelayCommand(ClearAsync, CanClear);
            PlayCommand = new AsyncRelayCommand(PlayAsync, CanSaveSource);
            PauseCommand = new AsyncRelayCommand(PauseAsync, CanSaveSource);
            StopCommand = new AsyncRelayCommand(StopAsync, CanSaveSource);
            SaveCommand = new AsyncRelayCommand(SaveAsync, CanSaveSource);
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoadSource);
            CopyCommand = new AsyncRelayCommand(CopyAsync, CanCopySource);
            PasteCommand = new AsyncRelayCommand(PasteAsync, CanPasteSource);
            MuteCommand = new RelayCommand(() => Volume = 0);
            InitializeComponent();
        }

        public static readonly DependencyProperty ConfigurationProperty = DependencyProperty.Register(nameof(Configuration), typeof(IUIConfiguration), typeof(AudioElement));
        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(Source), typeof(AudioInput), typeof(AudioElement), new PropertyMetadata<AudioElement>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty IsReplayEnabledProperty = DependencyProperty.Register(nameof(IsReplayEnabled), typeof(bool), typeof(AudioElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsAutoPlayEnabledProperty = DependencyProperty.Register(nameof(IsAutoPlayEnabled), typeof(bool), typeof(AudioElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadEnabledProperty = DependencyProperty.Register(nameof(IsLoadEnabled), typeof(bool), typeof(AudioElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsSaveEnabledProperty = DependencyProperty.Register(nameof(IsSaveEnabled), typeof(bool), typeof(AudioElement), new PropertyMetadata(true));
        public static readonly DependencyProperty VolumeProperty = DependencyProperty.Register(nameof(Volume), typeof(double), typeof(AudioElement));

        public IUIConfiguration Configuration
        {
            get { return (IUIConfiguration)GetValue(ConfigurationProperty); }
            set { SetValue(ConfigurationProperty, value); }
        }

        public AudioInput Source
        {
            get { return (AudioInput)GetValue(SourceProperty); }
            set { SetValue(SourceProperty, value); }
        }


        public ProgressInfo Progress
        {
            get { return _progress; }
            set { SetProperty(ref _progress, value); }
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

        public double Volume
        {
            get { return (double)GetValue(VolumeProperty); }
            set { SetValue(VolumeProperty, value); }
        }


        public AsyncRelayCommand ClearCommand { get; }
        public AsyncRelayCommand LoadCommand { get; }
        public AsyncRelayCommand SaveCommand { get; }
        public AsyncRelayCommand CopyCommand { get; }
        public AsyncRelayCommand PasteCommand { get; }
        public AsyncRelayCommand PlayCommand { get; }
        public AsyncRelayCommand PauseCommand { get; }
        public AsyncRelayCommand StopCommand { get; }
        public RelayCommand MuteCommand { get; }
        public bool HasAudio => Source != null;

        public string FileSource
        {
            get { return _fileSource; }
            set { _fileSource = value; NotifyPropertyChanged(); }
        }

        public MediaState MediaState
        {
            get { return _mediaState; }
            set
            {
                _mediaState = value;
                AudioControl.LoadedBehavior = _mediaState;
                NotifyPropertyChanged();
            }
        }

        public TimeSpan ProgressPosition
        {
            get { return _progressPosition; }
            set { _progressPosition = value; NotifyPropertyChanged(); }
        }


        private async Task OnValueChanged()
        {
            if (HasAudio)
            {
                // If filename is null, create and save temp file
                FileSource = await GetOrCreateFileSource(Source);
            }
        }


        private Task ClearAsync()
        {
            MediaState = MediaState.Close;
            FileSource = null;
            Source = null;
            Progress.Value = 0;
            ProgressPosition = TimeSpan.Zero;
            _progressTimer.Stop();
            MediaState = IsAutoPlayEnabled
                ? MediaState.Play
                : MediaState.Stop;
            return Task.CompletedTask;
        }

        private bool CanClear()
        {
            return HasAudio;
        }


        private async Task LoadAsync()
        {
            var source = await LoadAudioAsync();
            if (source != null)
                Source = source;
        }


        private bool CanLoadSource()
        {
            return IsLoadEnabled;
        }


        private async Task SaveAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Audio", "Audio", filter: "wav files (*.wav)|*.wav", defualtExt: "wav");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                await Source.SaveAsync(saveFilename);
            }
        }


        private bool CanSaveSource()
        {
            return IsSaveEnabled && HasAudio;
        }

        private Task CopyAsync()
        {
            Clipboard.SetFileDropList([Source.SourceFile]);
            return Task.CompletedTask;
        }

        private bool CanCopySource()
        {
            return HasAudio;
        }


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


        private bool CanPasteSource()
        {
            return IsLoadEnabled;
        }



        private Task PlayAsync()
        {
            if (MediaState == MediaState.Close || MediaState == MediaState.Play)
                return Task.CompletedTask;

            MediaState = MediaState.Play;
            return Task.CompletedTask;
        }


        private Task PauseAsync()
        {
            if (MediaState != MediaState.Play)
                return Task.CompletedTask;

            MediaState = MediaState.Pause;
            return Task.CompletedTask;
        }


        private Task StopAsync()
        {
            if (MediaState == MediaState.Close || MediaState == MediaState.Stop)
                return Task.CompletedTask;

            Progress.Value = 0;
            ProgressPosition = TimeSpan.Zero;
            AudioControl.Position = TimeSpan.Zero;
            MediaState = MediaState.Stop;
            return Task.CompletedTask;
        }


        private async Task<AudioInput> LoadAudioAsync(string initialFilename = null)
        {
            var sourceFilename = initialFilename ?? await DialogService.OpenFileAsync("Load Audio", "Audio", filter: "wav files (*.wav)|*.wav", defualtExt: "wav");
            if (string.IsNullOrEmpty(sourceFilename))
                return default;

            return await AudioInput.CreateAsync(sourceFilename);
        }



        protected override void OnMouseEnter(MouseEventArgs e)
        {
            if (!IsKeyboardFocusWithin)
                Keyboard.Focus(this);

            base.OnMouseEnter(e);
        }


        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            if (e.OriginalSource is MediaElement audioElement)
            {
                AllowDrop = false;
                DragDrop.DoDragDrop(audioElement, new DataObject(typeof(Uri), audioElement.Source), DragDropEffects.Copy);
                AllowDrop = true;
            }
            base.OnMouseLeftButtonDown(e);
        }


        protected override async void OnDrop(DragEventArgs e)
        {
            if (!IsLoadEnabled)
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


        private void AudioControl_Loaded(object sender, RoutedEventArgs e)
        {
            if (IsAutoPlayEnabled)
            {
                MediaState = MediaState.Play;
            }
        }


        private void AudioControl_MediaOpened(object sender, RoutedEventArgs e)
        {
            if (HasAudio)
            {
                _progressTimer.Start();
                if (Progress is not null)
                    Progress.Maximum = (int)Source.Duration.TotalMilliseconds;
            }
        }


        private async void AudioControl_MediaEnded(object sender, RoutedEventArgs e)
        {
            await StopAsync();
            if (IsReplayEnabled)
            {
                MediaState = MediaState.Play;
            }
        }


        private void AudioControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            MediaState = MediaState == MediaState.Pause || MediaState == MediaState.Stop
                 ? MediaState.Play
                 : MediaState.Pause;
        }


        private void UpdateProgress(object sender, EventArgs e)
        {
            if (AudioControl.HasAudio)
            {
                ProgressPosition = AudioControl.Position;
                if (Progress is not null)
                    Progress.Value = (int)ProgressPosition.TotalMilliseconds;
            }
        }


        private async Task<string> GetOrCreateFileSource(AudioInput source)
        {
            if (!string.IsNullOrEmpty(source.SourceFile))
                return source.SourceFile;

            var tempFilename = FileHelper.RandomFileName(Configuration.DirectoryTemp, "wav");
            await source.SaveAsync(tempFilename);
            return tempFilename;
        }
    }
}
