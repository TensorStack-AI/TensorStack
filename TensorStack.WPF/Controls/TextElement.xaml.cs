using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using TensorStack.Common;
using TensorStack.WPF.Services;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for TextElement.xaml
    /// </summary>
    public partial class TextElement : BaseControl
    {
        public TextElement()
        {
            LoadCommand = new AsyncRelayCommand(LoadAsync, CanLoad);
            SaveCommand = new AsyncRelayCommand(SaveAsync, CanSave);
            ClearCommand = new AsyncRelayCommand(ClearAsync, CanClear);
            InitializeComponent();
        }

        public static readonly DependencyProperty TextSourceProperty = DependencyProperty.Register(nameof(TextSource), typeof(TextInput), typeof(TextElement), new FrameworkPropertyMetadata(new TextInput(string.Empty), FrameworkPropertyMetadataOptions.BindsTwoWayByDefault, default, default, true, defaultUpdateSourceTrigger: System.Windows.Data.UpdateSourceTrigger.PropertyChanged));
        public static readonly DependencyProperty IsReadOnlyProperty = DependencyProperty.Register(nameof(IsReadOnly), typeof(bool), typeof(TextElement), new PropertyMetadata(false));
        public static readonly DependencyProperty AcceptsReturnProperty = DependencyProperty.Register(nameof(AcceptsReturn), typeof(bool), typeof(TextElement), new PropertyMetadata(true));
        public static readonly DependencyProperty TextWrappingProperty = DependencyProperty.Register(nameof(TextWrapping), typeof(TextWrapping), typeof(TextElement), new PropertyMetadata(TextWrapping.Wrap));
        public static readonly DependencyProperty IsLoadEnabledProperty = DependencyProperty.Register(nameof(IsLoadEnabled), typeof(bool), typeof(TextElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsSaveEnabledProperty = DependencyProperty.Register(nameof(IsSaveEnabled), typeof(bool), typeof(TextElement), new PropertyMetadata(true));
        public static readonly DependencyProperty IsRemoveEnabledProperty = DependencyProperty.Register(nameof(IsRemoveEnabled), typeof(bool), typeof(TextElement), new PropertyMetadata(true));
        public event EventHandler<MediaImportEventArgs> OnMediaImport;

        public TextInput TextSource
        {
            get { return (TextInput)GetValue(TextSourceProperty); }
            set { SetValue(TextSourceProperty, value); }
        }

        public bool IsReadOnly
        {
            get { return (bool)GetValue(IsReadOnlyProperty); }
            set { SetValue(IsReadOnlyProperty, value); }
        }

        public bool AcceptsReturn
        {
            get { return (bool)GetValue(AcceptsReturnProperty); }
            set { SetValue(AcceptsReturnProperty, value); }
        }

        public TextWrapping TextWrapping
        {
            get { return (TextWrapping)GetValue(TextWrappingProperty); }
            set { SetValue(TextWrappingProperty, value); }
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
        public AsyncRelayCommand ClearCommand { get; }
        public AsyncRelayCommand LoadCommand { get; }
        public AsyncRelayCommand SaveCommand { get; }
        public bool HasSourceText => TextSource?.Length > 0;


        private async Task LoadAsync()
        {
            var sourceFilename = await DialogService.OpenFileAsync("Load Text", filter: "Text files |*.txt;|All Files|*.*;", defualtExt: "txt");
            if (!string.IsNullOrEmpty(sourceFilename))
            {
                var textInput = await TensorStack.Common.TextInput.CreateAsync(sourceFilename, Encoding.UTF8);
                OnMediaImport?.Invoke(this, new MediaImportEventArgs(MediaType.Text, sourceFilename));
                TextSource = textInput;
            }
        }


        /// <summary>
        /// Determines whether this instance can load source.
        /// </summary>
        /// <returns><c>true</c> if this instance can load source; otherwise, <c>false</c>.</returns>
        private bool CanLoad()
        {
            return true;
        }


        private async Task SaveAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Text", "Text", filter: "Text files (*.txt)|*.txt", defualtExt: "txt");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                await File.WriteAllTextAsync(saveFilename, TextSource.Text);
            }
        }


        /// <summary>
        /// Determines whether this instance can save source.
        /// </summary>
        /// <returns><c>true</c> if this instance can save source; otherwise, <c>false</c>.</returns>
        private bool CanSave()
        {
            return HasSourceText;
        }



        private Task ClearAsync()
        {
            TextSource = new TextInput(string.Empty);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can clear.
        /// </summary>
        /// <returns><c>true</c> if this instance can clear; otherwise, <c>false</c>.</returns>
        private bool CanClear()
        {
            return HasSourceText;
        }


        /// <summary>
        /// Invoked when an unhandled <see cref="E:System.Windows.DragDrop.DragEnter" /> attached event reaches an element in its route that is derived from this class. Implement this method to add class handling for this event.
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.DragEventArgs" /> that contains the event data.</param>
        protected override async void OnDrop(DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var fileDrop = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (fileDrop.IsNullOrEmpty())
                    return;

                var filename = fileDrop.FirstOrDefault();
                if (!File.Exists(filename))
                    return;

                TextSource = await TensorStack.Common.TextInput.CreateAsync(filename, Encoding.UTF8);
            }
            else if (e.Data.GetDataPresent(DataFormats.Text))
            {
                TextSource = new TextInput((string)e.Data.GetData(DataFormats.Text));
            }
            base.OnDrop(e);
        }


        /// <summary>
        /// Invoked when an unhandled <see cref="E:System.Windows.Input.Mouse.MouseEnter" /> attached event is raised on this element. Implement this method to add class handling for this event.
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseEnter(MouseEventArgs e)
        {
            if (!IsKeyboardFocusWithin)
                Keyboard.Focus(this);

            base.OnMouseEnter(e);
        }

        private void NoDropTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void TextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            NotifyPropertyChanged(nameof(TextSource));
        }
    }

    public class NoDropTextBox : TextBox
    {
        protected override void OnDrop(DragEventArgs e) { /*swallow   */ }
        protected override void OnDragOver(DragEventArgs e) { /*swallow   */ }
    }
}
