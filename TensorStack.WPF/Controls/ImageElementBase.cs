using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using TensorStack.Common;
using TensorStack.Image;
using TensorStack.WPF.Dialogs;
using TensorStack.WPF.Services;

namespace TensorStack.WPF.Controls
{
    public class ImageElementBase : BaseControl
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageElementBase"/> class.
        /// </summary>
        public ImageElementBase()
        {
            ClearCommand = new AsyncRelayCommand(ClearAsync, CanClear);
            SaveSourceCommand = new AsyncRelayCommand(SaveSourceAsync, CanSaveSource);
            LoadSourceCommand = new AsyncRelayCommand(LoadSourceAsync, CanLoadSource);
            CopySourceCommand = new AsyncRelayCommand(CopySourceAsync, CanCopySource);
            PasteSourceCommand = new AsyncRelayCommand(PasteSourceAsync, CanPasteSource);
        }

        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(Source), typeof(ImageInput), typeof(ImageElementBase), new PropertyMetadata<ImageElementBase>((c) => c.OnValueChanged()));
        public static readonly DependencyProperty CropWidthProperty = DependencyProperty.Register(nameof(CropWidth), typeof(int), typeof(ImageElementBase));
        public static readonly DependencyProperty CropHeightProperty = DependencyProperty.Register(nameof(CropHeight), typeof(int), typeof(ImageElementBase));
        public static readonly DependencyProperty IsToolbarEnabledProperty = DependencyProperty.Register(nameof(IsToolbarEnabled), typeof(bool), typeof(ImageElementBase), new PropertyMetadata(true));
        public static readonly DependencyProperty IsLoadEnabledProperty = DependencyProperty.Register(nameof(IsLoadEnabled), typeof(bool), typeof(ImageElementBase), new PropertyMetadata(true));
        public static readonly DependencyProperty IsSaveEnabledProperty = DependencyProperty.Register(nameof(IsSaveEnabled), typeof(bool), typeof(ImageElementBase), new PropertyMetadata(true));
        public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress), typeof(ProgressInfo), typeof(ImageElementBase), new PropertyMetadata(new ProgressInfo()));
        public static readonly DependencyProperty PlaceholderProperty = DependencyProperty.Register(nameof(Placeholder), typeof(BitmapSource), typeof(ImageElementBase));

        public ImageInput Source
        {
            get { return (ImageInput)GetValue(SourceProperty); }
            set { SetValue(SourceProperty, value); }
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
        public bool HasSourceImage => Source != null;


        /// <summary>
        /// Called when DependencyProperty changeded.
        /// </summary>
        protected virtual Task OnValueChanged()
        {
            return Task.CompletedTask;
        }


        /// <summary>
        /// Clears thes control
        /// </summary>
        protected virtual Task ClearAsync()
        {
            Source = null;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can clear.
        /// </summary>
        /// <returns><c>true</c> if this instance can clear; otherwise, <c>false</c>.</returns>
        protected virtual bool CanClear()
        {
            return HasSourceImage;
        }


        /// <summary>
        /// Load source
        /// </summary>
        protected virtual async Task LoadSourceAsync()
        {
            var image = await LoadImageAsync();
            if (image != null)
                Source = image;
        }


        /// <summary>
        /// Determines whether this instance can load source.
        /// </summary>
        /// <returns><c>true</c> if this instance can load source; otherwise, <c>false</c>.</returns>
        protected virtual bool CanLoadSource()
        {
            return IsLoaded;
        }


        /// <summary>
        /// Saves the source
        /// </summary>
        protected virtual async Task SaveSourceAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Image", "Image", filter: "png files (*.png)|*.png", defualtExt: "png");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                await Source.SaveAsync(saveFilename);
            }
        }


        /// <summary>
        /// Determines whether this instance can save source.
        /// </summary>
        /// <returns><c>true</c> if this instance can save source; otherwise, <c>false</c>.</returns>
        protected virtual bool CanSaveSource()
        {
            return HasSourceImage;
        }


        /// <summary>
        /// Copies the source.
        /// </summary>
        protected virtual Task CopySourceAsync()
        {
            Clipboard.SetImage(Source.Image);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can copy source.
        /// </summary>
        /// <returns><c>true</c> if this instance can copy source; otherwise, <c>false</c>.</returns>
        protected virtual bool CanCopySource()
        {
            return HasSourceImage;
        }


        /// <summary>
        /// Paste source
        /// </summary>
        protected virtual async Task PasteSourceAsync()
        {
            if (!IsLoadEnabled)
                return;

            if (Clipboard.ContainsImage())
            {
                var image = await LoadImageAsync(initialImage: Clipboard.GetImage());
                if (image != null)
                    Source = image;
            }
            else if (Clipboard.ContainsFileDropList())
            {
                var imageFilename = Clipboard.GetFileDropList()
                    .OfType<string>()
                    .FirstOrDefault();
                var image = await LoadImageAsync(imageFilename);
                if (image != null)
                    Source = image;
            }
        }


        /// <summary>
        /// Determines whether this instance can paste source.
        /// </summary>
        /// <returns><c>true</c> if this instance can paste source; otherwise, <c>false</c>.</returns>
        protected virtual bool CanPasteSource()
        {
            return IsLoadEnabled;
        }


        /// <summary>
        /// Load image as an image from file
        /// </summary>
        /// <param name="initialFilename">The initial filename.</param>
        /// <param name="initialImage">The initial image.</param>
        /// <returns>ImageInput</returns>
        protected virtual async Task<ImageInput> LoadImageAsync(string initialFilename = null, BitmapSource initialImage = null)
        {
            if (CropWidth > 0 && CropHeight > 0)
            {
                if (!string.IsNullOrEmpty(initialFilename))
                    initialImage = await ImageService.LoadFromFileAsync(initialFilename);

                if (initialImage?.Width == CropWidth && initialImage?.Height == CropHeight)
                    return new ImageInput(initialImage);

                var loadImageDialog = DialogService.GetDialog<CropImageDialog>();
                if (await loadImageDialog.ShowDialogAsync(CropWidth, CropHeight, initialImage))
                {
                    return new ImageInput(loadImageDialog.GetImageResult());
                }
            }
            else if (initialImage is not null)
            {
                return new ImageInput(initialImage);
            }
            else
            {
                var imageFilename = initialFilename ?? await DialogService.OpenFileAsync("Open Image", filter: "Image Files|*.bmp;*.jpg;*.jpeg;*.png;*.gif;*.tif;*.tiff|All Files|*.*");
                if (!string.IsNullOrEmpty(imageFilename))
                {
                    return new ImageInput(imageFilename);
                }
            }
            return null;
        }


        /// <summary>
        /// On Drop
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.DragEventArgs" /> that contains the event data.</param>
        protected override async void OnDrop(DragEventArgs e)
        {
            base.OnDrop(e);
            if (!IsLoadEnabled)
                return;

            Progress.Indeterminate();
            var fileNames = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (!fileNames.IsNullOrEmpty())
            {
                var image = await LoadImageAsync(fileNames.FirstOrDefault());
                if (image != null)
                    Source = image;
            }
            else
            {
                var bitmapImage = e.Data.GetData(typeof(BitmapSource)) as BitmapSource;
                if (bitmapImage is not null)
                {
                    var image = await LoadImageAsync(initialImage: bitmapImage);
                    if (image != null)
                        Source = image;
                }
            }

            Progress.Clear();
        }
    }
}
