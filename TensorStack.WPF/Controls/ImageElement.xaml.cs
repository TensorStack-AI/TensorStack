using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using TensorStack.Image;
using TensorStack.WPF.Services;
using TensorStack.WPF.Utils;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for ImageElement.xaml
    /// </summary>
    public partial class ImageElement : ImageElementBase
    {
        public ImageElement()
        {
            LoadOverlayCommand = new AsyncRelayCommand(LoadOverlayAsync, CanLoadOverlay);
            SaveOverlayCommand = new AsyncRelayCommand(SaveOverlayAsync, CanSaveOverlay);
            CopyOverlayCommand = new AsyncRelayCommand(CopyOverlayAsync, CanCopyOverlay);

            SaveCanvasCommand = new AsyncRelayCommand(SaveCanvasAsync, CanSaveCanvas);
            CopyCanvasCommand = new AsyncRelayCommand(CopyCanvasAsync, CanCopyCanvas);
            InitializeComponent();
        }


        public static readonly DependencyProperty OverlaySourceProperty =
            DependencyProperty.Register(nameof(OverlaySource), typeof(ImageInput), typeof(ImageElement), new PropertyMetadata<ImageElement>((c) => c.OnValueChanged()));

        public static readonly DependencyProperty SplitterPositionProperty =
            DependencyProperty.Register(nameof(SplitterPosition), typeof(SplitterPosition), typeof(ImageElement), new PropertyMetadata<ImageElement>((c) => c.OnValueChanged()));

        public static readonly DependencyProperty SplitterVisibilityProperty =
            DependencyProperty.Register(nameof(SplitterVisibility), typeof(SplitterVisibility), typeof(ImageElement), new PropertyMetadata<ImageElement>((c) => c.OnValueChanged()));

        public static readonly DependencyProperty SplitterDirectionProperty =
            DependencyProperty.Register(nameof(SplitterDirection), typeof(SplitterDirection), typeof(ImageElement), new PropertyMetadata<ImageElement>((c) => c.OnValueChanged()));

        public static readonly DependencyProperty IsLoadOverlayEnabledProperty =
            DependencyProperty.Register(nameof(IsLoadOverlayEnabled), typeof(bool), typeof(ImageElement));

        public static readonly DependencyProperty IsSaveOverlayEnabledProperty =
            DependencyProperty.Register(nameof(IsSaveOverlayEnabled), typeof(bool), typeof(ImageElement));

        public static readonly DependencyProperty IsSaveCanvasEnabledProperty =
            DependencyProperty.Register(nameof(IsSaveCanvasEnabled), typeof(bool), typeof(ImageElement));

        public ImageInput OverlaySource
        {
            get { return (ImageInput)GetValue(OverlaySourceProperty); }
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

        public bool IsSaveCanvasEnabled
        {
            get { return (bool)GetValue(IsSaveCanvasEnabledProperty); }
            set { SetValue(IsSaveCanvasEnabledProperty, value); }
        }

        public AsyncRelayCommand LoadOverlayCommand { get; }
        public AsyncRelayCommand SaveOverlayCommand { get; }
        public AsyncRelayCommand SaveCanvasCommand { get; }
        public AsyncRelayCommand CopyOverlayCommand { get; }
        public AsyncRelayCommand CopyCanvasCommand { get; }

        public bool HasOverlayImage => OverlaySource != null;


        /// <summary>
        /// Called when DependencyProperty changeded.
        /// </summary>
        protected override async Task OnValueChanged()
        {
            await base.OnValueChanged();
            if (HasOverlayImage)
            {
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
                    await Task.Delay(10);
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
        /// Clears thes control
        /// </summary>
        /// <returns>Task.</returns>
        protected override Task ClearAsync()
        {
            Source = null;
            OverlaySource = null;
            GridSplitterContainer.Visibility = Visibility.Hidden;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can clear.
        /// </summary>
        /// <returns><c>true</c> if this instance can clear; otherwise, <c>false</c>.</returns>
        protected override bool CanClear()
        {
            return HasSourceImage || HasOverlayImage;
        }


        /// <summary>
        /// Load overlay
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        private async Task LoadOverlayAsync()
        {
            var image = await LoadImageAsync();
            if (image != null)
                OverlaySource = image;
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
        /// Save the overlay
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        private async Task SaveOverlayAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Image", "Overlay", filter: "png files (*.png)|*.png", defualtExt: "png");
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
            return HasOverlayImage;
        }


        /// <summary>
        /// Save the canvas
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        private async Task SaveCanvasAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Image", "Canavs", filter: "png files (*.png)|*.png", defualtExt: "png");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                var canvasSource = CreateCanvasSource();
                using (var imageInput = new ImageInput(canvasSource))
                {
                    await imageInput.SaveAsync(saveFilename);
                }
            }
        }


        /// <summary>
        /// Determines whether this instance can save canvas.
        /// </summary>
        /// <returns><c>true</c> if this instance can save canvas; otherwise, <c>false</c>.</returns>
        private bool CanSaveCanvas()
        {
            return HasSourceImage && HasOverlayImage;
        }


        /// <summary>
        /// Copies the overlay.
        /// </summary>
        /// <returns>Task.</returns>
        private Task CopyOverlayAsync()
        {
            Clipboard.SetImage(OverlaySource.Image);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can copy overlay.
        /// </summary>
        /// <returns><c>true</c> if this instance can copy overlay; otherwise, <c>false</c>.</returns>
        private bool CanCopyOverlay()
        {
            return HasOverlayImage;
        }


        /// <summary>
        /// Copies the canvas.
        /// </summary>
        /// <returns>Task.</returns>
        private Task CopyCanvasAsync()
        {
            var canvasSource = CreateCanvasSource();
            Clipboard.SetImage(canvasSource);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can copy canvas.
        /// </summary>
        /// <returns><c>true</c> if this instance can copy canvas; otherwise, <c>false</c>.</returns>
        private bool CanCopyCanvas()
        {
            return HasSourceImage && HasOverlayImage;
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
        /// Creates the canvas BitmapSource.
        /// </summary>
        /// <returns>BitmapSource.</returns>
        private BitmapSource CreateCanvasSource()
        {
            var visual = new DrawingVisual();
            var renderBitmap = new RenderTargetBitmap(Source.Width, Source.Height, 96, 96, PixelFormats.Bgra32);
            using (var drawingContext = visual.RenderOpen())
            {
                drawingContext.DrawRectangle(new VisualBrush(SourceContainer), null, new Rect(new Point(0, 0), new Point(Source.Width, Source.Height)));
            }
            renderBitmap.Render(visual);
            return renderBitmap;
        }


        /// <summary>
        /// GridSplitter SizeChanged event, Update Overlay Clip
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="SizeChangedEventArgs"/> instance containing the event data.</param>
        private void GridSplitter_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            ImageOverlayControl.Clip = SplitterDirection == SplitterDirection.LeftToRight
                ? new RectangleGeometry(new Rect(0, 0, e.NewSize.Width, e.NewSize.Height))
                : new RectangleGeometry(new Rect(e.NewSize.Width, 0, ImageOverlayControl.ActualWidth, ImageOverlayControl.ActualHeight));
        }


        /// <summary>
        /// MouseEnter
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseEnter(MouseEventArgs e)
        {
            if (HasOverlayImage && SplitterVisibility != SplitterVisibility.Manual)
            {
                GridSplitterContainer.Visibility = Visibility.Visible;
            }
            Keyboard.Focus(this);
            base.OnMouseEnter(e);
        }


        /// <summary>
        /// MouseLeave
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseLeave(MouseEventArgs e)
        {
            if (HasOverlayImage && SplitterVisibility != SplitterVisibility.Manual)
            {
                GridSplitterContainer.Visibility = Visibility.Hidden;
            }
            //Keyboard.ClearFocus();
            base.OnMouseLeave(e);
        }


        /// <summary>
        /// MouseLeftButtonDown
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseButtonEventArgs" /> that contains the event data. The event data reports that the left mouse button was pressed.</param>
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            if (e.OriginalSource is System.Windows.Controls.Image image)
            {
                if (image.Source is BitmapSource bitmapSource)
                {
                    DragDropHelper.DoDragDropObject(this, bitmapSource, DragDropType.Image, ImageControl, 4);
                }
            }
        }
    }

    public enum SplitterPosition
    {
        Source,
        Center,
        Overlay
    }

    public enum SplitterDirection
    {
        LeftToRight,
        RightToLeft
    }

    public enum SplitterVisibility
    {
        Auto,
        Mouse,
        Manual
    }
}
