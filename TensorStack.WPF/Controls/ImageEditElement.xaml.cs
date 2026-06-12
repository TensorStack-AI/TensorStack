using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using TensorStack.Image;
using TensorStack.WPF.Adorner;
using TensorStack.WPF.Services;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for ImageEditElement.xaml
    /// </summary>
    public partial class ImageEditElement : ImageElementBase
    {
        private readonly ResizeAdorner _resizeImageAdorner;
        private readonly List<Stroke> _maskStrokedRemoved;
        private int _maskDrawSize;
        private bool _isMaskInvertEnabled;
        private DrawingAttributes _maskAttributes;
        private InkCanvasEditingMode _maskEditingMode;
        private ImageEditMode _selectedEditMode;
        private Point _selectionStartPoint;
        private Rectangle _selectionRectangle;
        private int _selectionSize;
        private bool _isSelectionFilled;
        private bool _isResizeEnabled;
        private bool _isSelectEnabled;
        private bool _isMaskEnabled;

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageEditElement"/> class.
        /// </summary>
        public ImageEditElement()
        {
            MaskDrawSize = 20;
            SelectionSize = 4;
            _maskStrokedRemoved = new List<Stroke>();
            MaskEditingMode = InkCanvasEditingMode.None;

            ChangeModeCommand = new AsyncRelayCommand<ImageEditMode>(ChangeModeAsync);
            SaveCanvasCommand = new AsyncRelayCommand(SaveCanvasAsync, CanSaveCanvas);
            CopyCanvasCommand = new AsyncRelayCommand(CopyCanvasAsync, CanCopyCanvas);

            LoadMaskCommand = new AsyncRelayCommand(LoadMaskAsync, CanLoadMask);
            SaveMaskCommand = new AsyncRelayCommand(SaveMaskAsync, CanSaveMask);
            CopyMaskCommand = new AsyncRelayCommand(CopyMaskAsync, CanCopyMask);
            MaskClearCommand = new AsyncRelayCommand(MaskClear);
            MaskUndoCommand = new AsyncRelayCommand(MaskUndo, CanMaskUndo);
            MaskRedoCommand = new AsyncRelayCommand(MaskRedo, CanMaskRedo);

            InitializeComponent();
            _resizeImageAdorner = new ResizeAdorner(ResizeContainer, true, 3f, x => NotifyPropertyChanged(nameof(IsResizeModified)));
        }

        public static readonly DependencyProperty SourceMaskProperty = DependencyProperty.Register(nameof(SourceMask), typeof(ImageInput), typeof(ImageEditElement), new PropertyMetadata<ImageEditElement>((c) => c.OnMaskValueChanged()));
        public static readonly DependencyProperty CanvasWidthProperty = DependencyProperty.Register(nameof(CanvasWidth), typeof(int), typeof(ImageEditElement), new PropertyMetadata<ImageEditElement>((c) => c.OnSizeChanged()) { DefaultValue = 512 });
        public static readonly DependencyProperty CanvasHeightProperty = DependencyProperty.Register(nameof(CanvasHeight), typeof(int), typeof(ImageEditElement), new PropertyMetadata<ImageEditElement>((c) => c.OnSizeChanged()) { DefaultValue = 512 });
        public static readonly DependencyProperty IsLoadMaskEnabledProperty = DependencyProperty.Register(nameof(IsLoadMaskEnabled), typeof(bool), typeof(ImageEditElement));
        public static readonly DependencyProperty IsSaveMaskEnabledProperty = DependencyProperty.Register(nameof(IsSaveMaskEnabled), typeof(bool), typeof(ImageEditElement));
        public static readonly DependencyProperty IsSaveCanvasEnabledProperty = DependencyProperty.Register(nameof(IsSaveCanvasEnabled), typeof(bool), typeof(ImageEditElement));
        public static readonly DependencyProperty MaskColorProperty = DependencyProperty.Register(nameof(MaskColor), typeof(Color), typeof(ImageEditElement), new PropertyMetadata<ImageEditElement>((c) => c.OnMaskColorChanged()) { DefaultValue = Color.FromArgb(128, 255, 0, 0) });
        public static readonly DependencyProperty CanvasColorProperty = DependencyProperty.Register(nameof(CanvasColor), typeof(Color), typeof(ImageEditElement), new PropertyMetadata(Color.FromArgb(255, 128, 128, 128)));
        public static readonly DependencyProperty SelectionColorProperty = DependencyProperty.Register(nameof(SelectionColor), typeof(Color), typeof(ImageEditElement), new PropertyMetadata(Colors.Red));
        public AsyncRelayCommand<ImageEditMode> ChangeModeCommand { get; }
        public AsyncRelayCommand SaveCanvasCommand { get; }
        public AsyncRelayCommand LoadMaskCommand { get; }
        public AsyncRelayCommand SaveMaskCommand { get; }
        public AsyncRelayCommand CopyMaskCommand { get; }
        public AsyncRelayCommand CopyCanvasCommand { get; }
        public AsyncRelayCommand MaskClearCommand { get; }
        public AsyncRelayCommand MaskUndoCommand { get; }
        public AsyncRelayCommand MaskRedoCommand { get; }
        public bool HasMaskImage => SourceMask != null;
        public bool HasImage => HasSourceImage || HasMaskImage;

        public int CanvasWidth
        {
            get { return (int)GetValue(CanvasWidthProperty); }
            set { SetValue(CanvasWidthProperty, value); }
        }

        public int CanvasHeight
        {
            get { return (int)GetValue(CanvasHeightProperty); }
            set { SetValue(CanvasHeightProperty, value); }
        }

        public ImageInput SourceMask
        {
            get { return (ImageInput)GetValue(SourceMaskProperty); }
            set { SetValue(SourceMaskProperty, value); }
        }

        public bool IsLoadMaskEnabled
        {
            get { return (bool)GetValue(IsLoadMaskEnabledProperty); }
            set { SetValue(IsLoadMaskEnabledProperty, value); }
        }

        public bool IsSaveMaskEnabled
        {
            get { return (bool)GetValue(IsSaveMaskEnabledProperty); }
            set { SetValue(IsSaveMaskEnabledProperty, value); }
        }

        public bool IsSaveCanvasEnabled
        {
            get { return (bool)GetValue(IsSaveCanvasEnabledProperty); }
            set { SetValue(IsSaveCanvasEnabledProperty, value); }
        }

        public Color MaskColor
        {
            get { return (Color)GetValue(MaskColorProperty); }
            set { SetValue(MaskColorProperty, value); }
        }

        public Color CanvasColor
        {
            get { return (Color)GetValue(CanvasColorProperty); }
            set { SetValue(CanvasColorProperty, value); }
        }

        public Color SelectionColor
        {
            get { return (Color)GetValue(SelectionColorProperty); }
            set { SetValue(SelectionColorProperty, value); }
        }

        public bool IsMaskEnabled
        {
            get { return _isMaskEnabled; }
            set { SetProperty(ref _isMaskEnabled, value); }
        }

        public bool IsResizeEnabled
        {
            get { return _isResizeEnabled; }
            set { SetProperty(ref _isResizeEnabled, value); }
        }

        public bool IsSelectEnabled
        {
            get { return _isSelectEnabled; }
            set { SetProperty(ref _isSelectEnabled, value); }
        }

        public ImageEditMode SelectedEditMode
        {
            get { return _selectedEditMode; }
            set
            {
                SetProperty(ref _selectedEditMode, value);
                UpdateEditModeAsync();
            }
        }

        public InkCanvasEditingMode MaskEditingMode
        {
            get { return _maskEditingMode; }
            set { SetProperty(ref _maskEditingMode, value); }
        }

        public DrawingAttributes MaskAttributes
        {
            get { return _maskAttributes; }
            set { SetProperty(ref _maskAttributes, value); }
        }

        public bool IsMaskInvertEnabled
        {
            get { return _isMaskInvertEnabled; }
            set { SetProperty(ref _isMaskInvertEnabled, value); }
        }

        public int MaskDrawSize
        {
            get { return _maskDrawSize; }
            set
            {
                SetProperty(ref _maskDrawSize, value);
                UpdateMaskAttributes();
            }
        }

        public int SelectionSize
        {
            get { return _selectionSize; }
            set { SetProperty(ref _selectionSize, value); UpdateSelectionSize(); }
        }

        public bool IsSelectionFilled
        {
            get { return _isSelectionFilled; }
            set { SetProperty(ref _isSelectionFilled, value); }
        }


        /// <summary>
        /// Gets a value indicating whether mask canvas is modified.
        /// </summary>
        public bool IsMaskModified => MaskCanvas?.Strokes.Count > 0;

        /// <summary>
        /// Gets a value indicating whether resize canvas is modified.
        /// </summary>
        public bool IsResizeModified => IsResizeCanvasChanged();

        /// <summary>
        /// Gets a value indicating whether selection rectangle is modified.
        /// </summary>
        public bool IsSelectionModified => _selectionRectangle != null;


        /// <summary>
        /// Gets the input image.
        /// </summary>
        public ImageInput GetImage()
        {
            if (IsResizeModified)
                return new ImageInput(CreateBitmap(ImageContainer));

            return Source;
        }


        /// <summary>
        /// Gets the input image mask.
        /// </summary>
        public ImageInput GetImageMask()
        {
            if (IsMaskModified && IsSelectionModified)
                return new ImageInput(CreateBitmap(MaskContainer).ToBlackWhiteMask(_isMaskInvertEnabled));

            if (!IsMaskModified && IsSelectionModified)
                return new ImageInput(CreateBitmap(SelectionCanvas).ToBlackWhiteMask(_isMaskInvertEnabled));

            return new ImageInput(CreateBitmap(MaskCanvas).ToBlackWhiteMask(_isMaskInvertEnabled));
        }


        /// <summary>
        /// Gets the input image canvas.
        /// </summary>
        public ImageInput GetImageCanvas()
        {
            if (IsMaskModified || IsSelectionModified || IsResizeModified)
                return new ImageInput(CreateBitmap(SourceContainer));

            return Source;
        }


        /// <summary>
        /// Called when DependencyProperty changeded.
        /// </summary>
        protected override async Task OnValueChanged()
        {
            await base.OnValueChanged();
            UpdateMaskAttributes();
            NotifyPropertyChanged(nameof(HasImage));
        }


        /// <summary>
        /// Called when DependencyProperty changeded.
        /// </summary>
        private async Task OnMaskValueChanged()
        {
            if (HasMaskImage)
            {
                await CreateMaskImage(SourceMask);
            }
            UpdateMaskAttributes();
            NotifyPropertyChanged(nameof(HasImage));
        }


        /// <summary>
        /// Clears the control
        /// </summary>
        protected override Task ClearAsync()
        {
            Source = null;
            SourceMask = null;
            ResetCanvas();
            MaskClear();
            ChangeModeAsync(ImageEditMode.None);
            NotifyPropertyChanged(nameof(HasImage));
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can clear.
        /// </summary>
        /// <returns><c>true</c> if this instance can clear; otherwise, <c>false</c>.</returns>
        protected override bool CanClear()
        {
            return (HasSourceImage || HasMaskImage) && IsRemoveEnabled;
        }


        /// <summary>
        /// Load Mask
        /// </summary>
        private async Task LoadMaskAsync()
        {
            var image = await LoadImageAsync();
            if (image != null)
            {
                await CreateMaskImage(image);
            }
        }


        /// <summary>
        /// Determines whether this instance can load Mask.
        /// </summary>
        /// <returns><c>true</c> if this instance can load overMasklay; otherwise, <c>false</c>.</returns>
        private bool CanLoadMask()
        {
            return IsLoadMaskEnabled && IsInputEnabled;
        }


        /// <summary>
        /// Save the Mask
        /// </summary>
        private async Task SaveMaskAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Mask", "Mask", filter: "png files (*.png)|*.png", defualtExt: "png");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                var maskImage = GetImageMask();
                await maskImage.SaveAsync(saveFilename);
            }
        }


        /// <summary>
        /// Determines whether this instance can save Mask.
        /// </summary>
        /// <returns><c>true</c> if this instance can save Mask; otherwise, <c>false</c>.</returns>
        private bool CanSaveMask()
        {
            return HasSourceImage && (IsMaskModified || IsSelectionModified);
        }


        /// <summary>
        /// Save the canvas
        /// </summary>
        private async Task SaveCanvasAsync()
        {
            var saveFilename = await DialogService.SaveFileAsync("Save Canvas", "Canvas", filter: "png files (*.png)|*.png", defualtExt: "png");
            if (!string.IsNullOrEmpty(saveFilename))
            {
                var canvasImage = GetImageCanvas();
                await canvasImage.SaveAsync(saveFilename);
            }
        }


        /// <summary>
        /// Determines whether this instance can save canvas.
        /// </summary>
        /// <returns><c>true</c> if this instance can save canvas; otherwise, <c>false</c>.</returns>
        private bool CanSaveCanvas()
        {
            return HasSourceImage && (IsMaskModified || IsResizeModified || IsSelectionModified);
        }


        /// <summary>
        /// Copies the Mask.
        /// </summary>
        private Task CopyMaskAsync()
        {
            var maskImage = GetImageMask();
            Clipboard.SetImage(maskImage.Image);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can copy Mask.
        /// </summary>
        /// <returns><c>true</c> if this instance can copy Mask; otherwise, <c>false</c>.</returns>
        private bool CanCopyMask()
        {
            return HasSourceImage && (IsMaskModified || IsSelectionModified);
        }


        /// <summary>
        /// Copies the canvas.
        /// </summary>
        private Task CopyCanvasAsync()
        {
            var canvasImage = GetImageCanvas();
            Clipboard.SetImage(canvasImage.Image);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether this instance can copy canvas.
        /// </summary>
        /// <returns><c>true</c> if this instance can copy canvas; otherwise, <c>false</c>.</returns>
        private bool CanCopyCanvas()
        {
            return HasSourceImage;
        }


        /// <summary>
        /// Clear the Mask
        /// </summary>
        /// <returns>Task.</returns>
        private Task MaskClear()
        {
            if (_selectedEditMode == ImageEditMode.Select)
            {
                ClearSelection();
            }
            else if (_selectedEditMode == ImageEditMode.Resize)
            {
                ResetCanvas();
            }
            else if (_selectedEditMode == ImageEditMode.Draw || _selectedEditMode == ImageEditMode.Erase)
            {
                MaskCanvas.Strokes.Clear();
                MaskCanvas.Background = Brushes.Transparent;
                NotifyPropertyChanged(nameof(IsMaskModified));
            }

            return Task.CompletedTask;
        }


        /// <summary>
        /// Undo last mask stroke
        /// </summary>
        private Task MaskUndo()
        {
            if (MaskCanvas.Strokes.Count == 0)
                return Task.CompletedTask;

            var lastStroke = MaskCanvas.Strokes.Last();
            if (MaskCanvas.Strokes.Remove(lastStroke))
            {
                _maskStrokedRemoved.Add(lastStroke);
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Can undo last mask stroke
        /// </summary>
        private bool CanMaskUndo()
        {
            return MaskCanvas.Strokes.Count > 0;
        }


        /// <summary>
        /// Redo mask stroke
        /// </summary>
        private Task MaskRedo()
        {
            if (_maskStrokedRemoved.Count == 0)
                return Task.CompletedTask;

            var lastStroke = _maskStrokedRemoved.Last();
            if (_maskStrokedRemoved.Remove(lastStroke))
            {
                MaskCanvas.Strokes.Add(lastStroke);
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Can redo mask stroke
        /// </summary>
        private bool CanMaskRedo()
        {
            return _maskStrokedRemoved.Count > 0;
        }


        /// <summary>
        /// Change mask mode
        /// </summary>
        /// <param name="mode">The mode.</param>
        private Task ChangeModeAsync(ImageEditMode mode)
        {
            SelectedEditMode = mode;
            return Task.CompletedTask;
        }


        /// <summary>
        /// Updates the EditMode.
        /// </summary>
        /// <returns>Task.</returns>
        private Task UpdateEditModeAsync()
        {
            MaskEditingMode = InkCanvasEditingMode.None;
            AdornerLayer.GetAdornerLayer(ResizeContainer)?.Remove(_resizeImageAdorner);
            switch (_selectedEditMode)
            {
                case ImageEditMode.None:
                    MaskCanvas.Cursor = Cursors.Arrow;
                    break;
                case ImageEditMode.Draw:
                    MaskCanvas.Cursor = Cursors.Pen;
                    MaskEditingMode = InkCanvasEditingMode.Ink;
                    break;
                case ImageEditMode.Erase:
                    MaskCanvas.Cursor = Cursors.Pen;
                    MaskEditingMode = InkCanvasEditingMode.EraseByPoint;
                    break;
                case ImageEditMode.Resize:
                    MaskCanvas.Cursor = Cursors.SizeAll;
                    AdornerLayer.GetAdornerLayer(ResizeContainer)?.Add(_resizeImageAdorner);
                    break;
                case ImageEditMode.Select:
                    MaskCanvas.Cursor = Cursors.Arrow;
                    break;
                default:
                    break;
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Updates the MaskAttributes.
        /// </summary>
        private void UpdateMaskAttributes()
        {
            var multiplier = CanvasWidth >= 1024 || CanvasHeight >= 1024 ? 2 : 1;
            MaskAttributes = new DrawingAttributes
            {
                Color = MaskColor,
                Height = _maskDrawSize * multiplier,
                Width = _maskDrawSize * multiplier,
            };
        }


        /// <summary>
        /// Resets the canvas.
        /// </summary>
        private void ResetCanvas()
        {
            ResizeContainer.Width = CanvasWidth;
            ResizeContainer.Height = CanvasHeight;
            Canvas.SetLeft(ResizeContainer, 0.0);
            Canvas.SetTop(ResizeContainer, 0.0);
            NotifyPropertyChanged(nameof(IsResizeModified));
        }


        /// <summary>
        /// Called when canvas size changed.
        /// </summary>
        /// <returns>Task.</returns>
        private Task OnSizeChanged()
        {
            if (CanvasWidth > 0 && CanvasHeight > 0)
            {
                ResetCanvas();
                UpdateMaskAttributes();
            }
            return Task.CompletedTask;
        }


        /// <summary>
        /// Called when MaskColor changed.
        /// </summary>
        private Task OnMaskColorChanged()
        {
            UpdateMaskAttributes();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Determines whether if the resize canvas changed.
        /// </summary>
        /// <returns><c>true</c> if [is resize canvas changed]; otherwise, <c>false</c>.</returns>
        private bool IsResizeCanvasChanged()
        {
            if (Source is null || ResizeContainer is null || ImageCanvas is null)
                return false;

            return ResizeContainer.Width != ImageCanvas.Width || ResizeContainer.Height != ImageCanvas.Height || Canvas.GetLeft(ResizeContainer) != 0 || Canvas.GetTop(ResizeContainer) != 0;
        }


        /// <summary>
        /// Creates the selection rectangle.
        /// </summary>
        /// <param name="position">The position.</param>
        private void CreateSelection(Point position)
        {
            ClearSelection();
            var transform = TransformToDescendant(SelectionCanvas);
            var selectionBrush = new SolidColorBrush(SelectionColor);
            _selectionStartPoint = transform.Transform(position);
            _selectionRectangle = new Rectangle
            {
                Width = 1,
                Height = 1
            };

            if (IsSelectionFilled)
            {
                _selectionRectangle.Fill = selectionBrush;
            }
            else
            {
                _selectionRectangle.Stroke = selectionBrush;
                _selectionRectangle.StrokeThickness = _selectionSize;
            }

            Canvas.SetLeft(_selectionRectangle, _selectionStartPoint.X);
            Canvas.SetTop(_selectionRectangle, _selectionStartPoint.Y);
            SelectionCanvas.Children.Add(_selectionRectangle);
        }


        /// <summary>
        /// Updates the selection rectangle.
        /// </summary>
        /// <param name="position">The position.</param>
        private void UpdateSelection(Point position)
        {
            if (_selectionRectangle == null)
                return;

            var transform = TransformToDescendant(SelectionCanvas);
            var canvasPoint = transform.Transform(position);
            double left = Math.Min(_selectionStartPoint.X, canvasPoint.X);
            double top = Math.Min(_selectionStartPoint.Y, canvasPoint.Y);
            double width = Math.Abs(canvasPoint.X - _selectionStartPoint.X);
            double height = Math.Abs(canvasPoint.Y - _selectionStartPoint.Y);

            Canvas.SetLeft(_selectionRectangle, left);
            Canvas.SetTop(_selectionRectangle, top);
            _selectionRectangle.Width = width;
            _selectionRectangle.Height = height;
        }


        /// <summary>
        /// Updates the size of the selection rectangle.
        /// </summary>
        private void UpdateSelectionSize()
        {
            if (_selectionRectangle != null)
                _selectionRectangle.StrokeThickness = _selectionSize;
        }


        /// <summary>
        /// Clears the selection rectangle.
        /// </summary>
        private void ClearSelection()
        {
            if (_selectionRectangle == null)
                return;

            SelectionCanvas.Children.Remove(_selectionRectangle);
            _selectionRectangle = null;
            NotifyPropertyChanged(nameof(IsSelectionModified));
        }


        /// <summary>
        /// Creates the mask image.
        /// </summary>
        /// <param name="image">The image.</param>
        private Task CreateMaskImage(ImageInput image)
        {
            var maskImage = image.Image.ToColorMask(MaskColor);
            MaskCanvas.Background = new ImageBrush(maskImage);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Creates the bitmap source.
        /// </summary>
        private RenderTargetBitmap CreateBitmap(UIElement element)
        {
            return element.CreateBitmap(CanvasWidth, CanvasHeight);
        }


        /// <summary>
        /// MouseEnter
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.Input.MouseEventArgs" /> that contains the event data.</param>
        protected override void OnMouseEnter(MouseEventArgs e)
        {
            if (!IsKeyboardFocusWithin)
                Keyboard.Focus(this);

            base.OnMouseEnter(e);
        }


        /// <summary>
        /// Handles the MouseLeftButtonDown event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Input.MouseButtonEventArgs"/> instance containing the event data.</param>
        private void MaskCanvas_MouseLeftButtonDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            _maskStrokedRemoved.Clear();
            if (_selectedEditMode == ImageEditMode.Select)
            {
                CreateSelection(e.GetPosition(this));
            }
        }


        /// <summary>
        /// Handles the MouseLeftButtonUp event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="System.Windows.Input.MouseButtonEventArgs"/> instance containing the event data.</param>
        private void MaskCanvas_MouseLeftButtonUp(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            NotifyPropertyChanged(nameof(IsMaskModified));
            NotifyPropertyChanged(nameof(IsResizeModified));
            NotifyPropertyChanged(nameof(IsSelectionModified));
        }


        /// <summary>
        /// Handles the StrokeCollected event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="InkCanvasStrokeCollectedEventArgs"/> instance containing the event data.</param>
        private void MaskCanvas_StrokeCollected(object sender, InkCanvasStrokeCollectedEventArgs e)
        {
            NotifyPropertyChanged(nameof(IsMaskModified));
        }


        /// <summary>
        /// Handles the StrokeErased event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="RoutedEventArgs"/> instance containing the event data.</param>
        private void MaskCanvas_StrokeErased(object sender, RoutedEventArgs e)
        {
            NotifyPropertyChanged(nameof(IsMaskModified));
        }


        /// <summary>
        /// Handles the PreviewMouseMove event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseEventArgs"/> instance containing the event data.</param>
        private void MaskCanvas_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Released)
                return;

            if (_selectedEditMode == ImageEditMode.Select)
            {
                if (_selectionRectangle == null)
                    return;

                UpdateSelection(e.GetPosition(this));
            }
        }


        /// <summary>
        /// Handles the OnMouseWheel event of the MaskCanvas control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseWheelEventArgs"/> instance containing the event data.</param>
        private void MaskCanvas_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (_selectedEditMode == ImageEditMode.Select)
            {
                SelectionSize = e.Delta > 0
                    ? Math.Min(100, _selectionSize + 1)
                    : Math.Max(1, _selectionSize - 1);

            }
            else if (_selectedEditMode == ImageEditMode.Draw || _selectedEditMode == ImageEditMode.Erase)
            {
                MaskDrawSize = e.Delta > 0
                    ? Math.Min(100, MaskDrawSize + 1)
                    : Math.Max(1, MaskDrawSize - 1);
            }
        }


        /// <summary>
        /// On Drop
        /// </summary>
        /// <param name="e">The <see cref="T:System.Windows.DragEventArgs" /> that contains the event data.</param>
        protected override async void OnDrop(DragEventArgs e)
        {
            try
            {
                Progress?.Indeterminate();
                var image = await GetDropImage(e);
                if (image == null)
                    return;

                if (IsMaskEnabled)
                {
                    var pos = e.GetPosition(this);
                    var halfWidth = ActualWidth / 2.0;
                    var droppedOnLeft = pos.X < halfWidth;
                    if (droppedOnLeft)
                    {
                        if (!IsLoadEnabled)
                            return;

                        Source = image;
                    }
                    else
                    {
                        if (!IsLoadMaskEnabled)
                            return;

                        SourceMask = image;
                    }
                }
                else
                {
                    if (!IsLoadEnabled)
                        return;

                    Source = image;
                }
            }
            finally
            {
                Progress?.Clear();
            }
        }

    }

    public enum ImageEditMode
    {
        None = 0,
        Draw = 1,
        Erase = 2,
        Resize = 3,
        Select = 4
    }
}
