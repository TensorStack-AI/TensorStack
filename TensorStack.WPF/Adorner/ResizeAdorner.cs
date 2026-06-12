using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace TensorStack.WPF.Adorner
{
    public class ResizeAdorner : System.Windows.Documents.Adorner
    {
        private readonly VisualCollection _visuals;
        private readonly Thumb _topLeft, _topRight, _bottomLeft, _bottomRight, _border;
        private Point _startPoint;
        private bool _isDragging;
        private Action<bool> _refreshCallback;
        private float _scale = 0f;

        public ResizeAdorner(UIElement adornedElement, bool resize, float scale, Action<bool> refreshCallback = default) : base(adornedElement)
        {
            _scale = scale;
            IsHitTestVisible = true;
            MouseLeftButtonDown += MoveAdorner_MouseLeftButtonDown;
            MouseMove += MoveAdorner_MouseMove;
            MouseLeftButtonUp += MoveAdorner_MouseLeftButtonUp;
            _visuals = new VisualCollection(this);
            _topLeft = CreateThumb(Cursors.SizeNWSE);
            _topRight = CreateThumb(Cursors.SizeNESW);
            _bottomLeft = CreateThumb(Cursors.SizeNESW);
            _bottomRight = CreateThumb(Cursors.SizeNWSE);
            _border = CreateBorder();
            _visuals.Add(_border);
            if (resize)
            {
                _topLeft.DragDelta += (s, e) => Resize(e.HorizontalChange, e.VerticalChange, true, true);
                _topRight.DragDelta += (s, e) => Resize(e.HorizontalChange, e.VerticalChange, false, true);
                _bottomLeft.DragDelta += (s, e) => Resize(e.HorizontalChange, e.VerticalChange, true, false);
                _bottomRight.DragDelta += (s, e) => Resize(e.HorizontalChange, e.VerticalChange, false, false);
                _visuals.Add(_topLeft);
                _visuals.Add(_topRight);
                _visuals.Add(_bottomLeft);
                _visuals.Add(_bottomRight);
            }
            _refreshCallback = refreshCallback;
        }

        protected override int VisualChildrenCount => _visuals.Count;
        protected override Visual GetVisualChild(int index) => _visuals[index];

        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(RenderSize));
        }


        protected override Size ArrangeOverride(Size finalSize)
        {
            if (AdornedElement is FrameworkElement adorned)
            {
                double offset = -_topLeft.Width / 2;

                _border.Width = finalSize.Width;
                _border.Height = finalSize.Height;
                _border.Arrange(new Rect(0, 0, finalSize.Width, finalSize.Height));

                _topLeft.Arrange(new Rect(offset, offset, _topLeft.Width, _topLeft.Height));
                _topRight.Arrange(new Rect(adorned.ActualWidth - offset - _topRight.Width, offset, _topRight.Width, _topRight.Height));
                _bottomLeft.Arrange(new Rect(offset, adorned.ActualHeight - offset - _bottomLeft.Height, _bottomLeft.Width, _bottomLeft.Height));
                _bottomRight.Arrange(new Rect(adorned.ActualWidth - offset - _bottomRight.Width, adorned.ActualHeight - offset - _bottomRight.Height, _bottomRight.Width, _bottomRight.Height));
            }
            return finalSize;
        }


        protected override void OnMouseEnter(MouseEventArgs e)
        {
            Cursor = Cursors.SizeAll;
            foreach (var thumb in _visuals.OfType<Thumb>())
            {
                thumb.Visibility = Visibility.Visible;
            }
            base.OnMouseEnter(e);
        }


        protected override void OnMouseLeave(MouseEventArgs e)
        {
            Cursor = null;
            foreach (var thumb in _visuals.OfType<Thumb>())
            {
                thumb.Visibility = Visibility.Hidden;
            }
            base.OnMouseLeave(e);
        }


        private void Resize(double deltaX, double deltaY, bool adjustLeft, bool adjustTop)
        {
            if (AdornedElement is not FrameworkElement element)
                return;

            double newWidth = element.Width + deltaX * (adjustLeft ? -1 : 1);
            double newHeight = element.Height + deltaY * (adjustTop ? -1 : 1);
            newWidth = Math.Max(newWidth, 10);
            newHeight = Math.Max(newHeight, 10);
            if (adjustLeft)
                Canvas.SetLeft(element, Canvas.GetLeft(element) + (element.Width - newWidth));
            if (adjustTop)
                Canvas.SetTop(element, Canvas.GetTop(element) + (element.Height - newHeight));

            element.Width = newWidth;
            element.Height = newHeight;
            InvalidateArrange();
        }


        private void MoveAdorner_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _startPoint = Mouse.GetPosition(this);
            _isDragging = true;
            CaptureMouse();
            e.Handled = true;
        }


        private void MoveAdorner_MouseMove(object sender, MouseEventArgs e)
        {
            var currentPoint = Mouse.GetPosition(this);
            if (_isDragging && AdornedElement is FrameworkElement element)
            {
                var left = Canvas.GetLeft(element);
                var top = Canvas.GetTop(element);
                var deltaX = _startPoint.X - currentPoint.X;
                var deltaY = _startPoint.Y - currentPoint.Y;
                Canvas.SetLeft(element, left - deltaX);
                Canvas.SetTop(element, top - deltaY);
                _refreshCallback?.Invoke(true);
            }
        }


        private void MoveAdorner_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (_isDragging)
            {
                _isDragging = false;
                ReleaseMouseCapture();
                e.Handled = true;
                _refreshCallback?.Invoke(false);
            }
        }


        private void Thumb_MouseLeftButtonUp(object sender, MouseEventArgs e)
        {
            _refreshCallback?.Invoke(false);
        }


        private Thumb CreateThumb(Cursor cursor)
        {
            var scale = GetScale();
            var borderSize = 1 * scale;
            var ellipseSize = 10.0 * scale;
            var ellipseFactory = new FrameworkElementFactory(typeof(Ellipse));
            ellipseFactory.SetValue(Shape.FillProperty, Brushes.Red);
            ellipseFactory.SetValue(Shape.StrokeProperty, Brushes.Black);
            ellipseFactory.SetValue(Shape.StrokeThicknessProperty, borderSize);
            ellipseFactory.SetValue(FrameworkElement.WidthProperty, ellipseSize);
            ellipseFactory.SetValue(FrameworkElement.HeightProperty, ellipseSize);

            var thumb = new Thumb
            {
                Width = ellipseSize,
                Height = ellipseSize,
                Cursor = cursor,
                Visibility = Visibility.Hidden,
                Template = new ControlTemplate(typeof(Thumb))
                {
                    VisualTree = ellipseFactory
                }
            };

            thumb.PreviewMouseUp += Thumb_MouseLeftButtonUp;
            return thumb;
        }


        private Thumb CreateBorder()
        {
            var borderSize = GetScale();
            var rectangleFactory = new FrameworkElementFactory(typeof(Rectangle));
            rectangleFactory.SetValue(Shape.StrokeProperty, Brushes.Red);
            rectangleFactory.SetValue(Shape.StrokeThicknessProperty, borderSize);
            var thumb = new Thumb
            {
                IsHitTestVisible = false,
                Visibility = Visibility.Hidden,
                Template = new ControlTemplate(typeof(Thumb))
                {
                    VisualTree = rectangleFactory
                }
            };
            return thumb;
        }


        private double GetScale()
        {
            if (_scale > 0)
                return _scale;

            var scale = 1.0;
            if (AdornedElement is FrameworkElement adorned && adorned.Parent is FrameworkElement parent)
            {
                scale *= Math.Max(parent.Width, parent.Height) / 512;
            }
            return scale;
        }

    }
}
