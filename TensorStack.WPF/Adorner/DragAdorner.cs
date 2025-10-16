// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Windows;
using System.Windows.Media;


namespace TensorStack.WPF.Adorner
{
    public class DragAdorner : System.Windows.Documents.Adorner
    {
        private readonly VisualBrush _brush;
        private Point _offset;
        private readonly Size _size;

        public DragAdorner(UIElement adornedElement, UIElement adornVisual, double scale = 1f)
            : base(adornedElement)
        {
            _brush = new VisualBrush(adornVisual)
            {
                Opacity = 0.7
            };

            _size = new Size(adornVisual.RenderSize.Width / scale, adornVisual.RenderSize.Height / scale);

            IsHitTestVisible = false; // ignore mouse events
        }

        public Size Size => _size;

        public void SetOffset(Point offset)
        {
            _offset = offset;
            InvalidateVisual(); // redraw at new position
        }

        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawRectangle(_brush, null, new Rect(_offset, _size));
        }
    }
}
