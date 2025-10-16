// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Windows;
using System.Windows.Documents;
using System.Windows.Input;
using TensorStack.WPF.Adorner;

namespace TensorStack.WPF.Controls
{
    public class WindowMainBase : WindowBase, ILifetimeSingleton
    {
        private int _mouseX;
        private int _mouseY;
        private bool _isDragDrop;
        private DragDropType _dragDropType;
        private DragAdorner _dragAdorner;
        private AdornerLayer _adornerLayer;


        public WindowMainBase()
        {
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
        }


        public bool IsDragDrop
        {
            get { return _isDragDrop; }
            set { SetProperty(ref _isDragDrop, value); }
        }

        public DragDropType DragDropType
        {
            get { return _dragDropType; }
            set { SetProperty(ref _dragDropType, value); }
        }

        public int MouseX
        {
            get { return _mouseX; }
            set
            {
                if (SetProperty(ref _mouseX, value))
                {
                    UpdateDragAdorner();
                }
            }
        }

        public int MouseY
        {
            get { return _mouseY; }
            set
            {
                if (SetProperty(ref _mouseY, value))
                {
                    UpdateDragAdorner();
                }
            }
        }

        protected override void OnPreviewMouseMove(MouseEventArgs e)
        {
            var point = e.GetPosition(this);
            MouseX = (int)point.X;
            MouseY = (int)point.Y;
        }


        protected override void OnGiveFeedback(GiveFeedbackEventArgs e)
        {
            base.OnGiveFeedback(e);
            var point = WindowExtensions.GetMousePosition(this);
            MouseX = (int)point.X;
            MouseY = (int)point.Y;
        }


        private void UpdateDragAdorner()
        {
            if (_dragAdorner != null && _adornerLayer != null)
            {
                _dragAdorner.SetOffset(new Point(MouseX - (_dragAdorner.Size.Width / 2), MouseY - (_dragAdorner.Size.Height / 2)));
            }
        }


        public DragDropEffects DoDragDropFile(DependencyObject dragSource, string filename, DragDropType dataType, UIElement visual = null, double visualScale = 1f)
        {
            var dropData = new DataObject(DataFormats.FileDrop, new[] { filename });
            return DoDragDropData(dragSource, dropData, dataType, visual, visualScale);
        }


        public DragDropEffects DoDragDropObject<T>(DependencyObject dragSource, T dropObject, DragDropType dataType, UIElement visual = null, double visualScale = 1f)
        {
            var dropData = new DataObject(typeof(T), dropObject);
            return DoDragDropData(dragSource, dropData, dataType, visual, visualScale);
        }


        private DragDropEffects DoDragDropData(DependencyObject dragSource, DataObject dropData, DragDropType dataType, UIElement visual = null, double visualScale = 1f)
        {
            try
            {
                DragDropType = dataType;
                IsDragDrop = true;
                OnDragBegin(dataType);
                if (visual != null)
                {
                    var root = Content as UIElement;
                    _adornerLayer = AdornerLayer.GetAdornerLayer(root);
                    if (_adornerLayer != null)
                    {
                        _dragAdorner = new DragAdorner(root, visual, visualScale);
                        _adornerLayer.Add(_dragAdorner);
                    }

                    var dragDropEffects = DragDrop.DoDragDrop(dragSource, dropData, DragDropEffects.Copy | DragDropEffects.Move);
                    if (_adornerLayer != null && _dragAdorner != null)
                        _adornerLayer.Remove(_dragAdorner);

                    _adornerLayer = null;
                    _dragAdorner = null;
                    return dragDropEffects;
                }

                return DragDrop.DoDragDrop(dragSource, dropData, DragDropEffects.Copy | DragDropEffects.Move);
            }
            catch { return DragDropEffects.None; }
            finally
            {
                DragDropType = DragDropType.None;
                IsDragDrop = false;
                OnDragEnd();
            }
        }

        public virtual void OnDragBegin(DragDropType type) { }
        public virtual void OnDragEnd() { }
    }


    public enum DragDropType
    {
        None = 0,
        Text = 1,
        Image = 2,
        Video = 3,
        Audio = 4
    }
}
