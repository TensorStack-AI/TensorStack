// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Input;
using TensorStack.Common;
using TensorStack.WPF.Adorner;

namespace TensorStack.WPF.Controls
{
    public class WindowMainBase : WindowBase, ILifetimeSingleton
    {
        private int _mouseX;
        private int _mouseY;
        private bool _isDragDrop;
        private bool _isExternalDragDrop;
        private DragDropType _dragDropType;
        private DragAdorner _dragAdorner;
        private AdornerLayer _adornerLayer;

        public WindowMainBase()
        {
            AllowDrop = true;
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

            if (e.LeftButton != MouseButtonState.Pressed && _isExternalDragDrop)
            {
                OnExternalDragEnd();
            }
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

        public virtual void OnDragBegin(DragDropType type) { }
        public virtual void OnDragEnd() { }


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
                    _adornerLayer ??= AdornerLayer.GetAdornerLayer(root);
                    if (_adornerLayer != null && _dragAdorner == null)
                    {
                        _dragAdorner = new DragAdorner(root, visual, visualScale);
                        _adornerLayer.Add(_dragAdorner);
                    }
                }

                return DragDrop.DoDragDrop(dragSource, dropData, DragDropEffects.Copy | DragDropEffects.Move);
            }
            catch { return DragDropEffects.None; }
            finally
            {
                ClearAdorner();

                DragDropType = DragDropType.None;
                IsDragDrop = false;
                OnDragEnd();
            }
        }


        protected override void OnPreviewDragEnter(DragEventArgs e)
        {
            if (IsDragDrop)
                return;

            if (e.Source == this && _isExternalDragDrop == false)
            {
                var fileNames = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (fileNames.IsNullOrEmpty())
                    return;

                OnExternalDragBegin(fileNames[0]);

                e.Effects = DragDropEffects.None;
            }
            base.OnPreviewDragEnter(e);
        }


        protected override void OnPreviewDragLeave(DragEventArgs e)
        {
            var point = e.GetPosition(this);
            if (point.X == 0 && point.Y == 0 && _isExternalDragDrop)
            {
                OnExternalDragEnd();
            }
            base.OnPreviewDragLeave(e);
        }


        private void OnExternalDragBegin(string filename)
        {
            var extension = Path.GetExtension(filename);
            if (Common.ImageFileExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            {
                DragDropType = DragDropType.Image;
            }
            else if (Common.VideoFileExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            {
                DragDropType = DragDropType.Video;
            }
            else if (Common.TextFileExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            {
                DragDropType = DragDropType.Text;
            }
            else if (Common.AudioFileExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            {
                DragDropType = DragDropType.Audio;
            }

            if (DragDropType != DragDropType.None)
            {
                _isExternalDragDrop = true;

                IsDragDrop = true;
                OnDragBegin(DragDropType);
                Debug.WriteLine($"OnExternalDragBegin: {DragDropType}");
            }
        }


        private void OnExternalDragEnd()
        {
            DragDropType = DragDropType.None;
            IsDragDrop = false;
            OnDragEnd();
            _isExternalDragDrop = false;
            Debug.WriteLine("OnExternalDragEnd");
        }


        private void ClearAdorner()
        {
            _dragAdorner = null;
            if (_adornerLayer == null)
                return;

            var root = Content as UIElement;
            var adorners = _adornerLayer.GetAdorners(root);
            if (adorners.IsNullOrEmpty())
                return;

            foreach (var adorner in adorners)
            {
                _adornerLayer.Remove(adorner);
            }
        }
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
