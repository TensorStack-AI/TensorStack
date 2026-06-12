// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace TensorStack.WPF.Controls
{
    public class WindowBase : Window, INotifyPropertyChanged, System.Windows.Forms.IWin32Window
    {
        private readonly WindowInteropHelper _interopHelper;
        private bool _isDialogVisible;

        public WindowBase()
        {
            _interopHelper = new WindowInteropHelper(this);
            AllowsTransparency = true;
            SnapsToDevicePixels = true;
            WindowStyle = WindowStyle.None;
            WindowStartupLocation = WindowStartupLocation.CenterOwner;
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.Fant);
            RenderOptions.SetClearTypeHint(this, ClearTypeHint.Enabled);
            TextOptions.SetTextFormattingMode(this, TextFormattingMode.Ideal);
            TextOptions.SetTextRenderingMode(this, TextRenderingMode.ClearType);
            TextOptions.SetTextHintingMode(this, TextHintingMode.Fixed);

            CloseCommand = new AsyncRelayCommand(CloseAsync);
            RestoreCommand = new AsyncRelayCommand(RestoreAsync);
            MinimizeCommand = new AsyncRelayCommand(MinimizeAsync);
            MaximizeCommand = new AsyncRelayCommand(MaximizeAsync);
        }

        public nint Handle => _interopHelper.Handle;
        public AsyncRelayCommand MinimizeCommand { get; }
        public AsyncRelayCommand RestoreCommand { get; }
        public AsyncRelayCommand MaximizeCommand { get; }
        public AsyncRelayCommand CloseCommand { get; }

        public bool IsDialogVisible
        {
            get { return _isDialogVisible; }
            set { _isDialogVisible = value; NotifyPropertyChanged(); }
        }

        protected override void OnSourceInitialized(EventArgs e)
        {
            base.OnSourceInitialized(e);
            _interopHelper.EnsureHandle();
            this.RegisterDisplayMonitor();
        }


        protected virtual Task CloseAsync()
        {
            Close();
            return Task.CompletedTask;
        }


        protected virtual Task RestoreAsync()
        {
            if (WindowState == WindowState.Maximized)
                WindowState = WindowState.Normal;
            else
                WindowState = WindowState.Maximized;
            return Task.CompletedTask;
        }


        protected virtual Task MinimizeAsync()
        {
            WindowState = WindowState.Minimized;
            return Task.CompletedTask;
        }


        protected virtual Task MaximizeAsync()
        {
            WindowState = WindowState.Maximized;
            return Task.CompletedTask;
        }


        #region INotifyPropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;
        public void NotifyPropertyChanged([CallerMemberName] string property = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }

        protected bool SetProperty<T>(ref T field, T newValue, [CallerMemberName] string propertyName = null)
        {
            if (!Equals(field, newValue))
            {
                field = newValue;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
                return true;
            }

            return false;
        }
        #endregion
    }
}
