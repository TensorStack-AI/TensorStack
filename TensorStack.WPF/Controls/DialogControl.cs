// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Animation;

namespace TensorStack.WPF.Controls
{
    public class DialogControl : Window, INotifyPropertyChanged, System.Windows.Forms.IWin32Window
    {
        private readonly WindowInteropHelper _interopHelper;

        public DialogControl()
        {
            _interopHelper = new WindowInteropHelper(this);
            AllowsTransparency = true;
            SnapsToDevicePixels = true;
            WindowStyle = WindowStyle.None;
            SizeToContent = SizeToContent.WidthAndHeight;
            WindowStartupLocation = WindowStartupLocation.CenterOwner;

            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.Fant);
            RenderOptions.SetClearTypeHint(this, ClearTypeHint.Enabled);
            TextOptions.SetTextFormattingMode(this, TextFormattingMode.Ideal);
            TextOptions.SetTextRenderingMode(this, TextRenderingMode.ClearType);
            TextOptions.SetTextHintingMode(this, TextHintingMode.Fixed);

            CloseCommand = new AsyncRelayCommand(CloseAsync);
            Loaded += (s, e) => CreateOpenAnimation();
        }

        public nint Handle => _interopHelper.Handle;
        public AsyncRelayCommand CloseCommand { get; }
        public WindowBase OwnerWindow => Owner as WindowBase;


        public virtual new bool ShowDialog()
        {
            Opacity = 0;
            return base.ShowDialog() ?? false;
        }


        public virtual Task<bool> ShowDialogAsync()
        {
            Opacity = 0;
            return Task.FromResult(ShowDialog());
        }


        protected virtual Task CloseAsync()
        {
            Close();
            OwnerWindow.IsDialogVisible = false;
            return Task.CompletedTask;
        }


        protected virtual Task SaveAsync()
        {
            return CreateCloseAnimation(true);
        }


        protected virtual bool CanExecuteSave()
        {
            return true;
        }


        protected virtual Task CancelAsync()
        {
            return CreateCloseAnimation(false);
        }


        protected virtual bool CanExecuteCancel()
        {
            return true;
        }


        protected override void OnSourceInitialized(EventArgs e)
        {
            base.OnSourceInitialized(e);
            _interopHelper.EnsureHandle();
            this.RegisterDisplayMonitor();
        }


        private void CreateOpenAnimation()
        {
            OwnerWindow.IsDialogVisible = true; // Dialog Open
            var fadeInAnimation = new DoubleAnimation
            {
                From = 0,
                To = 1,
                Duration = new Duration(TimeSpan.FromMilliseconds(200)),
                FillBehavior = FillBehavior.Stop,
                EasingFunction = new ExponentialEase() { EasingMode = EasingMode.EaseOut }
            };
            fadeInAnimation.Completed += (s, e) => { Opacity = 1; };
            BeginAnimation(OpacityProperty, fadeInAnimation);
        }


        private Task CreateCloseAnimation(bool dialogResult)
        {
            OwnerWindow.IsDialogVisible = false;  // Dialog Closed
            var tcs = new TaskCompletionSource();
            var fadeOutAnimation = new DoubleAnimation
            {
                From = 1,
                To = 0,
                Duration = new Duration(TimeSpan.FromMilliseconds(200)),
                FillBehavior = FillBehavior.Stop,
                EasingFunction = new ExponentialEase() { EasingMode = EasingMode.EaseOut }
            };

            fadeOutAnimation.Completed += (s, e) =>
            {
                try
                {
                    DialogResult = dialogResult;
                }
                finally
                {
                    tcs.SetResult();
                }
            };
            BeginAnimation(OpacityProperty, fadeOutAnimation);
            return tcs.Task;
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


    public readonly record struct DialogResult
    {
        private readonly bool _result;
        private readonly bool _dontAskAgain;

        public DialogResult(bool result, bool dontAskAgain)
        {
            _result = result;
            _dontAskAgain = dontAskAgain;
        }

        public readonly bool Result => _result;
        public readonly bool DontAskAgain => _dontAskAgain;

        public static implicit operator bool(DialogResult dialogResult)
        {
            return dialogResult._result;
        }

        public static implicit operator DialogResult(bool result)
        {
            return new DialogResult(result, false);
        }

        public static bool operator true(DialogResult dialogResult)
        {
            return dialogResult._result;
        }

        public static bool operator false(DialogResult dialogResult)
        {
            return !dialogResult._result;
        }
    }
}
